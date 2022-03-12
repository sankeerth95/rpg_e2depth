import numpy as np
import torch
import torch.optim as optim
import logging, math, atexit, os, json
from ..utils.util import ensure_dir

class BaseTrainer:


    def __init__(self, model, loss, loss_params, metrics, resume, config, train_logger=None):
        self.config = config
        self.model = model
        self.loss = loss
        self.loss_params = loss_params
        self.metrics = metrics

        self.logger = logging.getLogger(self.__class__.__name__)
        self.name = config['name']
        self.epochs = config['trainer']['epochs']
        self.save_freq = config['trainer']['save_freq']

        self.with_cuda = config['cuda'] and torch.cuda.is_available()
        if config['cuda'] and not torch.cuda.is_available():
            self.logger.warning('Warning: There\'s no CUDA support on this machine, '
                                'training is performed on CPU.')
        else:
            self.gpu = torch.device('cuda:' + str(config['gpu']))
            self.model = self.model.to(self.gpu)

        self.train_logger = train_logger
        self.optimizer = getattr(optim, config['optimizer_type'])(model.parameters(),
                                                                  **config['optimizer'])

        # lr scheduler
        self.lr_scheduler = getattr( optim.lr_scheduler, config['lr_scheduler_type'], None)
        if self.lr_scheduler:
            self.lr_scheduler = self.lr_scheduler(self.optimizer, **config['lr_scheduler'])
            self.lr_scheduler_freq = config['lr_scheduler_freq']

        self.start_epoch = 1

        self.checkpoint_dir = os.path.join(config['trainer']['save_dir'], self.name)
        ensure_dir(self.checkpoint_dir)
        json.dump(config, open(os.path.join(self.checkpoint_dir, 'config.json'), 'w'),
                  indent=4, sort_keys=False)

        atexit.register(self.cleanup)
        if resume:
            self._resume_checkpoint(resume)        

    def cleanup(self):
        self.writer.close()

    def train(self):

        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)
            log = {'epoch': epoch, 'loss': result['loss']}
            if epoch % self.save_freq == 0:
                self._save_checkpoint(epoch, log)


    def _train_epoch(self, epoch):
        raise NotImplementedError

    def _save_checkpoint(self, epoch, log):

        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'logger': self.train_logger,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config
        }
        filename = os.path.join(self.checkpoint_dir, 'checkpoint-epoch{:03d}-loss-{:.4f}.pth.tar'
                                .format(epoch, log['loss']))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))

    def _resume_checkpoint(self, resume_path):
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.with_cuda:
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda(self.gpu)
        self.train_logger = checkpoint['logger']
        #self.config = checkpoint['config']
        self.logger.info("Checkpoint '{}' (epoch {}) loaded".format(resume_path, self.start_epoch))



class E2DEPTHTrainer(BaseTrainer):

    def __init__(self, model, loss, loss_params, metrics, resume, config,
                 data_loader, valid_data_loader=None, train_logger=None):
        super().__init__(model, loss, loss_params, metrics, resume, config, train_logger)
        self.config = config
        self.batch_size = data_loader.batch_size
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.valid = True if self.valid_data_loader is not None else False
        self.log_step = int(np.sqrt(self.batch_size))

        try:
            self.weight_contrast_loss = config['weight_contrast_loss']
            print('Will use contrast loss with weight={:.2f}'.format(self.weight_contrast_loss))
        except KeyError:
            print('Will not use contrast loss')
            self.weight_contrast_loss = 0


    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros(len(self.metrics))
        output = output.cpu().data.numpy()
        target = target.cpu().data.numpy()
        # output = np.argmax(output, axis=1)
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target)
        return acc_metrics

    def calculate_loss(self, predicted_target, target):
        if self.loss_params is not None:
            reconstruction_loss = self.loss(predicted_target, target, **self.loss_params)
        else:
            reconstruction_loss = self.loss(predicted_target, target)
        contrast_loss = self.weight_contrast_loss * torch.pow(predicted_target.std() - target.std(), 2)
        return reconstruction_loss + contrast_loss


    def forward_pass_sequence(self, sequence):

        L = len(sequence)
        assert (L > 0)

        prev_states_lstm = {}
        for k in range(0, self.every_x_rgb_frame):
            prev_states_lstm['events{}'.format(k)] = None
            prev_states_lstm['depth{}'.format(k)] = None

        loss = 0
        for l in range(L):
            item = sequence[l]
            predicted_target, new_states_lstm = self.model(item, prev_states_lstm)
            prev_states_lstm = new_states_lstm

            target = item['depth_' + l].to(self.gpu)

            loss += self.calculate_loss(target, predicted_target)
#            total_metrics += self._eval_metrics(predicted_target, target)

        return loss


    def _train_epoch(self, epoch):
        self.model.train()

        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        for batch_idx, sequence in enumerate(self.data_loader):

            self.optimizer.zero_grad()
            loss = self.forward_pass_sequence(sequence)
            loss.backward()
            self.optimizer.step()


            total_loss += loss
            # TODO: need to add metrics as well: total_metrics

            if batch_idx % self.log_step == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.3f}, L_r: {:.3f}, L_contrast: {:.3f}'.format(
                    epoch,
                    batch_idx * self.data_loader.batch_size,
                    len(self.data_loader) * self.data_loader.batch_size,
                    100.0 * batch_idx / len(self.data_loader),
                    loss.item()))
                    # reconstruction_loss.item(),
                    # contrast_loss.item()))

        log = {
            'loss': total_loss / len(self.data_loader),
            'metrics': (total_metrics / len(self.data_loader)).tolist(),
        }

        if self.valid:
            val_log = self._valid_epoch()
            log = {**log, **val_log}

        return log

    def _valid_epoch(self):

        self.model.eval()
        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))

        with torch.no_grad():
            for batch_idx, sequence in enumerate(self.valid_data_loader):

                total_val_loss += self.forward_pass_sequence(sequence)
            # TODO: need to add metrics

        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist(),
        }


if __name__ == '__main__':

    pass

