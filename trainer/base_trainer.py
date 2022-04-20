import torch
import logging
import torch.optim as optim
import os, json, atexit
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

        self.gpu = torch.device('cuda:' + str(config['gpu']))
        self.model = self.model.to(self.gpu)

        self.train_logger = train_logger
        self.optimizer = getattr(optim, config["trainer"]['optimizer_type'])(model.parameters(),
                                                                  **config["trainer"]['optimizer'])

        # lr scheduler
        self.lr_scheduler = getattr( optim.lr_scheduler, config["trainer"]['lr_scheduler_type'], None)
        if self.lr_scheduler:
            self.lr_scheduler = self.lr_scheduler(self.optimizer, **config["trainer"]['lr_scheduler'])
            self.lr_scheduler_freq = config["trainer"]['lr_scheduler_freq']

        self.start_epoch = 1

        self.checkpoint_dir = os.path.join(config['trainer']['save_dir'], self.name)
        ensure_dir(self.checkpoint_dir)
        json.dump(config, open(os.path.join(self.checkpoint_dir, 'config.json'), 'w'),
                  indent=4, sort_keys=False)

        atexit.register(self.cleanup)
        if resume:
            self._resume_checkpoint(resume)        

    def cleanup(self):
        pass

    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            print(epoch)
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
                                .format(epoch, float(log['loss'].detach().cpu().numpy()))  )
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))

    def _resume_checkpoint(self, resume_path):
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        print("Loading checkpoint: {} ...".format(resume_path))
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

