from re import S
import numpy as np
import torch
import torch.optim as optim
import logging, math, atexit, os, json
from ..utils.util import ensure_dir
from ..utils.shift_utils import StoreIntermediateTensore, SweepDelta
from ..utils.event_tensor_utils import EventPreprocessor
from torch.nn import ReflectionPad2d
from kornia.filters.sobel import spatial_gradient, sobel

def caliberate_with_dmax(out,dmax=80,alpha=-3.7):
        #print(out.shape)
        out=out[0,0,2:262,3:349]
        out=torch.mul(out,-1)
        out=torch.add(out,1)
        out=torch.mul(out,alpha)
        out=torch.exp(out)
        out=torch.mul(out,dmax)
        out=out.double()
        out=torch.where(out>dmax-0.1,float(1000),out)
        #print(out.shape)
        return out

def scale_invariant_loss(y_input, y_target, weight = 1.0, n_lambda = 1.0):
        
        log_diff = y_input - y_target
        is_nan = torch.isnan(log_diff)
        return weight * ((log_diff[~is_nan]**2).mean()-(n_lambda*(log_diff[~is_nan].mean())**2))

class MultiScaleGradient(torch.nn.Module):
    def __init__(self, start_scale = 1, num_scales = 4):
        super(MultiScaleGradient,self).__init__()
        print('Setting up Multi Scale Gradient loss...')

        self.start_scale = start_scale
        self.num_scales = num_scales
        # for scale in range(self.num_scales):
        #     print(self.start_scale * (2**scale), self.start_scale * (2**scale))

        self.multi_scales = [torch.nn.AvgPool2d(self.start_scale * (2**scale), self.start_scale * (2**scale)) for scale in range(self.num_scales)]
        print('Done')

    def forward(self, prediction, target, preview = False):
        # helper to remove potential nan in labels
        def nan_helper(y):
            return torch.isnan(y), lambda z: z.nonzero()[0]
        
        loss_value = 0
        loss_value_2 = 0
        diff = prediction - target
        H,W = target.shape
        upsample = torch.nn.Upsample(size=(2*H,2*W), mode='bicubic', align_corners=True)
        record = []

        for m in self.multi_scales:
            # input and type are of the type [B x C x H x W]
            if preview:
                record.append(upsample(sobel(m(diff))))
            else:
                # Use kornia spatial gradient computation
                diff_unsq=torch.unsqueeze(diff,0)
                diff_unsq_uns=torch.unsqueeze(diff_unsq,0)
                m_diff=m(diff_unsq_uns)
                
                delta_diff = spatial_gradient(m_diff)
                is_nan = torch.isnan(delta_diff)
                is_not_nan_sum = (~is_nan).sum()
                # output of kornia spatial gradient is [B x C x 2 x H x W]
                loss_value += torch.abs(delta_diff[~is_nan]).sum()/is_not_nan_sum*target.shape[0]*2
                # * batch size * 2 (because kornia spatial product has two outputs).
                # replaces the following line to be able to deal with nan's.
                # loss_value += torch.abs(delta_diff).mean(dim=(3,4)).sum()

        if preview:
            return record
        else:
            return (loss_value/self.num_scales)


multi_scale_grad_loss_fn = MultiScaleGradient()


def multi_scale_grad_loss(prediction, target, preview = False):
    return multi_scale_grad_loss_fn.forward(prediction, target, preview)

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
        self.su = StoreIntermediateTensore([
            self.model.unetrecurrent.encoders[0].fence,
            self.model.unetrecurrent.encoders[2].fence
        ])
        self.states_in_validation = None

        class options:
            hot_pixels_file= None
            flip = False
            no_normalize = False
        self.event_preprocessor = EventPreprocessor(options)
        self.pad = ReflectionPad2d((3, 3, 2, 2)) # left, right, top, bottom: works according to this configuation



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

    
    
    
    def calculate_loss(self, predicted_target, target,lamb=0.5):

        calib_pred = caliberate_with_dmax(predicted_target)
        target=target[0,0,:,:]

        scale_inv_loss = scale_invariant_loss(calib_pred,target)

        multi_sc_loss=multi_scale_grad_loss(calib_pred,target)

        total = scale_inv_loss+(lamb * multi_sc_loss)

        #print(total)
        return total
        # if self.loss_params is not None:
        #     reconstruction_loss = self.loss(predicted_target, target, **self.loss_params)
        # else:
        #     reconstruction_loss = self.loss(predicted_target, target)
        # contrast_loss = self.weight_contrast_loss * torch.pow(predicted_target.std() - target.std(), 2)
        # return reconstruction_loss + contrast_loss

    def forward_pass_sequence(self, sequence):

        L = len(sequence)
        assert (L > 0)

        loss = 0
        prev_states_lstm = None
        for l in range(L):
            item = sequence[l]
            events = item['events0'].to(self.gpu)

            event_frame = self.pad(self.event_preprocessor(events))

            predicted_target, new_states_lstm = self.model(event_frame, prev_states_lstm)
            prev_states_lstm = new_states_lstm

            target = item['depth_events0'].to(self.gpu)
            #print("\n***\n target dim: ",target.size(),"\n predicted dim: ",predicted_target.size(),"\n***")
            loss += self.calculate_loss(predicted_target, target)
#            total_metrics += self._eval_metrics(predicted_target, target)

        return loss

    def forward_for_valid(self,sequence,states=None):
        # input_struct in experiments.py is analogous to sequence
        if states==None:
            states=self.states_in_validation
        events=self.pad(self.event_preprocessor(sequence))
        events=events.cuda()
       

        prediction,states=self.model(events,states)
        print("seq size :", len(sequence))
        return prediction,states


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
                pass
                # self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.3f}, L_r: {:.3f}, L_contrast: {:.3f}'.format(
                #     epoch,
                #     batch_idx * self.data_loader.batch_size,
                #     len(self.data_loader) * self.data_loader.batch_size,
                #     100.0 * batch_idx / len(self.data_loader),
                #     loss.item()))
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
    

    def infer(self,su):
        sweep_del=SweepDelta()
        infer_list=[]
        for module in su.store_tensors:
            tensors=su.store_tensors[module]
            diffs,numels=sweep_del.num_operations(tensors)
            infer_list.append(diffs)
        
        return infer_list


    def _valid_epoch(self):
        
        self.model.eval()
        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        
        with torch.no_grad():
            for sequence in self.valid_data_loader:
                self.su.register_hooks()
                prediction,states= self.forward_for_valid(sequence)
                self.su.deregister_hooks()
                self.states_in_validation=states
                print("for loss",sequence.size())
                #loss=self.calculate_loss(prediction,)
                
               


            infered_list = self.infer(self.su) 
            
            # TODO: need to add metrics

        return {
            'val_loss':0, #total_val_loss / len(self.valid_data_loader),
            'val_metrics': 0,#(total_val_metrics / len(self.valid_data_loader)).tolist(),
        }


if __name__ == '__main__':

    pass

