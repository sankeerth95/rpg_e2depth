import torch
from .trainer import E2DEPTHTrainer
from utils.shift_utils import StoreIntermediateTensors
import numpy as np

def perturbation_shift_loss(t_1: 'dict', t_2: 'dict', device='cuda'):

    loss = torch.zeros((1)).to(device)
    for key in t_1.keys():
        t1 = [ t[0] for t in t_1[key]]
        t2 = [ t[0] for t in t_2[key]]
        loss += torch.mean(torch.abs(
            torch.stack( t1 ) - torch.stack( t2 )
        )**2 )
    return loss


class E2DepthPerturbationTuner(E2DEPTHTrainer):

    def __init__(self, model, loss, loss_params, metrics, resume, config,
                 data_loader, valid_data_loader=None, train_logger=None):
        super().__init__(model, loss, loss_params, metrics, resume, config,
                 data_loader, valid_data_loader=valid_data_loader, train_logger=train_logger)
        self.su1 = StoreIntermediateTensors([
            self.model.unetrecurrent.encoders[0].recurrent_block.Gates,
            self.model.unetrecurrent.encoders[2].recurrent_block.Gates
        ])
        self.su2 = StoreIntermediateTensors([
            self.model.unetrecurrent.encoders[0].recurrent_block.Gates,
            self.model.unetrecurrent.encoders[2].recurrent_block.Gates
        ])

    def infer(self):
        sweep_del = SweepDelta()
        infer_list = []
        for module in self.su1.store_tensors:
            tensors = self.su1.store_tensors[module]
            diffs, numels = sweep_del.num_operations(tensors)
            infer_list.append(diffs)
        return infer_list

    def _valid_epoch(self):

        self.model.eval()
        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        
        with torch.no_grad():
            for sequence in self.valid_data_loader:
                self.su1.register_hooks()
                prediction,states= self.forward_for_valid(sequence)
                self.su1.deregister_hooks()
                self.states_in_validation=states
                print("for loss",sequence.size())
                #loss=self.calculate_loss(prediction,)

            infered_list = self.infer() 
            # TODO: need to add metrics

        return {
            'val_loss':0, #total_val_loss / len(self.valid_data_loader),
            'val_metrics': 0,#(total_val_metrics / len(self.valid_data_loader)).tolist(),
        }


    def forward_pass_sequence(self, sequence):
        L = len(sequence)
        assert (L > 0)

        loss = 0
        prev_states_lstm = None
        for l in range(L):
            item = sequence[l]

            self.su1.register_hooks()
            events0 = item['events0'].to(self.gpu)
            event_frame0 = self.pad(self.event_preprocessor(events0))
            predicted_target, new_states_lstm = self.model(event_frame0, prev_states_lstm)
            self.su1.deregister_hooks()


            self.su2.register_hooks()
            events1 = item['shifted_events0'].to(self.gpu)
            event_frame1 = self.pad(self.event_preprocessor(events1))
            predicted_target_, new_states_lstm = self.model(event_frame1, prev_states_lstm)
            self.su2.deregister_hooks()
 
            target = item['depth_image'].to(self.gpu)

            prev_states_lstm = new_states_lstm

        loss += self.calculate_loss(self.su1.store_tensors, self.su2.store_tensors)
        self.su1.clear_tensors()
        self.su2.clear_tensors()

        return loss


    def calculate_loss(self, t, t_shift):
        total = perturbation_shift_loss(t, t_shift)
        return total

    
    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        for batch_idx, sequence in enumerate(self.data_loader):


            self.optimizer.zero_grad()
            loss = self.forward_pass_sequence(sequence)
            print(loss)
            loss.backward()
            self.optimizer.step()

            total_loss += loss
            # TODO: need to add metrics as well: total_metrics

            if batch_idx % self.log_step == 0:
                self.logger.info(f'Train Epoch: {epoch} \
                    [{batch_idx * self.data_loader.batch_size}\
                        /{len(self.data_loader) * self.data_loader.batch_size} \
                    ({100.0 * batch_idx / len(self.data_loader):.2f}%)] Loss: , L_r: , L_contrast: ')

        log = {
            'loss': total_loss / len(self.data_loader),
            'metrics': (total_metrics / len(self.data_loader)).tolist(),
        }

        if self.valid:
            val_log = self._valid_epoch()
            log = {**log, **val_log}

        return log



