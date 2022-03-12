import json, os ,argparse, logging
import torch
import torch.nn as nn
from data_fetchers.EventDataUtilities.dataset import SequenceSynchronizedFramesEventsDataset
from ev_projs.rpg_e2depth.model.model import E2VIDRecurrent
from trainer import trainer

from torch.utils.data import DataLoader


def main(config):

    # model
    model = E2VIDRecurrent(config)

    # loss
    loss = nn.MSELoss()
    loss_params = None

    # metrics
    metrics = []

    #e2depth train dataloader
    dataset = SequenceSynchronizedFramesEventsDataset(base_folder='./data/test/', event_folder='events/data/', depth_folder = 'depth/data/', dataset_type='voxeltrain')
    dataloader = DataLoader(dataset)
    # validation_dataset = SequenceSynchronizedFramesEventsDataset(base_folder='./data/test/', event_folder='events/data/', depth_folder = 'depth/data/', dataset_type='voxeltrain')
    # dataloader_valid = DataLoader(validation_dataset)


    train_obj = trainer.E2DEPTHTrainer(model, loss, loss_params, metrics, resume=False, config=config, data_loader=dataloader)
    train_obj.train()



if __name__ == '__main__':

    logger = logging.getLogger()

    parser = argparse.ArgumentParser(
    description='Learning DVS depth est.')
    parser.add_argument('-c', '--config', default=None, type=str, help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    parser.add_argument('-i', '--initial_checkpoint', default=None, type=str, help='path to the checkpoint with which to initialize the model weights (default: None)')

    args = parser.parse_args()


    config = None
    if args.resume is not None:
        if args.config is not None:
            logger.warning('Warning: --config overridden by --resume')
        if args.initial_checkpoint is not None:
            logger.warning('Warning: --initial_checkpoint overriden by --resume')
        config = torch.load(args.resume)['config']

    if args.config is not None:
        config = json.load(open(args.config))
        path = os.path.join(config['trainer']['save_dir'], config['name'])
        if args.resume is None:
            assert not os.path.exists(path), "Path {} already exists!".format(path)
 
    assert config is not None

    config['num_encoders'] = 3
    main(config)


