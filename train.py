import torch.nn as nn
from data_fetchers.dataset_factory import SequenceDatasetFactory
# from data_fetchers.dataset_legacy import SequenceSynchronizedFramesEventsDataset
from .model.model import E2VIDRecurrent
from .trainer import trainer

from torch.utils.data import DataLoader


def train(config):

    # model
    model = E2VIDRecurrent(config["model"])

    # loss
    loss = nn.MSELoss()
    loss_params = None

    # metrics
    metrics = []

    #e2depth train dataloader
    dataset = SequenceSynchronizedFramesEventsDataset(base_folder=config["trainer"]["base_dir"], event_folder=config["trainer"]["event_dir"], depth_folder = config["trainer"]["depth_dir"], dataset_type='voxeltrain')
    dataloader = DataLoader(dataset)
    # validation_dataset = ContinuousEventsDataset(base_folder=config["validater"]["base_dir"], event_folder=config["validater"]["event_dir"],\
    #         width=346, height=260, window_size = config["validater"]['window_size'], time_shift = config["validater"]['time_shift'])
    # dataloader_valid = DataLoader(validation_dataset)
    try:
        ret=config["trainer"] ["resume"]
    except:
        ret=False

    train_obj = trainer.E2DEPTHTrainer(model, loss, loss_params, metrics, ret, config=config, data_loader=dataloader)
    train_obj.train()


def finetune(config):

    # model
    model = E2VIDRecurrent(config["model"])

    # loss
    loss = nn.MSELoss()
    loss_params = None

    # metrics
    metrics = []

    #e2depth train dataloader
#    dataset = SequenceSynchronizedFramesEventsDataset(base_folder=config["trainer"]["base_dir"], event_folder=config["trainer"]["event_dir"], depth_folder = config["trainer"]["depth_dir"], dataset_type='shifttrain')
    dataset = SequenceDatasetFactory.createSeqSyncFED(base_folder='/home/sankeerth/ev/depth_proj/data/test/', event_folder='events/data/', depth_folder = 'depth/data/', frame_folder='rgb/frames/')
    dataloader = DataLoader(dataset)
    # validation_dataset = ContinuousEventsDataset(base_folder=config["validater"]["base_dir"], event_folder=config["validater"]["event_dir"],\
    #         width=346, height=260, window_size = config["validater"]['window_size'], time_shift = config["validater"]['time_shift'])
    # dataloader_valid = DataLoader(validation_dataset)
    try:
        ret=config["trainer"]["resume"]
    except:
        ret=False

    train_obj = trainer.E2DepthPerturbationTuner(model, loss, loss_params, metrics, ret, config=config, data_loader=dataloader)
    train_obj.train()


