import json, os ,argparse, logging
import torch
import torch.nn as nn
from data_fetchers.EventDataUtilities.dataset import SequenceSynchronizedFramesEventsDataset
from .model.model import E2VIDRecurrent
from .trainer import trainer

from torch.utils.data import DataLoader


def main(config):

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
    # validation_dataset = SequenceSynchronizedFramesEventsDataset(base_folder='./data/test/', event_folder='events/data/', depth_folder = 'depth/data/', dataset_type='voxeltrain')
    # dataloader_valid = DataLoader(validation_dataset)
    try:
        ret=config["trainer"] ["resume"]
    except:
        ret=False

    train_obj = trainer.E2DEPTHTrainer(model, loss, loss_params, metrics, ret, config=config, data_loader=dataloader)
    train_obj.train()




