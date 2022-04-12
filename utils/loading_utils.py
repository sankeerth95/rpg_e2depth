import torch
from ..model.model import *
from incr_modules.fenced_module_sparse import E2VIDRecurrentIncr

def load_model(path_to_model, config_dict = {}):
    print('Loading model {}...'.format(path_to_model))
    raw_model = torch.load(path_to_model)
    arch = raw_model['arch']

    try:
        model_type = raw_model['model']
    except KeyError:
        model_type = raw_model['config']['model']

    print(arch)
    model_type.update(config_dict)

    # instantiate model
    model = eval(arch)(model_type)

    # load model weights
    model.load_state_dict(raw_model['state_dict'])

    return model


def load_model_incr(path_to_model, config_dict = {}):
    print('Loading model {}...'.format(path_to_model))
    raw_model = torch.load(path_to_model)
    arch = raw_model['arch'] + 'Incr'

    try:
        model_type = raw_model['model']
    except KeyError:
        model_type = raw_model['config']['model']


    model_type.update(config_dict)

    print ("Model Type", model_type)
    # instantiate model
    model = eval(arch)(model_type)

    # load model weights
    model.load_state_dict(raw_model['state_dict'], strict=False)

    return model


def get_device(use_gpu):
    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print('Device:', device)

    return device
