import json, os ,argparse, logging
import torch




def main():
    


if __name__ == '__main__':

    logger = logging.getLogger()

    parser = argparse.ArgumentParser(
    description='Learning DVS Image Reconstruction')
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


    main(config)


