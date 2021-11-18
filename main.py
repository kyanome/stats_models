import argparse
from utils.utils import read_config, setting_optimizer

def main(args) -> None:
    config = read_config(args.filename)
    optimizer = setting_optimizer(config)
    optimizer.fit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',  '-c',
                        dest="filename",
                        metavar='FILE',
                        help =  'path to the config file',
                        default='configs/vae.yaml')
    args = parser.parse_args()
    main(args)