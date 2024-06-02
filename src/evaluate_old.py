import torch
import sys

sys.path.append('/home/parinayok/food.com_net')

import init_config
from evaluator import get_evaluator

def main():
    # init configurations from config file
    _, config = init_config.get_arguments()
    logger = init_config.init_logger('.','log.txt')
    # set random seed
    init_config.set_random_seed(config.TRAIN.SEED)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(device)
    evaluator = get_evaluator(config,device)
    evaluator.evaluate()

if __name__ == '__main__':
    main()
