import logging
import os
import sys
import torch

sys.path.append('/home/parinayok/food.com_net')

import init_config
from trainer import get_trainer

logger = logging.getLogger()

def main():
    # init configurations from config file
    _, config = init_config.get_arguments()
    os.makedirs(config.SAVE_PATH, exist_ok=True)
    os.makedirs(os.path.join(config.SAVE_PATH,'checkpoints'), exist_ok=True)

    #init logger and dump config
    log_path = os.path.join("log", config.SAVE_PATH + ".txt")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = init_config.init_logger(
        os.path.dirname(log_path), os.path.basename(log_path)
    )
    logger.info(config.dump())

    # dump current config
    dump_path = os.path.join("config", config.SAVE_PATH + ".yaml")
    os.makedirs(os.path.dirname(dump_path), exist_ok=True)
    with open(dump_path, "w") as f:
        f.write(config.dump())  # type: ignore

    # set random seed
    init_config.set_random_seed(config.TRAIN.SEED)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(device)
    trainer = get_trainer(config, device)
    trainer.train()

if __name__ == '__main__':
    main()
