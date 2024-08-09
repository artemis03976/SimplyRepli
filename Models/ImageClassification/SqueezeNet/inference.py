from config.config import SqueezeNetConfig
from model import SqueezeNet

from global_utilis import save_and_load
from Models.ImageClassification.inference_template import *


def main():
    config_path = "config/config.yaml"
    config = SqueezeNetConfig(config_path)

    model = SqueezeNet(
        config.channel,
        config.network,
        config.dropout,
        config.num_classes,
    ).to(config.device)

    save_and_load.load_weight(config, model)

    inference(config, model)


if __name__ == '__main__':
    main()
