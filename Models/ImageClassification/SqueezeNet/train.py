from config.config import SqueezeNetConfig
from model import SqueezeNet

from Models.ImageClassification.train_template import *


def main():
    config_path = "config/config.yaml"
    config = SqueezeNetConfig(config_path)

    model = SqueezeNet(
        config.channel,
        config.network,
        config.dropout,
        config.num_classes,
    ).to(config.device)

    train(config, model)


if __name__ == '__main__':
    main()
