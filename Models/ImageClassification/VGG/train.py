from config.config import VGGConfig
from model import VGG

from Models.ImageClassification.train_template import *


def main():
    config_path = "config/config.yaml"
    config = VGGConfig(config_path)

    model = VGG(
        config.channel,
        config.network,
        config.num_classes,
        config.dropout,
    ).to(config.device)

    train(config, model)


if __name__ == '__main__':
    main()
