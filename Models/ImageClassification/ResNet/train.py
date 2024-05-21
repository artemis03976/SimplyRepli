from config.config import ResNetConfig
from model import ResNet

from Models.ImageClassification.train_template import *


def main():
    config_path = "config/config.yaml"
    config = ResNetConfig(config_path)

    model = ResNet(
        config.channel,
        config.network,
        config.num_classes,
    ).to(config.device)

    train(config, model)


if __name__ == '__main__':
    main()
