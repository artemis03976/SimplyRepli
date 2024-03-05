from config.config import AlexNetConfig
from model import AlexNet

from Models.TraditionalCNN.train_template import *


def main():
    config_path = "config/config.yaml"
    config = AlexNetConfig(config_path)

    model = AlexNet(
        config.num_classes,
        config.dropout,
    ).to(config.device)

    train(config, model)


if __name__ == '__main__':
    main()
