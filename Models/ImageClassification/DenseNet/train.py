from config.config import DenseNetConfig
from model import DenseNet

from Models.ImageClassification.train_template import *


def main():
    config_path = "config/config.yaml"
    config = DenseNetConfig(config_path)

    model = DenseNet(
        config.channel,
        config.network,
        config.num_classes,
        config.growth_rate,
        config.dropout
    ).to(config.device)

    train(config, model)


if __name__ == '__main__':
    main()
