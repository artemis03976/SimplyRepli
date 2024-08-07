from config.config import ConvNeXtConfig
from model import ConvNeXt

from Models.ImageClassification.train_template import *


def main():
    config_path = "config/config.yaml"
    config = ConvNeXtConfig(config_path)

    model = ConvNeXt(
        config.channel,
        config.network,
        config.num_classes,
    ).to(config.device)

    train(config, model)


if __name__ == '__main__':
    main()
