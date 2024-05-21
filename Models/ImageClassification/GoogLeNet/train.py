from config.config import GoogLeNetConfig
from model import GoogLeNet

from Models.ImageClassification.train_template import *


def main():
    config_path = "config/config.yaml"
    config = GoogLeNetConfig(config_path)

    model = GoogLeNet(
        config.channel,
        config.num_classes,
        config.dropout,
        with_aux_logits=True,
    ).to(config.device)

    train(config, model)


if __name__ == '__main__':
    main()
