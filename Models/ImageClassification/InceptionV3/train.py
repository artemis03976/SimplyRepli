from config.config import InceptionConfig
from model import InceptionV3

from Models.ImageClassification.train_template import *


def main():
    config_path = "config/config.yaml"
    config = InceptionConfig(config_path)

    model = InceptionV3(
        config.channel,
        config.num_classes,
        config.dropout,
        with_aux_logits=True,
    ).to(config.device)

    train(config, model)


if __name__ == '__main__':
    main()
