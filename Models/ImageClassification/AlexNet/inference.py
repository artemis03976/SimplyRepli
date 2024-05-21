from config.config import AlexNetConfig
from model import AlexNet

from global_utilis import save_and_load
from Models.ImageClassification.inference_template import *


def main():
    config_path = "config/config.yaml"
    config = AlexNetConfig(config_path)

    model = AlexNet(
        config.channel,
        config.num_classes,
        config.dropout,
    ).to(config.device)

    save_and_load.load_weight(config, model)

    inference(config, model)


if __name__ == '__main__':
    main()
