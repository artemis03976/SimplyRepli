from config.config import VGGConfig
from model import VGG

from global_utilis import save_and_load
from Models.ImageClassification.inference_template import *


def main():
    config_path = "config/config.yaml"
    config = VGGConfig(config_path)

    model = VGG(
        config.channel,
        config.network,
        config.num_classes,
        config.dropout,
    ).to(config.device)

    save_and_load.load_weight(config, model)

    inference(config, model)


if __name__ == '__main__':
    main()
