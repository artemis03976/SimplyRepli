from config.config import ConvNeXtConfig
from model import ConvNeXt

from global_utilis import save_and_load
from Models.ImageClassification.inference_template import *


def main():
    config_path = "config/config.yaml"
    config = ConvNeXtConfig(config_path)

    model = ConvNeXt(
        config.channel,
        config.network,
        config.num_classes,
    ).to(config.device)

    save_and_load.load_weight(config, model)

    inference(config, model)


if __name__ == '__main__':
    main()
