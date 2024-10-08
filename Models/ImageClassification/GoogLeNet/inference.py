from config.config import GoogLeNetConfig
from model import GoogLeNet

from global_utilis import save_and_load
from Models.ImageClassification.inference_template import *


def main():
    config_path = "config/config.yaml"
    config = GoogLeNetConfig(config_path)

    model = GoogLeNet(
        config.channel,
        config.num_classes,
        config.dropout,
        with_aux_logits=True,
    ).to(config.device)

    save_and_load.load_weight(config, model)

    inference(config, model)


if __name__ == '__main__':
    main()
