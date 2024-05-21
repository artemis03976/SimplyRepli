from config.config import DenseNetConfig
from model import DenseNet

from global_utilis import save_and_load
from Models.ImageClassification.inference_template import *


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

    save_and_load.load_weight(config, model)

    inference(config, model)


if __name__ == '__main__':
    main()
