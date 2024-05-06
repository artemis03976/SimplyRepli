from config.config import ResNetConfig
from model import ResNet

from global_utilis import save_and_load
from Models.TraditionalCNN.inference_template import *


def main():
    config_path = "config/config.yaml"
    config = ResNetConfig(config_path)

    model = ResNet(
        config.network,
        config.num_classes,
    ).to(config.device)

    save_and_load.load_weight(config, model)

    inference(config, model)


if __name__ == '__main__':
    main()
