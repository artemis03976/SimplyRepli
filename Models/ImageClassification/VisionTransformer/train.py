from config.config import ViTConfig
from model import VisionTransformer

from Models.ImageClassification.train_template import *


def main():
    config_path = "config/config.yaml"
    config = ViTConfig(config_path)

    model = VisionTransformer(
        config.channel,
        config.patch_size,
        config.img_size,
        config.num_classes,
        config.num_encoders,
        config.num_heads,
        config.mlp_dim,
        config.dropout,
    ).to(config.device)

    train(config, model)


if __name__ == '__main__':
    main()
