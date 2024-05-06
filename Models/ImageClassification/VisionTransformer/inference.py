from config.config import ViTConfig
from model import VisionTransformer

from global_utilis import save_and_load
from Models.ImageClassification.inference_template import *


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

    save_and_load.load_weight(config, model)

    inference(config, model)


if __name__ == '__main__':
    main()