import torch
import math

from config.config import UNetConfig
from model import UNet
from Models.UNet.utilis import crop, load_data
from global_utilis import save_and_load, plot


def inference(config, model):
    # switch mode
    model.eval()
    # get test data loader
    _, test_loader = load_data.get_train_val_loader(config)

    with torch.no_grad():
        image, mask = next(iter(test_loader))
        image = image.to(config.device)

        prediction = model(image)
        mask = crop.center_crop(prediction, mask)
        # generate binary mask from prediction
        prediction = (prediction > config.mask_threshold).float()

        plot.show_img(image, cols=int(math.sqrt(config.num_samples)))
        # plot every class
        for i in range(mask.shape[1]):
            plot.show_img(mask[:, i].unsqueeze(1), cols=int(math.sqrt(config.num_samples)))
            plot.show_img(prediction[:, i].unsqueeze(1), cols=int(math.sqrt(config.num_samples)))


def main():
    config_path = "config/config.yaml"
    config = UNetConfig(config_path)

    model = UNet(
        config.channel,
        config.num_classes,
        config.ch_multi,
    ).to(config.device)

    save_and_load.load_weight(config, model)

    inference(config, model)


if __name__ == '__main__':
    main()
