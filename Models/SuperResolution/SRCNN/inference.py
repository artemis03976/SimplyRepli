from global_utilis import plot, save_and_load
from Models.SuperResolution.utilis import load_data
from config.config import SRCNNConfig
from model import SRCNN


def inference(config, model):
    print("Start reconstruction...")

    test_loader = load_data.get_test_loader(config)
    lr_img, sr_img = next(iter(test_loader))
    plot.show_img(lr_img, cols=4)
    plot.show_img(sr_img, cols=4)

    lr_img = lr_img.to(config.device)

    recon_sr_img = model(lr_img)

    print("End reconstruction...")

    plot.show_img(recon_sr_img, cols=4)


def main():
    config_path = "config/config.yaml"
    config = SRCNNConfig(config_path)

    model = SRCNN(
        config.channel,
    ).to(config.device)

    save_and_load.load_weight(config, model)

    inference(config, model)


if __name__ == "__main__":
    main()
