import torch
from global_utilis import save_and_load, plot
from config.config import DDPMConfig
from model import DDPM


def inference(config, model):
    noise_sample = torch.randn((config.num_samples, config.channel, config.img_size, config.img_size), device=config.device)

    generation = model.inference(noise_sample)
    out = 0.5 * (generation + 1)
    out = out.clamp(0, 1)
    out = out.view(config.num_samples, config.channel, config.img_size, config.img_size)

    plot.show_img(out, cols=4)


def main():
    config_path = "config/config.yaml"
    config = DDPMConfig(config_path)

    in_channel = out_channel = config.channel

    model = DDPM(
        in_channel,
        out_channel,
        config.num_res_blocks,
        config.base_channel,
        config.time_embed_channel,
        config.ch_mult,
        config.num_time_step,
        config.betas
    ).to(config.device)

    save_and_load.load_weight(config, model)

    inference(config, model)


if __name__ == "__main__":
    main()
