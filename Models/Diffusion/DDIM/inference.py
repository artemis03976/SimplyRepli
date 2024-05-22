import torch
from global_utilis import save_and_load, plot
from config.config import DDIMConfig
from model import DDIM


def inference(config, model):
    # sample random noise for generation
    noise_sample = torch.randn((config.num_samples, config.channel, config.img_size, config.img_size), device=config.device)

    generation = model.inference(noise_sample)
    # rescale to [0, 1]
    out = 0.5 * (generation + 1)
    out = out.clamp(0, 1)
    out = out.view(config.num_samples, config.channel, config.img_size, config.img_size)
    # show generated images
    plot.show_img(out, cols=4)


def main():
    config_path = "config/config.yaml"
    config = DDIMConfig(config_path)

    in_channel = out_channel = config.channel

    model = DDIM(
        in_channel,
        out_channel,
        config.num_res_blocks,
        config.base_channel,
        config.time_embed_channel,
        config.ch_mult,
        config.num_time_step,
        config.num_sample_step,
        config.betas,
        config.eta,
    ).to(config.device)

    save_and_load.load_weight(config, model)

    inference(config, model)


if __name__ == "__main__":
    main()
