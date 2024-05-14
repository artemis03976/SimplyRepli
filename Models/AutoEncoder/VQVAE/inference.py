import torch
import torch.nn.functional as F

from global_utilis import plot, save_and_load
from Models.AutoEncoder.utilis import load_data
from config.config import VQVAEConfig
from model import VQVAE
from modules.pixelcnn import PixelCNN


def inference(config, model, prior, reconstruct=True, generate=True):
    model.eval()
    prior.eval()

    if reconstruct:
        reconstruction(config, model)
    if generate:
        generation(config, model, prior)


def generation(config, model, prior):
    print("Start generation...")
    shape = (config.feature_size, config.feature_size)

    sample = torch.zeros(config.num_samples, shape[0], shape[1], dtype=torch.int64, device=config.device)

    with torch.no_grad():
        for i in range(shape[0]):
            for j in range(shape[1]):
                logits = prior(sample)
                probs = F.softmax(logits[:, :, i, j], dim=-1).data

                pixel = torch.multinomial(probs, 1)
                pixel = pixel.squeeze(-1)
                sample[:, i, j] = pixel

        quantized = model.quantizer.embedding(sample)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        samples = model.decoder(quantized)
        samples = samples.view(config.num_samples, config.channel, config.img_size, config.img_size)

    print("End generation...")

    plot.show_img(samples, cols=8)


def reconstruction(config, model):
    print("Start reconstruction...")

    test_loader = load_data.get_test_loader(config)
    images, labels = next(iter(test_loader))
    plot.show_img(images, cols=8)

    with torch.no_grad():
        images = images.to(config.device)
        reconstruction_images, _, _ = model(images)

    print("End reconstruction...")

    plot.show_img(reconstruction_images, cols=8)


def main():
    config_path = "config/config.yaml"
    config = VQVAEConfig(config_path)

    in_channel = out_channel = config.channel

    model = VQVAE(
        in_channel,
        out_channel,
        config.num_embeddings,
        config.embed_dim,
        config.num_res_blocks,
    ).to(config.device)

    save_and_load.load_weight(config, model)

    prior = PixelCNN(
        config.num_embeddings,
        config.num_embeddings,
        config.mid_channel,
        config.num_res_blocks_prior,
    ).to(config.device)

    save_and_load.load_weight(config, prior, network='pixelcnn')

    inference(config, model, prior)


if __name__ == "__main__":
    main()
