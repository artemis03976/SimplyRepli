import os
import torch

save_dir = "checkpoints/"


def save_model(config, model, network=None):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if network is None:
        network = config.network

    if isinstance(model, tuple):
        state_dict = {
            'generator_state_dict': model[0].state_dict(),
            'discriminator_state_dict': model[1].state_dict()
        }
    else:
        state_dict = {
            'model_state_dict': model.state_dict(),
        }

    torch.save(state_dict, save_dir + network + '.pth')

    print("Model Saved.")


def load_weight(config, model, network=None):
    if network is None:
        network = config.network

    model_path = save_dir + network + ".pth"

    if isinstance(model, tuple):
        model[0].load_state_dict(torch.load(model_path)['generator_state_dict'])
        model[1].load_state_dict(torch.load(model_path)['discriminator_state_dict'])

    else:
        model.load_state_dict(torch.load(model_path)['model_state_dict'])

    return model
