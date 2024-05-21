import os
import torch

save_dir = "checkpoints/"  # default save directory

# TODO:breakpoint saving and loading


def save_model(config, model, network=None):
    # create save directory if not exist
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # get name for saving
    if network is None:
        network = config.network

    # special setting for GANs
    if isinstance(model, tuple):
        if config.network == 'cyclegan':
            state_dict = {
                'G_A2B_state_dict': model[0].state_dict(),
                'G_B2A_state_dict': model[1].state_dict(),
                'D_A_state_dict': model[2].state_dict(),
                'D_B_state_dict': model[3].state_dict()
            }
        else:
            state_dict = {
                'generator_state_dict': model[0].state_dict(),
                'discriminator_state_dict': model[1].state_dict()
            }
    # default saving setting
    else:
        state_dict = {
            'model_state_dict': model.state_dict(),
        }

    torch.save(state_dict, save_dir + network + '.pth')

    print("Model Saved.")


def load_weight(config, model, network=None, **kwargs):
    # get name for loading
    if network is None:
        network = config.network
    model_path = save_dir + network + ".pth"

    # special setting for GANs
    if 'gan' in network or network == 'pix2pix':
        # param deciding whether to load discriminator
        load_discriminator = kwargs.get('load_discriminator', False)

        if config.network == 'cyclegan':
            model[0].load_state_dict(torch.load(model_path)['G_A2B_state_dict'])
            model[1].load_state_dict(torch.load(model_path)['G_B2A_state_dict'])
        else:
            if isinstance(model, tuple):
                model[0].load_state_dict(torch.load(model_path)['generator_state_dict'])
            else:
                model.load_state_dict(torch.load(model_path)['generator_state_dict'])

        if load_discriminator:
            if config.network == 'cyclegan':
                assert len(model) == 4, "model should contain 4 modules for cyclegan"
                model[2].load_state_dict(torch.load(model_path)['D_A_state_dict'])
                model[3].load_state_dict(torch.load(model_path)['D_B_state_dict'])
            else:
                model[1].load_state_dict(torch.load(model_path)['discriminator_state_dict'])
    # default loading setting
    else:
        model.load_state_dict(torch.load(model_path)['model_state_dict'])

    return model
