from yaml import safe_load


class WGANGPConfig:
    def __init__(self, config_path: str) -> None:
        self.config = safe_load(open(config_path, 'r', encoding='utf-8'))

        self.project_name = self.config['project_name']

        self.in_channel = self.config['model']['in_channel']
        self.out_channel = self.config['model']['out_channel']
        self.G_mid_channels = self.config['model']['G_mid_channels']
        self.D_mid_channels = self.config['model']['D_mid_channels']
        self.latent_dim = self.config['model']['latent_dim']

        self.network = self.config['model']['network']

        self.batch_size = self.config['train']['batch_size']
        self.epochs = self.config['train']['epochs']
        self.generator_lr = self.config['train']['generator_lr']
        self.discriminator_lr = self.config['train']['discriminator_lr']
        self.grad_penalty_weight = self.config['train']['grad_penalty_weight']
        self.d_step = self.config['train']['d_step']
        self.g_step = self.config['train']['g_step']

        self.num_samples = self.config['inference']['num_samples']

        self.device = self.config['device']
