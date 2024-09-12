from yaml import safe_load


class StyleGANConfig:
    def __init__(self, config_path: str) -> None:
        self.config = safe_load(open(config_path, 'r', encoding='utf-8'))

        self.project_name = self.config['project_name']

        self.generator_in_channel = self.config['model']['generator_in_channel']
        self.latent_dim = self.config['model']['latent_dim']
        self.style_dim = self.config['model']['style_dim']

        self.dataset = self.config['dataset']['name']
        self.channel = self.config['dataset']['channel']
        self.img_size = self.config['dataset']['img_size']

        self.loss = self.config['model']['loss']
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
