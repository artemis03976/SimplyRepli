from yaml import safe_load


class InfoGANConfig:
    def __init__(self, config_path: str) -> None:
        self.config = safe_load(open(config_path, 'r', encoding='utf-8'))

        self.project_name = self.config['project_name']

        self.latent_noise_dim = self.config['model']['latent_noise_dim']
        self.latent_discrete_dim = self.config['model']['latent_discrete_dim']
        self.num_latent_discrete = self.config['model']['num_latent_discrete']
        self.latent_continuous_dim = self.config['model']['latent_continuous_dim']
        self.feature_size = self.config['model']['feature_size']
        self.base_channel = self.config['model']['base_channel']
        self.num_layers = self.config['model']['num_layers']
        self.lambda_discrete = self.config['model']['lambda_discrete']
        self.lambda_continuous = self.config['model']['lambda_continuous']

        self.channel = self.config['model']['channel']

        self.network = self.config['model']['network']

        self.batch_size = self.config['train']['batch_size']
        self.epochs = self.config['train']['epochs']
        self.generator_lr = self.config['train']['generator_lr']
        self.discriminator_lr = self.config['train']['discriminator_lr']
        self.d_step = self.config['train']['d_step']
        self.g_step = self.config['train']['g_step']

        self.num_samples = self.config['inference']['num_samples']

        self.device = self.config['device']