from yaml import safe_load


class AEConfig:
    def __init__(self, config_path: str) -> None:
        self.config = safe_load(open(config_path, 'r', encoding='utf-8'))

        self.project_name = self.config['project_name']

        self.latent_dim_linear = self.config['model']['linear']['latent_dim_linear']
        self.hidden_dims = self.config['model']['linear']['hidden_dims']

        self.mid_channels = self.config['model']['conv']['mid_channels']
        self.latent_dim_conv = self.config['model']['conv']['latent_dim_conv']
        self.kernel_size = self.config['model']['conv']['kernel_size']

        self.img_size = self.config['model']['img_size']
        self.channel = self.config['model']['channel']

        self.network = self.config['model']['network']

        self.batch_size = self.config['train']['batch_size']
        self.epochs = self.config['train']['epochs']
        self.learning_rate = self.config['train']['learning_rate']

        self.num_samples = self.config['inference']['num_samples']

        self.device = self.config['device']
