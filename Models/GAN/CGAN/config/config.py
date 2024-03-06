from yaml import safe_load


class CGANConfig:
    def __init__(self, config_path: str) -> None:
        self.config = safe_load(open(config_path, 'r', encoding='utf-8'))

        self.project_name = self.config['project_name']

        self.input_dim = self.config['model']['linear']['input_dim']
        self.output_dim = self.config['model']['linear']['output_dim']
        self.latent_dim_linear = self.config['model']['linear']['latent_dim_linear']
        self.G_hidden_dims = self.config['model']['linear']['G_hidden_dims']
        self.D_hidden_dims = self.config['model']['linear']['D_hidden_dims']

        self.in_channel = self.config['model']['conv']['in_channel']
        self.out_channel = self.config['model']['conv']['out_channel']
        self.G_mid_channels = self.config['model']['conv']['G_mid_channels']
        self.D_mid_channels = self.config['model']['conv']['D_mid_channels']
        self.latent_dim_conv = self.config['model']['conv']['latent_dim_conv']

        self.num_classes = self.config['model']['num_classes']
        self.proj_dim = self.config['model']['proj_dim']

        self.network = self.config['model']['network']

        self.batch_size = self.config['train']['batch_size']
        self.epochs = self.config['train']['epochs']
        self.generator_lr = self.config['train']['generator_lr']
        self.discriminator_lr = self.config['train']['discriminator_lr']
        self.d_step = self.config['train']['d_step']
        self.g_step = self.config['train']['g_step']

        self.num_samples = self.config['inference']['num_samples']

        self.device = self.config['device']
