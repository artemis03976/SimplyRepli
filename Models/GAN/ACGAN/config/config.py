from yaml import safe_load


class ACGANConfig:
    def __init__(self, config_path: str) -> None:
        self.config = safe_load(open(config_path, 'r', encoding='utf-8'))

        self.project_name = self.config['project_name']

        self.latent_dim = self.config['model']['latent_dim']
        self.proj_dim = self.config['model']['proj_dim']
        self.base_channel = self.config['model']['base_channel']
        self.G_num_layers = self.config['model']['G_num_layers']
        self.D_num_layers = self.config['model']['D_num_layers']
        self.dropout = self.config['model']['dropout']

        self.dataset = self.config['dataset']['name']
        self.img_size = self.config['dataset']['img_size']
        self.channel = self.config['dataset']['channel']
        self.num_classes = self.config['dataset']['num_classes']

        self.network = self.config['model']['network']

        self.batch_size = self.config['train']['batch_size']
        self.epochs = self.config['train']['epochs']
        self.generator_lr = self.config['train']['generator_lr']
        self.discriminator_lr = self.config['train']['discriminator_lr']
        self.d_step = self.config['train']['d_step']
        self.g_step = self.config['train']['g_step']

        self.num_samples = self.config['inference']['num_samples']

        self.device = self.config['device']
