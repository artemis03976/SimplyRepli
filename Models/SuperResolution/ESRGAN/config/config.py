from yaml import safe_load


class SRGANConfig:
    def __init__(self, config_path: str) -> None:
        self.config = safe_load(open(config_path, 'r', encoding='utf-8'))

        self.project_name = self.config['project_name']

        self.base_channel = self.config['model']['base_channel']
        self.growing_channel = self.config['model']['growing_channel']
        self.num_blocks_g = self.config['model']['num_blocks_g']
        self.num_blocks_d = self.config['model']['num_blocks_d']

        self.dataset = self.config['dataset']['name']
        self.img_size = self.config['dataset']['img_size']
        self.channel = self.config['dataset']['channel']
        self.scale_factor = self.config['dataset']['scale_factor']

        self.network = self.config['model']['network']

        self.batch_size = self.config['train']['batch_size']
        self.epochs = self.config['train']['epochs']
        self.generator_lr = self.config['train']['generator_lr']
        self.discriminator_lr = self.config['train']['discriminator_lr']
        self.d_step = self.config['train']['d_step']
        self.g_step = self.config['train']['g_step']

        self.num_samples = self.config['inference']['num_samples']

        self.device = self.config['device']
