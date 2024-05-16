from yaml import safe_load


class VDSRConfig:
    def __init__(self, config_path: str) -> None:
        self.config = safe_load(open(config_path, 'r', encoding='utf-8'))

        self.project_name = self.config['project_name']

        self.num_layers = self.config['model']['num_layers']

        self.dataset = self.config['dataset']['name']
        self.img_size = self.config['dataset']['img_size']
        self.channel = self.config['dataset']['channel']
        self.scale_factor = self.config['dataset']['scale_factor']

        self.network = self.config['model']['network']

        self.batch_size = self.config['train']['batch_size']
        self.epochs = self.config['train']['epochs']
        self.learning_rate = self.config['train']['learning_rate']
        self.clip_grad = self.config['train']['clip_grad']

        self.num_samples = self.config['inference']['num_samples']

        self.device = self.config['device']
