from yaml import safe_load


class DRCNConfig:
    def __init__(self, config_path: str) -> None:
        self.config = safe_load(open(config_path, 'r', encoding='utf-8'))

        self.project_name = self.config['project_name']

        self.inference_depth = self.config['model']['inference_depth']
        self.alpha = self.config['model']['alpha']
        self.alpha_decay_epoch = self.config['model']['alpha_decay_epoch']
        self.beta = self.config['model']['beta']

        self.dataset = self.config['dataset']['name']
        self.img_size = self.config['dataset']['img_size']
        self.channel = self.config['dataset']['channel']
        self.scale_factor = self.config['dataset']['scale_factor']

        self.network = self.config['model']['network']

        self.batch_size = self.config['train']['batch_size']
        self.epochs = self.config['train']['epochs']
        self.learning_rate = self.config['train']['learning_rate']

        self.num_samples = self.config['inference']['num_samples']

        self.device = self.config['device']
