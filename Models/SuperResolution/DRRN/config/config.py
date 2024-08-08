from yaml import safe_load


class DRRNConfig:
    def __init__(self, config_path: str) -> None:
        self.config = safe_load(open(config_path, 'r', encoding='utf-8'))

        self.project_name = self.config['project_name']

        self.mid_channel = self.config['model']['mid_channel']
        self.num_layers = self.config['model']['num_layers']
        self.recurse_depth = self.config['model']['recurse_depth']

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
