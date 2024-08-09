from yaml import safe_load


class TransUNetConfig:
    def __init__(self, config_path: str) -> None:
        self.config = safe_load(open(config_path, 'r', encoding='utf-8'))

        self.project_name = self.config['project_name']

        self.out_channel = self.config['model']['out_channel']
        self.patch_size = self.config['model']['patch_size']
        self.num_layers = self.config['model']['num_layers']
        self.num_heads = self.config['model']['num_heads']
        self.mlp_dim = self.config['model']['mlp_dim']

        self.network = self.config['model']['network']

        self.dataset = self.config['dataset']['name']
        self.img_size = self.config['dataset']['img_size']
        self.channel = self.config['dataset']['channel']
        self.num_classes = self.config['dataset']['num_classes']

        self.batch_size = self.config['train']['batch_size']
        self.epochs = self.config['train']['epochs']
        self.learning_rate = self.config['train']['learning_rate']

        self.num_samples = self.config['inference']['num_samples']
        self.mask_threshold = self.config['inference']['mask_threshold']

        self.device = self.config['device']
