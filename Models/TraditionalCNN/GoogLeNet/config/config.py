from yaml import safe_load


class GoogLeNetConfig:
    def __init__(self, config_path: str) -> None:
        self.config = safe_load(open(config_path, 'r', encoding='utf-8'))

        self.project_name = self.config['project_name']

        self.num_classes = self.config['model']['num_classes']
        self.dropout = self.config['model']['dropout']
        self.network = self.config['model']['network']
        self.img_size = self.config['model']['img_size']

        self.batch_size = self.config['train']['batch_size']
        self.epochs = self.config['train']['epochs']
        self.learning_rate = self.config['train']['learning_rate']

        self.num_samples = self.config['inference']['num_samples']

        self.device = self.config['device']
