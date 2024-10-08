from yaml import safe_load


class RNNConfig:
    def __init__(self, config_path: str) -> None:
        self.config = safe_load(open(config_path, 'r', encoding='utf-8'))

        self.project_name = self.config['project_name']

        self.embed_dim = self.config['model']['embed_dim']
        self.hidden_dim = self.config['model']['hidden_dim']
        self.num_layers = self.config['model']['num_layers']
        self.dropout = self.config['model']['dropout']
        self.bidirectional = self.config['model']['bidirectional']

        self.dataset = self.config['dataset']['name']
        self.num_classes = self.config['dataset']['num_classes']

        self.network = self.config['model']['network']

        self.batch_size = self.config['train']['batch_size']
        self.epochs = self.config['train']['epochs']
        self.learning_rate = self.config['train']['learning_rate']

        self.num_samples = self.config['inference']['num_samples']

        self.device = self.config['device']
