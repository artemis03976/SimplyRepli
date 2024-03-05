from yaml import safe_load


class Seq2SeqConfig:
    def __init__(self, config_path: str) -> None:
        self.config = safe_load(open(config_path, 'r', encoding='utf-8'))

        self.project_name = self.config['project_name']

        self.embed_dim = self.config['model']['embed_dim']
        self.hidden_dim = self.config['model']['hidden_dim']
        self.num_layers = self.config['model']['num_layers']
        self.encode_dropout = self.config['model']['encode_dropout']
        self.decode_dropout = self.config['model']['decode_dropout']
        self.bidirectional = self.config['model']['bidirectional']
        self.network = self.config['model']['network']

        self.batch_size = self.config['train']['batch_size']
        self.epochs = self.config['train']['epochs']
        self.learning_rate = self.config['train']['learning_rate']

        self.num_samples = self.config['inference']['num_samples']

        self.device = self.config['device']
