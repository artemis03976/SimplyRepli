from yaml import safe_load


class VQVAEConfig:
    def __init__(self, config_path: str) -> None:
        self.config = safe_load(open(config_path, 'r', encoding='utf-8'))

        self.project_name = self.config['project_name']

        self.num_embeddings = self.config['model']['num_embeddings']
        self.embed_dim = self.config['model']['embed_dim']
        self.num_res_blocks = self.config['model']['num_res_blocks']
        self.beta = self.config['model']['beta']

        self.img_size = self.config['model']['img_size']
        self.feature_size = self.config['model']['feature_size']
        self.channel = self.config['model']['channel']

        self.network = self.config['model']['network']
        self.prior_network = self.config['prior']['network']

        self.mid_channel = self.config['prior']['mid_channel']
        self.num_res_blocks_prior = self.config['prior']['num_res_blocks_prior']

        self.model_batch_size = self.config['train']['model']['batch_size']
        self.model_epochs = self.config['train']['model']['epochs']
        self.model_learning_rate = self.config['train']['model']['learning_rate']
        self.prior_batch_size = self.config['train']['prior']['batch_size']
        self.prior_epochs = self.config['train']['prior']['epochs']
        self.prior_learning_rate = self.config['train']['prior']['learning_rate']

        self.num_samples = self.config['inference']['num_samples']

        self.device = self.config['device']
