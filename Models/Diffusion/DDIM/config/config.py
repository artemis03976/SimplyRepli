from yaml import safe_load


class DDIMConfig:
    def __init__(self, config_path: str) -> None:
        self.config = safe_load(open(config_path, 'r', encoding='utf-8'))

        self.project_name = self.config['project_name']

        self.num_res_blocks = self.config['model']['num_res_blocks']
        self.base_channel = self.config['model']['base_channel']
        self.time_embed_channel = self.config['model']['time_embed_channel']
        self.ch_mult = self.config['model']['ch_mult']
        self.num_time_step = self.config['model']['num_time_step']
        self.num_sample_step = self.config['model']['num_sample_step']
        self.betas = self.config['model']['betas']
        self.eta = self.config['model']['eta']

        self.dataset = self.config['dataset']['name']
        self.img_size = self.config['dataset']['img_size']
        self.channel = self.config['dataset']['channel']

        self.network = self.config['model']['network']

        self.batch_size = self.config['train']['batch_size']
        self.epochs = self.config['train']['epochs']
        self.learning_rate = self.config['train']['learning_rate']

        self.num_samples = self.config['inference']['num_samples']

        self.device = self.config['device']
