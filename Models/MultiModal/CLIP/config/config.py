from yaml import safe_load


class CLIPConfig:
    def __init__(self, config_path: str) -> None:
        self.config = safe_load(open(config_path, 'r', encoding='utf-8'))

        self.project_name = self.config['project_name']

        self.patch_size = self.config['model']['vision']['patch_size']
        self.vision_num_layers = self.config['model']['vision']['num_layers']
        self.vision_num_heads = self.config['model']['vision']['num_heads']
        self.vision_mlp_dim = self.config['model']['vision']['mlp_dim']

        self.text_embed_dim = self.config['model']['text']['embed_dim']
        self.text_num_layers = self.config['model']['text']['num_layers']
        self.text_num_heads = self.config['model']['text']['num_heads']
        self.text_mlp_dim = self.config['model']['text']['mlp_dim']

        self.align_dim = self.config['model']['align_dim']

        self.dataset = self.config['dataset']['name']
        self.img_size = self.config['dataset']['img_size']
        self.channel = self.config['dataset']['channel']
        self.text_max_len = self.config['dataset']['text_max_len']

        self.network = self.config['model']['network']

        self.batch_size = self.config['train']['batch_size']
        self.epochs = self.config['train']['epochs']
        self.learning_rate = self.config['train']['learning_rate']

        self.num_samples = self.config['inference']['num_samples']

        self.device = self.config['device']
