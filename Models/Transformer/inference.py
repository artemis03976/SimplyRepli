from model import Transformer
from config.config import TransformerConfig
from global_utilis import save_and_load
from utilis import load_data


def inference(config, model, output_lang, val_loader):
    # switch mode
    model.eval()

    # get test data
    data = next(iter(val_loader))
    input_seq = data[0].to(config.device)
    ground_truth = data[1].to(config.device)

    for i in range(input_seq.shape[0]):
        encoder_input = input_seq[i].unsqueeze(0)
        decoder_output = model.inference(encoder_input, load_data.SOS_token)

        print(' '.join(translate(output_lang, ground_truth[i])))
        print(' '.join(translate(output_lang, decoder_output)))
        print('------------------------------------------------------')


def translate(output_lang, idx_seq):
    decoded_words = []
    # translate from ids to words by dict
    for idx in idx_seq:
        # add EOS token
        if idx.item() == load_data.EOS_token:
            decoded_words.append('<EOS>')
            break
        decoded_words.append(output_lang.index2word[idx.item()])

    return decoded_words


def main():
    config_path = "config/config.yaml"
    config = TransformerConfig(config_path)

    # get test data and vocab
    input_lang, output_lang, train_loader, val_loader = load_data.get_dataloader(config)
    src_vocab_size = input_lang.n_words
    tgt_vocab_size = output_lang.n_words

    model = Transformer(
        src_vocab_size,
        tgt_vocab_size,
        config.embed_dim,
        config.ffn_dim,
        config.num_heads,
        config.num_layers,
        config.dropout,
    ).to(config.device)

    save_and_load.load_weight(config, model)

    inference(config, model, output_lang, val_loader)


if __name__ == "__main__":
    main()
