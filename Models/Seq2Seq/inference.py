import random
from nltk.translate.bleu_score import sentence_bleu
from global_utilis import save_and_load
from models.seq2seq import Seq2SeqRNN, Seq2SeqGRU, Seq2SeqLSTM
from models.seq2seq_attn import Seq2SeqAttnRNN, Seq2SeqAttnGRU, Seq2SeqAttnLSTM
from config.config import Seq2SeqConfig
from Models.Seq2Seq.utilis.load_data import *


network_mapping = {
    'rnn': Seq2SeqRNN,
    'gru': Seq2SeqGRU,
    'lstm': Seq2SeqLSTM,
    'rnn_attn': Seq2SeqAttnRNN,
    'gru_attn': Seq2SeqAttnGRU,
    'lstm_attn': Seq2SeqAttnLSTM,
}


def translate(config, model, input_lang, output_lang, test_pair):
    # switch mode
    model.eval()
    with torch.no_grad():
        # convert test sentence to tensor
        input_tensor, output_tensor = pair_to_tensor(input_lang, output_lang, test_pair, config.device)
        # disable teacher forcing
        decoder_outputs, attentions = model(input_tensor, output_tensor, teacher_forcing_ratio=0)
        # filter to obtain the most likely token
        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        # translate from ids to words by dict
        for idx in decoded_ids:
            # add EOS token
            if idx.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            decoded_words.append(output_lang.index2word[idx.item()])

    # generate output sentence
    output_sentence = ' '.join(decoded_words)

    print('>', test_pair[0])
    print('=', test_pair[1])
    print('<', output_sentence)

    return test_pair[1], output_sentence


def inference(config, model, input_lang, output_lang):
    bleu = np.zeros(4)
    for _ in range(config.num_samples):
        reference, candidate = translate(config, model, input_lang, output_lang, random.choice(pairs))
        # calculate bleu score
        bleu += evaluate(reference, candidate)

    bleu /= config.num_samples
    print('Cumulative 1-gram:', bleu[0])
    print('Cumulative 2-gram:', bleu[1])
    print('Cumulative 3-gram:', bleu[2])
    print('Cumulative 4-gram:', bleu[3])


def evaluate(reference, candidate):
    reference = [reference.split(' ')]
    candidate = candidate.split(' ')[:-1]
    # cumulative bleu
    bleu1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
    bleu2 = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0))
    bleu3 = sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0))
    bleu4 = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))

    return np.array([bleu1, bleu2, bleu3, bleu4])


def main():
    config_path = "config/config.yaml"
    config = Seq2SeqConfig(config_path)

    # get test data and vocab
    input_lang, output_lang, pairs = prepare_data(config.input_lang, config.output_lang, config.reverse)
    src_vocab_size = input_lang.n_words
    tgt_vocab_size = output_lang.n_words

    if config.network not in network_mapping:
        raise NotImplementedError(f"Unsupported network: {config.network}")

    model = network_mapping[config.network](
        src_vocab_size,
        tgt_vocab_size,
        config.embed_dim,
        config.hidden_dim,
        config.num_layers,
        config.encode_dropout,
        config.decode_dropout,
        config.bidirectional,
        config.device,
    ).to(config.device)

    save_and_load.load_weight(config, model)

    inference(config, model, input_lang, output_lang)


if __name__ == '__main__':
    main()
