from lexicon import create_dictionary_from_lexicon, create_dictionary_from_punctuation_marks, create_phone_dictionary_from_lexicon, create_lexicon

def get_config():
    config = {}
    config['vocabulary'] = "/disk/data2/s1569734/acoustic_punctuation/mgb.150k.wlist"
    config['lexicon'] = create_lexicon("/disk/data2/s1569734/bbc_original/data/local/dict/lexicon.txt")
    config['phones'] = create_phone_dictionary_from_lexicon("/disk/data2/s1569734/acoustic_punctuation/nonsilence_phones.txt", "/disk/data2/s1569734/acoustic_punctuation/silence_phones.txt")
    config['phones_vocab_size'] = len(config['phones'])
    config['punctuation_marks'] = ["<FULL_STOP>", "<COMMA>", "<QUESTION_MARK>", "<EXCLAMATION_MARK>", "<DOTS>"]

    config["src_vocab"] = create_dictionary_from_lexicon(config["vocabulary"], config["punctuation_marks"])
    config["src_vocab_size"] = len(config["src_vocab"].values())
    config["trg_vocab"] = create_dictionary_from_punctuation_marks(config["punctuation_marks"])
    config["trg_vocab_size"] = len(config["trg_vocab"].values())
    config["src_eos_idx"] = config["src_vocab"]["</s>"]
    config["trg_eos_idx"] = config["trg_vocab"]["</s>"]
    config['bos_token'] = '<s>'
    config['eos_token'] = '</s>'
    config['unk_token'] = '<unk>'


    config['train_data_dir'] = "/disk/data2/s1569734/bbc_original/data/train_mer10/"
    config['train_alignment_dir'] = "/disk/data2/s1569734/bbc_without_punctuation/exp/alignment/train_mer10_without_punct/"

    config['dev_data_dir'] = "/disk/data2/s1569734/bbc_original/data/dev_for_punctuation_addition/"
    config['dev_alignment_dir'] = "/disk/data2/s1569734/bbc_without_punctuation/exp/alignment/dev_for_punctuation_addition/"

    config['data_dir'] = "/disk/data2/s1569734/acoustic_punctuation/"

    # Model related -----------------------------------------------------------

    config['input'] = 'phones'
    config['audio_feat_size'] = 43
    config['take_every_nth'] = 3


    # Sequences longer than this will be discarded
    config['seq_len'] = 1000

    # Number of hidden units in encoder/decoder GRU
    config['enc_nhids'] = 256
    config['dec_nhids'] = 256

    # Dimension of the word embedding matrix in encoder/decoder
    config['enc_embed'] = 256
    config['dec_embed'] = 256

    # Where to save model, this corresponds to 'prefix' in groundhog
    config['saveto'] = '/disk/data2/s1569734/acoustic_punctuation/nmt_punctuation_on_%s/' % config['input']

    # Optimization related ----------------------------------------------------

    # Batch size
    config['batch_size'] = 50

    # This many batches will be read ahead and sorted
    config['sort_k_batches'] = 50

    # Optimization step rule
    config['step_rule'] = 'AdaDelta'

    # Gradient clipping threshold
    config['step_clipping'] = 1.

    # Std of weight initialization
    config['weight_scale'] = 0.01

    # Regularization related --------------------------------------------------

    # Dropout ratio, applied only after readout maxout
    config['dropout'] = 0.5

    # Vocabulary/dataset related ----------------------------------------------

    # Root directory for dataset
    datadir = '/disk/data2/s1569734/nmt_data/'

    # Module name of the stream that will be used
    config['stream'] = 'stream'

    # Early stopping based on f1 related ------------------------------------

    # Normalize cost according to sequence length after beam-search
    config['normalized_f1'] = True

    # Use F1 validation
    config['f1_validation'] = True

    # Print validation output to file
    config['output_val_set'] = True

    # Validation output file
    config['val_set_out'] = config['saveto'] + '/validation_out.txt'

    # Beam-size
    config['beam_size'] = 6

    # Timing/monitoring related -----------------------------------------------

    # Maximum number of updates
    config['finish_after'] = 100000

    # Reload model from files if exist
    config['reload'] = True

    # Save model after this many updates
    config['save_freq'] = 500

    # Show samples from model after this many updates
    config['sampling_freq'] = 1000

    # Show this many samples at each sampling
    config['hook_samples'] = 10

    # Validate f1 after this many updates
    config['f1_val_freq'] = 5000

    # Start f1 validation after this many updates
    config['val_burn_in'] = 5000

    return config
