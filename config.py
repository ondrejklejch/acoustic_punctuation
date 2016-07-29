def get_config():
    config = {}
    config['lexicon'] = "/disk/data2/s1569734/bbc_original/data/lang/words.txt"

    config['train_data_dir'] = "/disk/data2/s1569734/bbc_original/data/train_mer10/"
    config['train_alignment_dir'] = "/disk/data2/s1569734/bbc_without_punctuation/exp/alignment/train_mer10_without_punct/"

    config['dev_data_dir'] = "/disk/data2/s1569734/bbc_original/data/dev_for_punctuation_addition/"
    config['dev_alignment_dir'] = "/disk/data2/s1569734/bbc_without_punctuation/exp/alignment/dev_for_punctuation_addition/"

    config['data_dir'] = "/disk/data2/s1569734/acoustic_punctuation/"
    config['punctuation_marks'] = ["<FULL_STOP>", "<COMMA>", "<QUESTION_MARK>", "<EXCLAMATION_MARK>", "<DOTS>"]

    # Model related -----------------------------------------------------------

    # Sequences longer than this will be discarded
    config['seq_len'] = 50

    # Number of hidden units in encoder/decoder GRU
    config['enc_nhids'] = 512
    config['dec_nhids'] = 512

    # Dimension of the word embedding matrix in encoder/decoder
    config['enc_embed'] = 512
    config['dec_embed'] = 512

    # Where to save model, this corresponds to 'prefix' in groundhog
    config['saveto'] = '/disk/data2/s1569734/acoustic_punctuation/nmt_punctuation_on_words_on_lm_data'

    # Optimization related ----------------------------------------------------

    # Batch size
    config['batch_size'] = 80

    # This many batches will be read ahead and sorted
    config['sort_k_batches'] = 12

    # Optimization step rule
    config['step_rule'] = 'AdaDelta'

    # Gradient clipping threshold
    config['step_clipping'] = 1.

    # Std of weight initialization
    config['weight_scale'] = 0.01

    # Regularization related --------------------------------------------------

    # Weight noise flag for feed forward layers
    config['weight_noise_ff'] = False

    # Weight noise flag for recurrent layers
    config['weight_noise_rec'] = False

    # Dropout ratio, applied only after readout maxout
    config['dropout'] = 0.5

    # Vocabulary/dataset related ----------------------------------------------

    # Root directory for dataset
    datadir = '/disk/scratch2/s1569734/nmt_data/'

    # Module name of the stream that will be used
    config['stream'] = 'stream'

    # Source and target vocabularies
    config['src_vocab'] = datadir + 'vocab.words-punctuations.words.pkl'
    config['trg_vocab'] = datadir + 'vocab.words-punctuations.punctuations.pkl'

    # Source and target datasets
    config['src_data'] = datadir + 'lm.words'
    config['trg_data'] = datadir + 'lm.punctuations'

    # Source and target vocabulary sizes, should include bos, eos, unk tokens
    config['src_vocab_size'] = 100000
    config['trg_vocab_size'] = 8

    # Special tokens and indexes
    config['unk_id'] = 1
    config['bos_token'] = '<s>'
    config['eos_token'] = '</s>'
    config['unk_token'] = '<unk>'

    # Early stopping based on f1 related ------------------------------------

    # Normalize cost according to sequence length after beam-search
    config['normalized_f1'] = True

    # Use F1 validation
    config['f1_validation'] = True

    # Validation set source file
    config['val_set'] = datadir + 'dev.words'

    # Validation set gold file
    config['val_set_grndtruth'] = datadir + 'dev.punctuations'

    # Print validation output to file
    config['output_val_set'] = True

    # Validation output file
    config['val_set_out'] = config['saveto'] + '/validation_out.txt'

    # Beam-size
    config['beam_size'] = 6

    # Timing/monitoring related -----------------------------------------------

    # Maximum number of updates
    config['finish_after'] = 1000000

    # Reload model from files if exist
    config['reload'] = True

    # Save model after this many updates
    config['save_freq'] = 500

    # Show samples from model after this many updates
    config['sampling_freq'] = 500

    # Show this many samples at each sampling
    config['hook_samples'] = 2

    # Validate f1 after this many updates
    config['f1_val_freq'] = 5000

    # Start f1 validation after this many updates
    config['val_burn_in'] = 5000

    return config
