def get_config():
    config = {}
    config['lexicon'] = "/disk/data2/s1569734/bbc_original/data/lang/words.txt"

    config['train_data_dir'] = "/disk/data2/s1569734/bbc_original/data/train_mer10/"
    config['train_alignment_dir'] = "/disk/data2/s1569734/bbc_without_punctuation/exp/alignment/train_mer10_without_punct/"

    config['dev_data_dir'] = "/disk/data2/s1569734/bbc_original/data/dev_for_punctuation_addition/"
    config['dev_alignment_dir'] = "/disk/data2/s1569734/bbc_without_punctuation/exp/alignment/dev_for_punctuation_addition/"

    config['data_dir'] = "/disk/data2/s1569734/acoustic_punctuation/"
    config['punctuation_marks'] = ["<FULL_STOP>", "<COMMA>", "<QUESTION_MARK>", "<EXCLAMATION_MARK>", "<DOTS>"]

    return config
