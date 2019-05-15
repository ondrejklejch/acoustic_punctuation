# Acoustic punctuation

This repository contains code for our ICASSP 2017 paper, in which we explored combining acoustic and lexical features for punctuation prediction using a neural machine translation approach with a hierarchical encoder that maps frame level acoustic features into word level acoustic embeddings. Note that this repository is intended only for research purposes as we found that purely lexical neural machine translation based system trained on large amounts of text data performs better in production.

```
@inproceedings{klejch2017sequence,
  title={Sequence-to-sequence models for punctuated transcription combining lexical and acoustic features},
  author={Klejch, Ond{\v{r}}ej and Bell, Peter and Renals, Steve},
  booktitle={2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2017},
  organization={IEEE}
}
```

## Usage

This repository is using Theano and Blocks and is built on top of the [Blocks NMT example](https://github.com/mila-iqia/blocks-examples/tree/master/machine_translation). In order to train and evaluate the model you will need to perform the following steps:

1. Prepare train, dev and test data directories in [Kaldi format](https://kaldi-asr.org/doc/data_prep.html) and obtain an phoneme level alignment in the ctm format using some pretrained ASR system.
2. Decode dev and test data using the pretrained ASR system and generate corresponding phoneme level alignment.
3. Update `config.py` with correct paths and model settings:
    ```
        KALDI_EXP_ROOT = "Set path to your KALDI exp root."
        config = {}
        config['vocabulary'] = "%s/data/local/dict/mgb.150k.wlist" % KALDI_EXP_ROOT
        config['lexicon'] = create_lexicon("%s/data/local/dict/lexicon.txt" % KALDI_EXP_ROOT)
        config['phones'] = create_phone_dictionary_from_lexicon(
            "%s/data/local/dict/nonsilence_phones.txt" % KALDI_EXP_ROOT,
            "%s/data/local/dict/silence_phones.txt" % KALDI_EXP_ROOT)
        config['phones_vocab_size'] = len(config['phones'])
        config['punctuation_marks'] = ["<FULL_STOP>", "<COMMA>", "<QUESTION_MARK>", "<EXCLAMATION_MARK>", "<DOTS>"]
        config['train_data_dir'] = "%s/data/train/" % KALDI_EXP_ROOT
        config['train_alignment_dir'] = "%s/exp/ali_train/" % KALDI_EXP_ROOT
        config['dev_data_dir'] = "%s/data/dev/" % KALDI_EXP_ROOT
        config['dev_alignment_dir'] = "%s/exp/ali_dev/" % KALDI_EXP_ROOT
        config['best_asr_data_dir'] = "%s/data/dev_asr/" % KALDI_EXP_ROOT
        config['best_asr_alignment_dir'] = "%s/exp/ali_dev_asr/" % KALDI_EXP_ROOT
        config['data_dir'] = "./data"
    ```
3. Prepare data files using `python prepare_data.py`.
4. Train the system using `python __main__.py`.
5. Punctuate dev data by updating the `config` section in `translate.py` and running `python translate.py`.
