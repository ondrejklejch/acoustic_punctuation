"""Encoder-Decoder with search for machine translation.

In this demo, encoder-decoder architecture with attention mechanism is used for
machine translation. The attention mechanism is implemented according to
[BCB]_. The training data used is WMT15 Czech to English corpus, which you have
to download, preprocess and put to your 'datadir' in the config file. Note
that, you can use `prepare_data.py` script to download and apply all the
preprocessing steps needed automatically.  Please see `prepare_data.py` for
further options of preprocessing.

.. [BCB] Dzmitry Bahdanau, Kyunghyun Cho and Yoshua Bengio. Neural
   Machine Translation by Jointly Learning to Align and Translate.
"""

import argparse
import logging
import pprint

import config

from __init__ import main
from lexicon import create_dictionary_from_lexicon, create_dictionary_from_punctuation_marks
from stream import get_tr_stream, get_dev_stream

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

# Get the arguments
parser = argparse.ArgumentParser()
parser.add_argument("--proto",  default="get_config", help="Prototype config to use for config")
parser.add_argument("--bokeh",  default=False, action="store_true", help="Use bokeh server for plotting")
args = parser.parse_args()


if __name__ == "__main__":
    config = getattr(config, args.proto)()
    #logger.info("Model options:\n{}".format(pprint.pformat(config)))

    data_path = "%s/data_global_cmvn_with_phones.h5" % config["data_dir"]
    tr_stream = get_tr_stream(data_path, config["src_eos_idx"], config["phones"]["sil"], config["trg_eos_idx"], seq_len=config["seq_len"], batch_size=config["batch_size"], sort_k_batches=config["sort_k_batches"])
    dev_stream = get_dev_stream(data_path)
    main(config, tr_stream, dev_stream, args.bokeh)
