import logging
import numpy
import os
import pickle

import theano
from theano import tensor

from blocks.model import Model
from blocks.select import Selector
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.search import BeamSearch

from collections import OrderedDict
from helpers import create_model
from model import BidirectionalEncoder, Decoder
from stream import get_dev_stream
from sampling import SamplingBase
from checkpoint import LoadNMT


logger = logging.getLogger(__name__)
theano.config.on_unused_input = 'warn'

def tile(x, beam_size):
    return numpy.tile(x, (beam_size,) + (1,) * x.ndim)

def main(config, model_dir, model_filename, data_path, input, output):
    logger.info("Loading the model..")
    cost, samples, search_model = create_model(config)
    loader = LoadNMT(model_dir, model_filename)
    loader.set_model_parameters(search_model, loader.load_parameters())
    beam_search = BeamSearch(samples=samples)

    # Get test set stream
    test_stream = get_dev_stream(data_path)
    ftrans = open(output, 'w')

    # Helper utilities
    sutils = SamplingBase()
    trg_eos_idx = config['trg_eos_idx']
    trg_ivocab = {v: k for k, v in config["trg_vocab"].items()}
    src_ivocab = {v: k for k, v in config["src_vocab"].items()}


    logger.info("Started translation: ")
    total_cost = 0.0

    for i, line in enumerate(test_stream.get_epoch_iterator()):
        beam_size = config['beam_size']
        available_inputs = dict(zip(["sampling_%s" % x for x in test_stream.sources], line))
        input_values = OrderedDict([(tensor, tile(available_inputs[name], beam_size)) for (name, tensor) in search_model.dict_of_inputs().iteritems()])
        seq = available_inputs["sampling_words"]
        original = available_inputs["sampling_text"]
        uttid = available_inputs["sampling_uttids"]

        # draw sample, checking to ensure we don't get an empty string back
        trans, costs = \
            beam_search.search(
                input_values=input_values,
                max_length=len(seq) + 2, eol_symbol=trg_eos_idx,
                ignore_first_eol=True)

        # normalize costs according to the sequence lengths
        lengths = numpy.array([len(s) for s in trans])
        costs = costs / lengths

        best = numpy.argsort(costs)[0]
        try:
            total_cost += costs[best]
            trans_out = trans[best]

            # convert idx to words
            trans_out = sutils._idx_to_word(trans_out, trg_ivocab)

        except ValueError:
            logger.info("Can NOT find a translation for line: {}".format(i+1))
            trans_out = '<UNK>'

        source_words = original.strip().split()[:-1]
        target_words = trans_out.split()

        output = []
        for (word, punct) in zip(source_words, target_words):
            if punct in config["punctuation_marks"]:
                output.append(word)
                output.append(punct)
            else:
                output.append(word)

        if len(source_words) > len(target_words):
            output.extend(source_words[len(target_words):])


        print uttid, " ".join(output)
        print >> ftrans, uttid, " ".join(output)

        if i != 0 and i % 100 == 0:
            logger.info("Translated {} lines of test set...".format(i))

    logger.info("Total cost of the test: {}".format(total_cost))
    ftrans.close()


if __name__ == "__main__":
    from config import get_config
    config = get_config()

    config["input"] = "both"
    config["combination"] = "dropout-add"
    config['audio_feat_size'] = 43
    model_dir = "/disk/scratch2/s1569734/acoustic_punctuation/nmt_punctuation_on_both_dropout-add/"
    model_filename = "best_f1_model_1473242935_F10.59.npz"
    data_path = "%s/data_global_cmvn_with_phones_alignment.h5" % config["data_dir"]

    main(config, model_dir, model_filename, data_path, "../nmt_punctuation/dev_raw.txt", "punctuated_dev.txt")
