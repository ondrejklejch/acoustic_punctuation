from __future__ import print_function

import logging
import numpy
import operator
import os
import re
import signal
import time

from blocks.extensions import SimpleExtension
from blocks.serialization import BRICK_DELIMITER
from blocks.search import BeamSearch

from subprocess import Popen, PIPE

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


class SamplingBase(object):
    """Utility class for F1Validator and Sampler."""

    def _get_attr_rec(self, obj, attr):
        return self._get_attr_rec(getattr(obj, attr), attr) \
            if hasattr(obj, attr) else obj

    def _get_true_length(self, seq, vocab):
        try:
            return seq.tolist().index(vocab['</s>']) + 1
        except ValueError:
            return len(seq)

    def _oov_to_unk(self, seq, vocab_size, unk_idx):
        return [x if x < vocab_size else unk_idx for x in seq]

    def _idx_to_word(self, seq, ivocab):
        return " ".join([ivocab.get(idx, "<unk>") for idx in seq])


class Sampler(SimpleExtension, SamplingBase):
    """Random Sampling from model."""

    def __init__(self, model, data_stream, hook_samples=1,
                 src_vocab=None, trg_vocab=None, src_ivocab=None,
                 trg_ivocab=None, src_vocab_size=None, **kwargs):
        super(Sampler, self).__init__(**kwargs)
        self.model = model
        self.hook_samples = hook_samples
        self.data_stream = data_stream
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.src_ivocab = src_ivocab
        self.trg_ivocab = trg_ivocab
        self.src_vocab_size = src_vocab_size
        self.is_synced = False
        self.sampling_fn = model.get_theano_function()

    def do(self, which_callback, *args):

        # Get dictionaries, this may not be the practical way
        sources = self._get_attr_rec(self.main_loop, 'data_stream')

        # Load vocabularies and invert if necessary
        # WARNING: Source and target indices from data stream
        #  can be different
        if not self.src_ivocab:
            self.src_ivocab = {v: k for k, v in self.src_vocab.items()}
        if not self.trg_ivocab:
            self.trg_ivocab = {v: k for k, v in self.trg_vocab.items()}
        if not self.src_vocab_size:
            self.src_vocab_size = len(self.src_vocab)

        # Randomly select source samples from the current batch
        # WARNING: Source and target indices from data stream
        #  can be different
        batch = args[0]
        batch_size = batch['words'].shape[0]
        hook_samples = min(batch_size, self.hook_samples)

        # TODO: this is problematic for boundary conditions, eg. last batch
        sample_idx = numpy.random.choice(batch_size, hook_samples, replace=False)

        words_batch = batch[self.main_loop.data_stream.mask_sources[0]][sample_idx, :]
        audio_batch = batch[self.main_loop.data_stream.mask_sources[1]][sample_idx, :]
        words_ends_batch = batch[self.main_loop.data_stream.mask_sources[2]][sample_idx, :]
        punctuation_marks_batch = batch[self.main_loop.data_stream.mask_sources[3]][sample_idx, :]

        # Sample
        print()
        for i in range(hook_samples):
            length = self._get_true_length(punctuation_marks_batch[i], self.trg_vocab)

            available_inputs = {
                'sampling_words': words_batch[i][:length][None, :],
                'sampling_words_ends': words_ends_batch[i][:length][None, :],
                'sampling_audio': audio_batch[i][:numpy.max(numpy.nonzero(numpy.sum(audio_batch[i], 1))) + 1][None, :],
            }

            inputs = [available_inputs[name] for name in self.model.dict_of_inputs().keys()]

            _1, outputs, _2, _3, costs = (self.sampling_fn(*inputs))
            outputs = outputs.flatten()
            costs = costs.T

            sample_length = self._get_true_length(outputs, self.trg_vocab)
            print("Input : ", self._idx_to_word(words_batch[i][:length], self.src_ivocab))
            print("Target: ", self._idx_to_word(punctuation_marks_batch[i][:length], self.trg_ivocab))
            print("Sample: ", self._idx_to_word(outputs[:sample_length], self.trg_ivocab))
            print("Sample cost: ", costs[:sample_length].sum())
            print()


class F1Validator(SimpleExtension, SamplingBase):
    # TODO: a lot has been changed in NMT, sync respectively
    """Implements early stopping based on F1 score."""

    def __init__(self, source_sentence, samples, model, data_stream,
                 config, n_best=1, track_n_models=1,
                 normalize=True, **kwargs):
        # TODO: change config structure
        super(F1Validator, self).__init__(**kwargs)
        self.source_sentence = source_sentence
        self.samples = samples
        self.model = model
        self.data_stream = data_stream
        self.config = config
        self.n_best = n_best
        self.track_n_models = track_n_models
        self.normalize = normalize
        self.verbose = config.get('val_set_out', None)

        # Helpers
        self.vocab = config["src_vocab"]
        self.unk_sym = config["unk_token"]
        self.eos_sym = config["eos_token"]
        self.trg_vocab = config["trg_vocab"]
        self.trg_ivocab = {v: k for k, v in self.trg_vocab.items()}
        self.trg_eos_idx = self.trg_vocab[config["eos_token"]]
        self.unk_idx = self.vocab[self.unk_sym]
        self.eos_idx = self.vocab[self.eos_sym]
        self.best_models = []
        self.val_f1_curve = []
        self.beam_search = BeamSearch(samples=samples)

        # Create saving directory if it does not exist
        if not os.path.exists(self.config['saveto']):
            os.makedirs(self.config['saveto'])

        if self.config['reload']:
            try:
                f1_score = numpy.load(os.path.join(self.config['saveto'],
                                        'val_f1_scores.npz'))
                self.val_f1_curve = f1_score['f1_scores'].tolist()

                # Track n best previous f1 scores
                for i, f1 in enumerate(
                        sorted(self.val_f1_curve, reverse=True)):
                    if i < self.track_n_models:
                        self.best_models.append(ModelInfo(f1))
                logger.info("F1Scores Reloaded")
            except:
                logger.info("F1Scores not Found")

    def do(self, which_callback, *args):

        # Track validation burn in
        if self.main_loop.status['iterations_done'] <= \
                self.config['val_burn_in']:
            return

        # Evaluate and save if necessary
        self._save_model(self._evaluate_model())

    def _evaluate_model(self):

        logger.info("Started Validation: ")
        val_start_time = time.time()
        total_cost = 0.0

        if self.verbose:
            ftrans = open(self.config['val_set_out'], 'w')

        C = 0
        S = 0
        I = 0
        D = 0
        for i, line in enumerate(self.data_stream.get_epoch_iterator()):
            """
            Load the sentence, retrieve the sample, write to file
            """

            seq = line[0]
            reference = line[1]
            beam_size = min(self.config['beam_size'], len(seq) * (self.config['trg_vocab_size'] - 1))
            input_ = numpy.tile(seq, (beam_size, 1))

            # draw sample, checking to ensure we don't get an empty string back
            trans, costs = \
                self.beam_search.search(
                    input_values={self.source_sentence: input_},
                    max_length=len(seq), eol_symbol=self.trg_eos_idx,
                    ignore_first_eol=True)

            # normalize costs according to the sequence lengths
            if self.normalize:
                lengths = numpy.array([len(s) for s in trans])
                costs = costs / lengths

            nbest_idx = numpy.argsort(costs)[:self.n_best]
            for j, best in enumerate(nbest_idx):
                try:
                    total_cost += costs[best]
                    trans_out = trans[best]

                    # convert idx to words
                    trans_out = self._idx_to_word(trans_out, self.trg_ivocab)
                    reference = self._idx_to_word(reference, self.trg_ivocab)

                except ValueError:
                    logger.info(
                        "Can NOT find a translation for line: {}".format(i+1))
                    trans_out = '<UNK>'

                if j == 0:
                    # Compute F-Measure
                    keywords = ['<FULL_STOP>', '<COMMA>', '<QUESTION_MARK>', '<EXCLAMATION_MARK>', '<DOTS>']

                    merged_tokens = zip(reference.split(), trans_out.split())
                    for (x,y) in merged_tokens:
                        if x == y:
                            if x in keywords:
                                C += 1
                        else:
                            if x in keywords and y in keywords:
                                S += 1
                            elif x not in keywords:
                                I += 1
                            elif y not in keywords:
                                D += 1

                        # If beam returns too short answer
                        if len(reference) > len(trans_out.split()):
                            D += len([w for w in reference[len(trans_out.split()):] if w in keywords])

                    if self.verbose:
                        print(trans_out, file=ftrans)

            if i != 0 and i % 100 == 0:
                f1_score = self.compute_f1_score(C, S, I, D)
                logger.info(
                    "Translated {} lines of validation set... F1 = {}, {}, {}, {}, {}".format(i, f1_score, C, S, I, D))

        logger.info("Total cost of the validation: {}".format(total_cost))
        self.data_stream.reset()
        if self.verbose:
            ftrans.close()

        logger.info("Validation Took: {} minutes".format(float(time.time() - val_start_time) / 60.))

        # extract the score
        f1_score = self.compute_f1_score(C, S, I, D)
        self.val_f1_curve.append(f1_score)
        logger.info(f1_score)

        return f1_score

    def compute_f1_score(self, C, S, I, D):
        C += 0.0001
        precision = float(C) / (C + S + I)
        recall = float(C) / (C + S + D)
        f1 = (2.0 * precision * recall) / (precision + recall)

        return f1

    def _is_valid_to_save(self, f1_score):
        if not self.best_models or min(self.best_models,
           key=operator.attrgetter('f1_score')).f1_score < f1_score:
            return True
        return False

    def _save_model(self, f1_score):
        if self._is_valid_to_save(f1_score):
            model = ModelInfo(f1_score, self.config['saveto'])

            # Manage n-best model list first
            if len(self.best_models) >= self.track_n_models:
                old_model = self.best_models[0]
                if old_model.path and os.path.isfile(old_model.path):
                    logger.info("Deleting old model %s" % old_model.path)
                    os.remove(old_model.path)
                self.best_models.remove(old_model)

            self.best_models.append(model)
            self.best_models.sort(key=operator.attrgetter('f1_score'))

            # Save the model here
            s = signal.signal(signal.SIGINT, signal.SIG_IGN)
            logger.info("Saving new model {}".format(model.path))
            params_to_save = self.main_loop.model.get_parameter_values()
            param_values = {name.replace("/", BRICK_DELIMITER): param for name, param in params_to_save.items()}
            numpy.savez(model.path, **param_values)


            numpy.savez(
                os.path.join(self.config['saveto'], 'val_f1_scores.npz'),
                f1_scores=self.val_f1_curve)
            signal.signal(signal.SIGINT, s)


class ModelInfo:
    """Utility class to keep track of evaluated models."""

    def __init__(self, f1_score, path=None):
        self.f1_score = f1_score
        self.path = self._generate_path(path)

    def _generate_path(self, path):
        gen_path = os.path.join(
            path, 'best_f1_model_%d_F1%.2f.npz' %
            (int(time.time()), self.f1_score) if path else None)
        return gen_path
