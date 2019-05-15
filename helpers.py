import logging
import numpy as np
import theano
from theano import tensor
from toolz import merge

from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.initialization import IsotropicGaussian, Orthogonal, Constant
from blocks.model import Model
from blocks.select import Selector

from model import BidirectionalEncoder, BidirectionalAudioEncoder, BidirectionalPhonesEncoder, BidirectionalPhonemeAudioEncoder, Decoder
from cost import stimulation_cost

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

rs = np.random.RandomState(1234)
rng = tensor.shared_randomstreams.RandomStreams(rs.randint(999999))

def create_model(config):
    if config["input"] == "words":
        encoder, training_representation, sampling_representation = create_word_encoder(config)
        models = [encoder]
    elif config["input"] == "audio":
        encoder, training_representation, sampling_representation = create_audio_encoder(config)
        models = [encoder]
    elif config["input"] == "phones":
        encoder, training_representation, sampling_representation = create_phones_encoder(config)
        models = [encoder]
    elif config["input"] == "phones-audio":
        encoder, training_representation, sampling_representation = create_phones_audio_encoder(config)
        models = [encoder]
    elif config["input"] == "both":
        words_encoder, words_training_representation, words_sampling_representation = create_word_encoder(config)
        audio_encoder, audio_training_representation, audio_sampling_representation = create_audio_encoder(config)

        def merge_representations(words, audio, train=True):
            if config["combination"] == "max":
                return tensor.max(tensor.stack([words, audio], axis=0), axis=0)
            if config["combination"] == "dropout-max":
                max = tensor.max(tensor.stack([words, audio], axis=0), axis=0)
                p = 0.5
                if train is True:
                    mask = tensor.extra_ops.repeat(rng.binomial(n=1, p=p, size=(1, words.shape[1], words.shape[2]), dtype=theano.config.floatX), words.shape[0], axis=0)
                    return mask * max
                else:
                    return p * max
            if config["combination"] == "avg":
                return tensor.mean(tensor.stack([words, audio], axis=0), axis=0)
            if config["combination"] == "add":
                return words + audio
            if config["combination"] == "dropout-add":
                p = 0.5
                if train is True:
                    mask = tensor.extra_ops.repeat(rng.binomial(n=1, p=p, size=(1, words.shape[1], words.shape[2]), dtype=theano.config.floatX), words.shape[0], axis=0)
                    return mask * (words + audio)
                else:
                    return p * (words + audio)
            if config["combination"] == "concat":
                return tensor.concatenate([words, audio], axis=2)
            if config["combination"] == "mask":
                p = 0.5
                if train is True:
                    #mask = rng.binomial(n=1, p=p, size=words.shape, dtype=theano.config.floatX)
                    mask = tensor.extra_ops.repeat(rng.binomial(n=1, p=p, size=(1, words.shape[1], words.shape[2]), dtype=theano.config.floatX), words.shape[0], axis=0)

                    return mask * words + (1-mask) * audio
                else:
                    return p * words + (1-p) * audio


        training_representation = merge_representations(words_training_representation, audio_training_representation)
        sampling_representation = merge_representations(words_sampling_representation, audio_sampling_representation, False)
        models = [words_encoder, audio_encoder]

    decoder, cost, samples, search_model, punctuation_marks, mask = create_decoder(config, training_representation, sampling_representation)

    # Add stimulation cost
    #weights = decoder.children[0].children[2].children[1].children[1].parameters[0]
    ##cost = cost + stimulation_cost(32, training_representation, weights, punctuation_marks)
    #cost = stimulation_cost(16, training_representation, weights, punctuation_marks, mask)
    #cost.name = "stimulated_cost"

    print_parameteters(models + [decoder])

    return cost, samples, search_model

def create_multitask_model(config):
    words_encoder, words_training_representation, words_sampling_representation = create_word_encoder(config)
    audio_encoder, audio_training_representation, audio_sampling_representation = create_audio_encoder(config)
    models = [words_encoder, audio_encoder]

    decoder, words_cost, words_samples, words_search_model, _, _ = create_decoder(config, words_training_representation, words_sampling_representation)
    audio_cost, audio_samples, audio_search_model, _, _ = use_decoder_on_representations(decoder, audio_training_representation, audio_sampling_representation)

    print_parameteters(models + [decoder])
    words_cost = words_cost + audio_cost

    cost = words_cost + audio_cost
    cost.name = "cost"

    return cost, words_samples, words_search_model



def create_word_encoder(config):
    encoder = BidirectionalEncoder(config['src_vocab_size'], config['enc_embed'], config['enc_nhids'])
    encoder.weights_init = IsotropicGaussian(config['weight_scale'])
    encoder.biases_init = Constant(0)
    encoder.push_initialization_config()
    encoder.bidir.prototype.weights_init = Orthogonal()
    encoder.initialize()

    input_words = tensor.lmatrix('words')
    input_words_mask = tensor.matrix('words_mask')
    training_representation = encoder.apply(input_words, input_words_mask)
    training_representation.name = "words_representation"

    sampling_input_words = tensor.lmatrix('sampling_words')
    sampling_input_words_mask = tensor.ones((sampling_input_words.shape[0], sampling_input_words.shape[1]))
    sampling_representation = encoder.apply(sampling_input_words, sampling_input_words_mask)

    return encoder, training_representation, sampling_representation

def create_audio_encoder(config):
    encoder = BidirectionalAudioEncoder(config['audio_feat_size'], config['enc_embed'], config['enc_nhids'])
    encoder.weights_init = IsotropicGaussian(config['weight_scale'])
    encoder.biases_init = Constant(0)
    encoder.push_initialization_config()
    encoder.bidir.prototype.weights_init = Orthogonal()
    encoder.embedding.prototype.weights_init = Orthogonal()
    encoder.initialize()

    audio = tensor.ftensor3('audio')
    audio_mask = tensor.matrix('audio_mask')
    words_ends = tensor.lmatrix('words_ends')
    words_ends_mask = tensor.matrix('words_ends_mask')
    training_representation = encoder.apply(audio, audio_mask, words_ends, words_ends_mask)
    training_representation.name = "audio_representation"

    sampling_audio = tensor.ftensor3('sampling_audio')
    sampling_audio_mask = tensor.ones((sampling_audio.shape[0], sampling_audio.shape[1]))
    sampling_words_ends = tensor.lmatrix('sampling_words_ends')
    sampling_words_ends_mask = tensor.ones((sampling_words_ends.shape[0], sampling_words_ends.shape[1]))
    sampling_representation = encoder.apply(sampling_audio, sampling_audio_mask, sampling_words_ends, sampling_words_ends_mask)

    return encoder, training_representation, sampling_representation

def create_phones_encoder(config):
    encoder = BidirectionalPhonesEncoder(config['phones_vocab_size'], config['enc_embed'], config['enc_nhids'])
    encoder.weights_init = IsotropicGaussian(config['weight_scale'])
    encoder.biases_init = Constant(0)
    encoder.push_initialization_config()
    encoder.bidir.prototype.weights_init = Orthogonal()
    encoder.embedding.prototype.weights_init = Orthogonal()
    encoder.initialize()

    phones = tensor.lmatrix('phones')
    phones_mask = tensor.matrix('phones_mask')
    phones_words_ends = tensor.lmatrix('phones_words_ends')
    phones_words_ends_mask = tensor.matrix('phones_words_ends_mask')
    training_representation = encoder.apply(phones, phones_mask, phones_words_ends, phones_words_ends_mask)
    training_representation.name = "phones_representation"

    sampling_phones = tensor.lmatrix('sampling_phones')
    sampling_phones_mask = tensor.ones((sampling_phones.shape[0], sampling_phones.shape[1]))
    sampling_phones_words_ends = tensor.lmatrix('sampling_phones_words_ends')
    sampling_phones_words_ends_mask = tensor.ones((sampling_phones_words_ends.shape[0], sampling_phones_words_ends.shape[1]))
    sampling_representation = encoder.apply(sampling_phones, sampling_phones_mask, sampling_phones_words_ends, sampling_phones_words_ends_mask)

    return encoder, training_representation, sampling_representation

def create_phones_audio_encoder(config):
    encoder = BidirectionalPhonemeAudioEncoder(config['audio_feat_size'], config['enc_embed'], config['enc_nhids'])
    encoder.weights_init = IsotropicGaussian(config['weight_scale'])
    encoder.biases_init = Constant(0)
    encoder.push_initialization_config()
    encoder.audio_embedding.prototype.weights_init = Orthogonal()
    encoder.phoneme_embedding.prototype.weights_init = Orthogonal()
    encoder.words_embedding.prototype.weights_init = Orthogonal()
    encoder.initialize()

    audio = tensor.ftensor3('audio')
    audio_mask = tensor.matrix('audio_mask')
    phones_words_acoustic_ends = tensor.lmatrix('phones_words_acoustic_ends')
    phones_words_acoustic_ends_mask = tensor.matrix('phones_words_acoustic_ends_mask')
    phones_words_ends = tensor.lmatrix('phones_words_ends')
    phones_words_ends_mask = tensor.matrix('phones_words_ends_mask')
    training_representation = encoder.apply(audio, audio_mask, phones_words_acoustic_ends, phones_words_acoustic_ends_mask, phones_words_ends, phones_words_ends_mask)
    training_representation.name = "phones_representation"

    sampling_audio = tensor.ftensor3('sampling_audio')
    sampling_audio_mask = tensor.ones((sampling_audio.shape[0], sampling_audio.shape[1]))
    sampling_phones_words_acoustic_ends = tensor.lmatrix('sampling_phones_words_acoustic_ends')
    sampling_phones_words_acoustic_ends_mask = tensor.ones((sampling_phones_words_acoustic_ends.shape[0], sampling_phones_words_acoustic_ends.shape[1]))
    sampling_phones = tensor.lmatrix('sampling_phones')
    sampling_phones_mask = tensor.ones((sampling_phones.shape[0], sampling_phones.shape[1]))
    sampling_phones_words_ends = tensor.lmatrix('sampling_phones_words_ends')
    sampling_phones_words_ends_mask = tensor.ones((sampling_phones_words_ends.shape[0], sampling_phones_words_ends.shape[1]))
    sampling_representation = encoder.apply(
        sampling_audio, sampling_audio_mask, sampling_phones_words_acoustic_ends,
        sampling_phones_words_acoustic_ends_mask, sampling_phones_words_ends, sampling_phones_words_ends_mask)

    return encoder, training_representation, sampling_representation

def create_decoder(config, training_representation, sampling_representation):
    if config["combination"] == 'concat':
        enc_nhids = config["enc_nhids"] * 4
    else:
        enc_nhids = config["enc_nhids"] * 2

    decoder = Decoder(config['trg_vocab_size'], config['dec_embed'], config['dec_nhids'], enc_nhids)
    decoder.weights_init = IsotropicGaussian(config['weight_scale'])
    decoder.biases_init = Constant(0)
    decoder.push_initialization_config()
    decoder.transition.weights_init = Orthogonal()
    decoder.initialize()

    cost, samples, search_model, punctuation_marks, mask = use_decoder_on_representations(decoder, training_representation, sampling_representation)

    return decoder, cost, samples, search_model, punctuation_marks, mask

def use_decoder_on_representations(decoder, training_representation, sampling_representation):
    punctuation_marks = tensor.lmatrix('punctuation_marks')
    punctuation_marks_mask = tensor.matrix('punctuation_marks_mask')
    cost = decoder.cost(training_representation, punctuation_marks_mask, punctuation_marks, punctuation_marks_mask)

    generated = decoder.generate(sampling_representation)
    search_model = Model(generated)
    _, samples = VariableFilter(bricks=[decoder.sequence_generator], name="outputs")(ComputationGraph(generated[1]))

    return cost, samples, search_model, punctuation_marks, punctuation_marks_mask


def print_parameteters(models):
    param_dict = merge(*[Selector(model).get_parameters() for model in models])
    number_of_parameters = 0

    logger.info("Parameter names: ")
    for name, value in param_dict.items():
        number_of_parameters += np.product(value.get_value().shape)
        logger.info('    {:15}: {}'.format(value.get_value().shape, name))
    logger.info("Total number of parameters: {}".format(number_of_parameters))
