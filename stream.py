import numpy

from fuel.datasets import H5PYDataset
from fuel.schemes import ConstantScheme
from fuel.streams import DataStream
from fuel.transformers import (Batch, Filter, Padding, SortMapping, Unpack, Mapping)

from six.moves import cPickle


def _length(sentence_pair):
    return max([len(x) for x in sentence_pair])


class PaddingWithEOS(Padding):
    """Padds a stream with given end of sequence idx."""
    def __init__(self, data_stream, padding, **kwargs):
        kwargs['data_stream'] = data_stream
        self.padding = padding
        super(PaddingWithEOS, self).__init__(**kwargs)

    def transform_batch(self, batch):
        data_with_masks = []
        for i, (source, source_data) in enumerate(
                zip(self.data_stream.sources, batch)):
            if source not in self.mask_sources:
                data_with_masks.append(source_data)
                continue

            shapes = [numpy.asarray(sample).shape for sample in source_data]
            lengths = [shape[0] for shape in shapes]
            max_sequence_length = max(lengths)
            rest_shape = shapes[0][1:]
            if not all([shape[1:] == rest_shape for shape in shapes]):
                raise ValueError("All dimensions except length must be equal")
            dtype = numpy.asarray(source_data[0]).dtype

            padded_data = numpy.ones(
                (len(source_data), max_sequence_length) + rest_shape,
                dtype=dtype) * self.padding[source]
            for i, sample in enumerate(source_data):
                padded_data[i, :len(sample)] = sample
            data_with_masks.append(padded_data)

            mask = numpy.zeros((len(source_data), max_sequence_length),
                               self.mask_dtype)
            for i, sequence_length in enumerate(lengths):
                mask[i, :sequence_length] = 1
            data_with_masks.append(mask)

        return tuple(data_with_masks)


class _too_long(object):
    """Filters sequences longer than given sequence length."""
    def __init__(self, seq_len=500):
        self.seq_len = seq_len

    def __call__(self, sentence_pair):
        return max([len(x) for x in sentence_pair]) <= self.seq_len


def get_tr_stream(path, src_eos_idx, phones_sil, tgt_eos_idx, seq_len=50, batch_size=80, sort_k_batches=12, **kwargs):
    """Prepares the training data stream."""

    sources = ('words', 'audio', 'words_ends', 'punctuation_marks', 'phones', 'phones_words_ends', 'phones_words_acoustic_ends')
    #sources = ('words', 'audio', 'words_ends', 'punctuation_marks', 'phones', 'phones_words_ends')
    dataset = H5PYDataset(path, which_sets=('train',), sources=sources, load_in_memory=False)
    print "creating example stream"
    stream = dataset.get_example_stream()
    print "example stream created"

    # Filter sequences that are too long
    stream = Filter(stream, predicate=_too_long(seq_len=seq_len))

    # Build a batched version of stream to read k batches ahead
    stream = Batch(stream, iteration_scheme=ConstantScheme(batch_size*sort_k_batches))

    # Sort all samples in the read-ahead batch
    stream = Mapping(stream, SortMapping(_length))

    # Convert it into a stream again
    stream = Unpack(stream)

    # Construct batches from the stream with specified batch size
    stream = Batch(stream, iteration_scheme=ConstantScheme(batch_size))

    # Pad sequences that are short
    masked_stream = PaddingWithEOS(stream, {
        'words': src_eos_idx,
        'phones': phones_sil,
        'punctuation_marks': tgt_eos_idx,
        'audio': 0,
        'words_ends': -1,
        'phones_words_ends': -1,
        'phones_words_acoustic_ends': -1,
    })

    return masked_stream


def get_dev_stream(path, **kwargs):
    """Setup development set stream if necessary."""

    sources = ('words', 'audio', 'words_ends', 'punctuation_marks', 'phones', 'phones_words_ends', 'phones_words_acoustic_ends', 'text', 'uttids')
    #sources = ('words', 'audio', 'words_ends', 'punctuation_marks', 'phones', 'phones_words_ends', 'text', 'uttids')
    dataset = H5PYDataset(path, which_sets=('dev',), sources=sources)
    return dataset.get_example_stream()
