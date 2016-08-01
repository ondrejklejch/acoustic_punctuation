import numpy

from fuel.datasets import H5PYDataset
from fuel.schemes import ConstantScheme
from fuel.streams import DataStream
from fuel.transformers import (Batch, Filter, Padding, SortMapping, Unpack, Mapping)

from six.moves import cPickle


def _length(sentence_pair):
    """Assumes target is the last element in the tuple."""
    return len(sentence_pair[-1])


class PaddingWithEOS(Padding):
    """Padds a stream with given end of sequence idx."""
    def __init__(self, data_stream, eos_idx, **kwargs):
        kwargs['data_stream'] = data_stream
        self.eos_idx = eos_idx
        super(PaddingWithEOS, self).__init__(**kwargs)

    def get_data_from_batch(self, request=None):
        if request is not None:
            raise ValueError
        data = list(next(self.child_epoch_iterator))
        data_with_masks = []
        for i, (source, source_data) in enumerate(
                zip(self.data_stream.sources, data)):
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
                dtype=dtype) * self.eos_idx[i]
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
    def __init__(self, seq_len=50):
        self.seq_len = seq_len

    def __call__(self, sentence_pair):
        return all([len(sentence) <= self.seq_len for sentence in sentence_pair])


def get_tr_stream(path, src_eos_idx, tgt_eos_idx, seq_len=50, batch_size=80, sort_k_batches=12, **kwargs):
    """Prepares the training data stream."""


    dataset = H5PYDataset(path, which_sets=('train',), sources=('words', 'punctuation_marks'), load_in_memory=False)
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
    masked_stream = PaddingWithEOS(stream, [src_eos_idx, tgt_eos_idx])

    return masked_stream


def get_dev_stream(path, **kwargs):
    """Setup development set stream if necessary."""

    dataset = H5PYDataset(path, which_sets=('dev',), sources=('words', 'punctuation_marks'))
    return dataset.get_example_stream()
