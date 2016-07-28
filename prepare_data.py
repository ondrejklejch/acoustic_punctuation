import h5py
import kaldi_io
import numpy as np

from collections import defaultdict
from config import get_config
from fuel.datasets.hdf5 import H5PYDataset
from lexicon import create_dictionary_from_lexicon, create_dictionary_from_punctuation_marks

def get_uttids_from_text_file(path):
    uttids = set()
    with open(path, 'r') as f:
        for line in f:
            (uttid, _) = line.strip().split(None, 1)
            uttids.add(uttid)

    return uttids

def get_utterances_from_text_file(path, punctuation_marks):
    with open(path, 'r') as f:
        for line in f:
            (uttid, text) = line.strip().split(None, 1)
            (utt_words, utt_punctuation_marks) = text_to_words_and_punctuation_marks(text, punctuation_marks)

            yield uttid, utt_words, utt_punctuation_marks

def text_to_words_and_punctuation_marks(text, punctuation_marks):
    output_words = []
    output_punctuation_marks = []

    words = text.split()
    for (word, punctuation_mark) in zip(words, words[1:] + [None]):
        if word in punctuation_marks:
            continue

        if punctuation_mark not in punctuation_marks:
            punctuation_mark = "<SPACE>"

        output_words.append(word)
        output_punctuation_marks.append(punctuation_mark)

    return output_words, output_punctuation_marks

def get_audio_features_from_file(path):
    return kaldi_io.SequentialBaseFloatMatrixReader(path)

def get_time_boundaries_from_ctm_file(path):
    time_boundaries = defaultdict(lambda: [])

    with open(path, 'r') as f:
        for line in f:
            (uttid, _, start, duration, _) = line.strip().split()
            time_boundaries[uttid].append(int((float(start) + float(duration)) * 100))

    return time_boundaries.iteritems()

def create_numpy_array_dataset(h5file, name, num_utts, ndim, dtype):
    shapes = h5file.create_dataset("%s_shapes" % name, (num_utts, ndim), dtype='int32')
    shape_labels = h5file.create_dataset("%s_shape_labels" % name, (ndim,), dtype='S7')
    shape_labels[...] = ['frame'.encode('utf8'), 'feature'.encode('utf8')][:ndim]

    dataset = h5file.create_dataset(name, (num_utts,), dtype=h5py.special_dtype(vlen=dtype))
    dataset.dims[0].label = 'batch'
    dataset.dims.create_scale(shapes, 'shapes')
    dataset.dims[0].attach_scale(shapes)

    dataset.dims.create_scale(shape_labels, 'shape_labels')
    dataset.dims[0].attach_scale(shape_labels)

    return shapes, dataset


if __name__ == "__main__":
    config = get_config()
    data_file = "%s/data.h5" % config["data_dir"]
    datasets = ["train", "dev"]

    with h5py.File(data_file, 'a') as h5file:
        words_dictionary = create_dictionary_from_lexicon(config["lexicon"], config["punctuation_marks"])
        punctuation_marks_dictionary = create_dictionary_from_punctuation_marks(config["punctuation_marks"])

        uttids = []
        for dataset in datasets:
            data_dir = config["%s_data_dir" % dataset]
            uttids.extend(get_uttids_from_text_file("%s/text" % data_dir))

        uttids = dict(zip(uttids, range(len(uttids))))
        num_utts = len(uttids)
        words_shapes, words_dataset = create_numpy_array_dataset(h5file, '%s_words' % dataset, num_utts, 1, 'int32')
        punctuation_marks_shapes, punctuation_marks_dataset = create_numpy_array_dataset(h5file, '%s_punctuation_marks' % dataset, num_utts, 1, 'int8')
        audio_shapes, audio = create_numpy_array_dataset(h5file, '%s_audio' % dataset, num_utts, 2, 'float32')
        words_ends_shapes, words_ends = create_numpy_array_dataset(h5file, '%s_words_ends' % dataset, num_utts, 1, 'int16')

        for dataset in datasets:
            data_dir = config["%s_data_dir" % dataset]

            for (uttid, words, punctuation_marks) in get_utterances_from_text_file("%s/text" % data_dir, config["punctuation_marks"]):
                if uttid not in uttids:
                    print "Text %s not in uttids" % uttid
                    continue

                uttid = uttids[uttid]

                words = np.array([words_dictionary.get(word, words_dictionary["<unk>"]) for word in words], dtype=np.int32)
                words_shapes[uttid] = words.shape
                words_dataset[uttid] = words

                punctuation_marks = np.array([punctuation_marks_dictionary[punctuation_mark] for punctuation_mark in punctuation_marks], dtype=np.int8)
                punctuation_marks_shapes[uttid] = punctuation_marks.shape
                punctuation_marks_dataset[uttid] = punctuation_marks

            for (uttid, features) in get_audio_features_from_file("scp:%s/feats.scp" % data_dir):
                if uttid not in uttids:
                    print "audio %s not in uttids" % uttid
                    continue

                uttid = uttids[uttid]
                audio_shapes[uttid] = features.shape
                audio[uttid] = features.ravel()

            alignment_dir = config["%s_alignment_dir" % dataset]
            for (uttid, boundaries) in get_time_boundaries_from_ctm_file("%s/forced_alignment.txt" % alignment_dir):
                if uttid not in uttids:
                    print "forced alignment %s not in uttids" % uttid
                    continue

                uttid = uttids[uttid]
                boundaries = np.array(boundaries, dtype=np.int16)
                words_ends_shapes[uttid] = boundaries.shape
                words_ends[uttid] = boundaries.ravel()

            print "Dataset %s processed" % dataset

        sources = []
        for dataset in h5file:
            if (dataset.endswith('_indices') or dataset.endswith('_shapes') or
                dataset.endswith('_shape_labels')):
                continue
            sources.append(dataset)

        split_dict = {}
        for dataset in datasets:
            idxs = []
            data_dir = config["%s_data_dir" % dataset]
            for uttid in get_uttids_from_text_file("%s/text" % data_dir):
                if(audio_shapes[uttids[uttid]][1] == 43):
                    idxs.append(uttids[uttid])

            indices_name = "%s_indices" % dataset
            h5file[indices_name] = np.array(idxs)
            indices_ref = h5file[indices_name].ref
            split_dict[dataset] = dict([(source, (-1, -1, indices_ref)) for source in sources])

        h5file.attrs['split'] = H5PYDataset.create_split_array(split_dict)

    print "Done."
