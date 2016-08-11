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

    return list(uttids)

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

    return output_words + ["</s>"], output_punctuation_marks + ["</s>"]

def get_mean_std_from_audio_features(path):
    sum = np.zeros((43,))
    sum_sq = np.zeros((43,))
    n = 0

    with kaldi_io.SequentialBaseFloatMatrixReader(path) as reader:
        for name,feats in reader:
            nframes, nfeats = feats.shape
            n += nframes
            sum += feats.sum(0)
            sum_sq += (feats*feats).sum(0)

    mean = np.asarray(sum/n, dtype=kaldi_io.KALDI_BASE_FLOAT())
    std = np.asarray(np.sqrt(sum_sq/n - mean**2), dtype=kaldi_io.KALDI_BASE_FLOAT())

    return mean, std

def get_audio_features_from_file(path, take_every_nth, mean, std):
    for (uttid, features) in kaldi_io.SequentialBaseFloatMatrixReader(path):
        features = features[::take_every_nth]
        features = (features - mean) / std

        yield uttid, features

def get_time_boundaries_from_ctm_file(path, take_every_nth):
    time_boundaries = defaultdict(lambda: [])

    with open(path, 'r') as f:
        for line in f:
            (uttid, _, start, duration, _) = line.strip().split()
            time_boundaries[uttid].append((int((float(start) + float(duration)) * 100) / take_every_nth) - 1)

    for uttid in time_boundaries:
        time_boundaries[uttid].append(-1)

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
    data_file = "%s/data_global_cmvn.h5" % config["data_dir"]
    datasets = ["train", "dev"]

    with h5py.File(data_file, 'a') as h5file:
        words_dictionary = config["src_vocab"]
        punctuation_marks_dictionary = config["trg_vocab"]

        uttids = []
        for dataset in datasets:
            data_dir = config["%s_data_dir" % dataset]
            uttids.extend(get_uttids_from_text_file("%s/text" % data_dir))

        uttids = dict(zip(uttids, range(len(uttids))))
        num_utts = len(uttids)

        text = h5file.create_dataset('text', (num_utts,), dtype=h5py.special_dtype(vlen=unicode))
        uttids_dataset = h5file.create_dataset('uttids', (num_utts,), dtype=h5py.special_dtype(vlen=unicode))
        words_shapes, words_dataset = create_numpy_array_dataset(h5file, 'words', num_utts, 1, 'int32')
        punctuation_marks_shapes, punctuation_marks_dataset = create_numpy_array_dataset(h5file, 'punctuation_marks', num_utts, 1, 'int8')
        audio_shapes, audio = create_numpy_array_dataset(h5file, 'audio', num_utts, 2, 'float32')
        words_ends_shapes, words_ends = create_numpy_array_dataset(h5file, 'words_ends', num_utts, 1, 'int16')


        mean, std = get_mean_std_from_audio_features("scp:%s/feats.scp" % config["train_data_dir"])
        for dataset in datasets:
            data_dir = config["%s_data_dir" % dataset]

            for (uttid, words, punctuation_marks) in get_utterances_from_text_file("%s/text" % data_dir, config["punctuation_marks"]):
                if uttid not in uttids:
                    print "Text %s not in uttids" % uttid
                    continue

                text_uttid = uttid
                uttid = uttids[uttid]

                text[uttid] = " ".join(words)
                uttids_dataset[uttid] = text_uttid
                words = np.array([words_dictionary.get(word, words_dictionary["<unk>"]) for word in words], dtype=np.int32)
                words_shapes[uttid] = words.shape
                words_dataset[uttid] = words

                punctuation_marks = np.array([punctuation_marks_dictionary[punctuation_mark] for punctuation_mark in punctuation_marks], dtype=np.int8)
                punctuation_marks_shapes[uttid] = punctuation_marks.shape
                punctuation_marks_dataset[uttid] = punctuation_marks

            for (uttid, features) in get_audio_features_from_file("scp:%s/feats.scp" % data_dir, config["take_every_nth"], mean, std):
                if uttid not in uttids:
                    print "audio %s not in uttids" % uttid
                    continue

                uttid = uttids[uttid]
                audio_shapes[uttid] = features.shape
                audio[uttid] = features.ravel()

            alignment_dir = config["%s_alignment_dir" % dataset]
            for (uttid, boundaries) in get_time_boundaries_from_ctm_file("%s/forced_alignment.txt" % alignment_dir, config["take_every_nth"]):
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
                if uttid in uttids and (audio_shapes[uttids[uttid]][1] == 43):
                    idxs.append(uttids[uttid])

                if uttid in uttids and len(words_ends[uttids[uttid]]) == 0:
                    uttid = uttids[uttid]

                    words = words_shapes[uttid][0]
                    frames = audio_shapes[uttid][0]
                    step = float(frames - 1) / words
                    boundaries = np.array([int(step * (i+1)) for i in range(words)], dtype=np.int16)
                    words_ends_shapes[uttid] = boundaries.shape
                    words_ends[uttid] = boundaries.ravel()

            indices_name = "%s_indices" % dataset
            h5file[indices_name] = np.array(idxs)
            indices_ref = h5file[indices_name].ref
            split_dict[dataset] = dict([(source, (-1, -1, indices_ref)) for source in sources])

        h5file.attrs['split'] = H5PYDataset.create_split_array(split_dict)

    print "Done."
