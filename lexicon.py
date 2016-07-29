def create_dictionary_from_lexicon(path, punctuation_marks):
    dictionary = dict()
    with open(path, 'r') as f:
        for line in f:
            (word, id) = line.strip().split(None, 1)

            if word not in punctuation_marks:
                dictionary[word] = int(id)

    return dictionary

def create_dictionary_from_punctuation_marks(punctuation_marks):
    punctuation_marks = ["<SPACE>"] + punctuation_marks + ["</s>"]
    return dict(zip(punctuation_marks, range(len(punctuation_marks))))
