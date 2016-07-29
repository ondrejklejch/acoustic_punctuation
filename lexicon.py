def create_dictionary_from_lexicon(path, punctuation_marks):
    words = ["<unk>", "</s>"]
    with open(path, 'r') as f:
        for line in f:
            word = line.strip()

            if word not in punctuation_marks:
                words.append(word)

    return dict(zip(words, range(len(words))))

def create_dictionary_from_punctuation_marks(punctuation_marks):
    punctuation_marks = ["<SPACE>"] + punctuation_marks + ["</s>"]
    return dict(zip(punctuation_marks, range(len(punctuation_marks))))
