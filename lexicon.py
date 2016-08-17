def create_dictionary_from_lexicon(path, punctuation_marks):
    words = ["<unk>", "</s>"]
    with open(path, 'r') as f:
        for line in f:
            word = line.strip()

            if word not in punctuation_marks:
                words.append(word)

    return dict(zip(words, range(len(words))))

def create_phone_dictionary_from_lexicon(nonsilence_phones_path, silence_phones_path):
    phones = []
    for path in [nonsilence_phones_path, silence_phones_path]:
        with open(path, 'r') as f:
            for line in f:
                phones.append(line.strip())

    return dict(zip(phones, range(len(phones))))

def create_lexicon(path):
    lexicon = dict()
    with open(path, 'r') as f:
        for line in f:
            (word, pronunciation) = line.strip().split(None, 1)
            lexicon[word] = pronunciation.split()

    return lexicon

def create_dictionary_from_punctuation_marks(punctuation_marks):
    punctuation_marks = ["<SPACE>"] + punctuation_marks + ["</s>"]
    return dict(zip(punctuation_marks, range(len(punctuation_marks))))
