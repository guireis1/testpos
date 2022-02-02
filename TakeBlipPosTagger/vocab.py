import csv
import logging
from take_text_preprocess.presentation import pre_process

from .process_option import PREPROCESS_OPTION


class Vocabulary(object):
    def __init__(self):
        self.f2i = {}
        self.i2f = {}

    def add(self, word, ignore_duplicates=True):
        if word in self.f2i:
            if not ignore_duplicates:
                raise ValueError(f"'{word}' already exists")
            return self.f2i[word]
        idx = len(self.f2i)
        self.f2i[word] = idx
        self.i2f[idx] = word
        return self.f2i[word]

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.i2f[item]
        elif isinstance(item, str):
            return self.f2i[item]
        elif hasattr(item, "__iter__"):
            return [self[ele] for ele in item]
        else:
            raise ValueError(f"Unknown type: {type(item)}")

    def __contains__(self, item):
        return item in self.f2i or item in self.i2f

    def __len__(self):
        return len(self.f2i)


def populate_vocab(sentences, vocab):
    for sentence in sentences:
        for word in sentence:
            vocab.add(word)


def line_pre_process(line, use_pre_processing):
    if use_pre_processing:
        return pre_process(line, PREPROCESS_OPTION).split()
    else:
        return line.split()


def create_vocabulary(input_path, column_name, pad_string, unk_string,
                      encoding, separator, use_pre_processing=False,
                      is_label=False, sentences=None):
    vocabulary = Vocabulary()
    vocabulary.add(pad_string)
    vocabulary.add(unk_string)
    if is_label:
        use_pre_processing = False
    
    if sentences:
        sentences = [line_pre_process(sentence['sentence'], use_pre_processing)
                     for sentence in sentences]
    else:
        sentences = read_sentences(input_path, column_name, encoding,
                                   separator, use_pre_processing)
    populate_vocab(sentences, vocabulary)
    return vocabulary


def read_sentences(path, column, encoding, separator, use_pre_processing):
    with open(path, newline='', encoding=encoding) as sentences_file:
        reader = csv.DictReader(sentences_file, delimiter = separator)
        if use_pre_processing:
            for line in reader:
                yield pre_process(line[column], PREPROCESS_OPTION).split()
        else:
            for line in reader:
                yield line[column].split()