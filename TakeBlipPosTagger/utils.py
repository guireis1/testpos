from gensim.models import KeyedVectors
import torch
import torch.autograd as autograd
import re
import csv

from take_text_preprocess.presentation import pre_process as pp

from .process_option import PREPROCESS_OPTION


def load_fasttext_embeddings(path, pad):
    fasttext = KeyedVectors.load(path, mmap=None)
    fasttext.add(pad, [0] * fasttext.vector_size, replace=True)
    return fasttext


class PeriodChecker(object):
    PATTERN = re.compile(r"([0-9]+)(e|i|)")

    def __init__(self, s: str):
        m = self.PATTERN.match(s)
        assert m is not None, \
            f"String must be in the format of [integer]['e'/'i'/'s']."

        self.scalar = int(m.group(1))
        self.unit = m.group(2)

    def __call__(self, epochs=None, iters=None, steps=None):
        assert epochs or iters or steps, \
            "One of the items must be provided."

        if self.unit == "e":
            if epochs is not None:
                return epochs % self.scalar == 0
        elif self.unit == "i":
            if iters is not None:
                return iters % self.scalar == 0
        elif self.unit == "s":
            if steps is not None:
                return steps % self.scalar == 0
        else:
            raise ValueError(f"Unrecognized unit: {self.unit}")

        return False


def prepare_batch(sentences, labels, lens):  ###igual ao de baselstmcrf
    lens, idx = torch.sort(lens, 0, True)
    if labels is not None:
        sentences, labels = sentences[:, idx], labels[idx]
        labels = autograd.Variable(labels)
    else:
        sentences = sentences[:, idx]

    sentences = autograd.Variable(sentences)
    lens = autograd.Variable(lens)
    return sentences, labels, lens


def tighten(sequences, lens):
    return [sequence[:seq_len] for sequence, seq_len in zip(sequences, lens)]


def lexicalize(sequence, vocab):
    return [vocab.i2f[word] if word in vocab else "<unk>" for word in sequence]


def unlexicalize(sentence, vocab):
    return [vocab.f2i[word] if word in vocab else vocab.f2i["<unk>"] for word in sentence]


def lexicalize_sequence(sequences, lens, vocab):
    return [lexicalize(sequence, vocab) for sequence in tighten(sequences, lens)]


def pre_process(sentence):
    return pp(sentence, PREPROCESS_OPTION)


def create_save_file(path, use_lstm_pred=False):
    with open(path, 'w') as file:
        writer = csv.writer(file, delimiter='|', lineterminator='\n')
        writer_parameter = ['Sentence', 'Tags', 'LSTM'] if use_lstm_pred else ['Sentence', 'Tags']
        writer.writerow(writer_parameter)


def save_predict(path, sequence, pred, lstm=None):
    with open(path, 'a') as file:
        writer = csv.writer(file, delimiter='|', lineterminator='\n')
        if lstm is not None:
            writer_parameter = zip([' '.join(k) for k in sequence], [' '.join(k) for k in pred],
                                   [' '.join(k) for k in lstm])
        else:
            writer_parameter = zip([' '.join(k) for k in sequence], [' '.join(k) for k in pred])
        for row in writer_parameter:
            writer.writerow(row)
