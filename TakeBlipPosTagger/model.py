import torch
import torch.nn as nn
from torch.autograd import Variable

def log_sum_exp(vec, dim=0):
    max, idx = torch.max(vec, dim)
    max_exp = max.unsqueeze(-1).expand_as(vec)
    return max + torch.log(torch.sum(torch.exp(vec - max_exp), dim))

class CRF(nn.Module):
    def __init__(self, vocab_size, pad_idx, unk_idx, device):
        super(CRF, self).__init__()
        self.device = device
        self.vocab_size = vocab_size
        self.n_labels = n_labels = vocab_size + 2
        self.start_idx = n_labels - 2
        self.stop_idx = n_labels - 1
        self.transitions = nn.Parameter(torch.randn(n_labels, n_labels).to(self.device))
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx

    def reset_parameters(self):
        nn.init.constant_(self.transitions.data, 0)
        nn.init.constant_(self.transitions.data[:, self.unk_idx], -3)
        nn.init.constant_(self.transitions.data[:, self.pad_idx], -3)
        nn.init.constant_(self.transitions.data[:, self.start_idx], -3)
        nn.init.constant_(self.transitions.data[:, self.stop_idx], -3)

    def forward(self, logits, lens):
        '''
        Arguments:
            logits: [batch_size, seq_len, n_labels] FloatTensor
            lens: [batch_size] LongTensor
        '''
        batch_size, seq_len, n_labels = logits.size()
        alpha = logits.data.new(batch_size, self.n_labels).fill_(-10000)
        alpha[:, self.start_idx] = 0
        alpha = Variable(alpha)
        cloned_lens = lens.clone()

        logits_transposed = logits.transpose(1, 0)
        for logit in logits_transposed:
            logit_expanded = logit.unsqueeze(-1).expand(batch_size,
                                                        *self.transitions.size())
            alpha_expanded = alpha.unsqueeze(1).expand(batch_size,
                                                       *self.transitions.size())
            transition_expanded = self.transitions.unsqueeze(0).expand_as(alpha_expanded)
            matrix = transition_expanded + alpha_expanded + logit_expanded
            alpha_next = log_sum_exp(matrix, 2).squeeze(-1)

            mask = (cloned_lens > 0).float().unsqueeze(-1).expand_as(alpha)
            alpha = mask * alpha_next + (1 - mask) * alpha
            cloned_lens = cloned_lens - 1

        alpha = alpha + self.transitions[self.stop_idx].unsqueeze(0).expand_as(alpha)
        norm = log_sum_exp(alpha, 1).squeeze(-1)

        return norm

    def viterbi_decode(self, logits, lens):
        '''Borrowed from pytorch tutorial

        Arguments:
            logits: [batch_size, seq_len, n_labels] FloatTensor
            lens: [batch_size] LongTensor
        '''
        batch_size, seq_len, n_labels = logits.size()
        viterbi = logits.data.new(batch_size, self.n_labels).fill_(-10000)
        viterbi[:, self.start_idx] = 0
        viterbi = Variable(viterbi)
        cloned_lens = lens.clone()

        logits_transposed = logits.transpose(1, 0)
        pointers = []
        for logit in logits_transposed:
            viterbi_expanded = viterbi.unsqueeze(1).expand(batch_size, n_labels, n_labels)
            transition_expanded = self.transitions.unsqueeze(0).expand_as(viterbi_expanded)
            viterbi_transition_sum = viterbi_expanded + transition_expanded
            viterbi_max, viterbi_argmax = viterbi_transition_sum.max(2)

            viterbi_max = viterbi_max.squeeze(-1)
            viterbi_next = viterbi_max + logit
            pointers.append(viterbi_argmax.squeeze(-1).unsqueeze(0))

            mask = (cloned_lens > 0).float().unsqueeze(-1).expand_as(viterbi_next)
            viterbi = mask * viterbi_next + (1 - mask) * viterbi

            mask = (cloned_lens == 1).float().unsqueeze(-1).expand_as(viterbi_next)
            viterbi += mask * self.transitions[self.stop_idx].unsqueeze(0).expand_as(viterbi_next)

            cloned_lens = cloned_lens - 1

        pointers = torch.cat(pointers)
        scores, idx = viterbi.max(1)
        paths = [idx.unsqueeze(1)]

        for argmax in reversed(pointers):
            idx_exp = idx.unsqueeze(-1)
            idx = torch.gather(argmax, 1, idx_exp)
            idx = idx.squeeze(-1)

            paths.insert(0, idx.unsqueeze(1))

        paths = torch.cat(paths[1:], 1)
        scores = scores.squeeze(-1)

        return scores, paths

    def transition_score(self, labels, lens):
        '''
        Arguments:
             labels: [batch_size, seq_len] LongTensor
             lens: [batch_size] LongTensor
        '''
        batch_size, seq_len = labels.size()

        # pad labels with <start> and <stop> indices
        labels_ext = Variable(labels.data.new(batch_size, seq_len + 2))
        labels_ext[:, 0] = self.start_idx
        labels_ext[:, 1:-1] = labels
        mask = sequence_mask(lens + 1, self.device, max_len=seq_len + 2).long()
        pad_stop = Variable(labels.data.new(1).fill_(self.stop_idx))
        pad_stop = pad_stop.unsqueeze(-1).expand(batch_size, seq_len + 2)
        labels_ext = (1 - mask) * pad_stop + mask * labels_ext
        labels = labels_ext

        transitions = self.transitions

        # obtain transition vector for each label in batch and timestep
        # (except the last ones)
        transitions_expanded = transitions.unsqueeze(0).expand(batch_size, *transitions.size())
        labels_except_last = labels[:, 1:]
        labels_except_last_expanded = labels_except_last.unsqueeze(-1)\
            .expand(*labels_except_last.size(), transitions.size(0))
        transitions_row = torch.gather(transitions_expanded, 1, labels_except_last_expanded)

        # obtain transition score from the transition vector for each label
        # in batch and timestep (except the first ones)
        labels_except_first_expanded = labels[:, :-1].unsqueeze(-1)
        transitions_score = torch.gather(transitions_row, 2, labels_except_first_expanded)
        transitions_score = transitions_score.squeeze(-1)

        mask = sequence_mask(lens + 1, self.device).float()
        transitions_score = transitions_score * mask
        score = transitions_score.sum(1).squeeze(-1)

        return score


class LSTMCRF(nn.Module):
    def __init__(self, crf, vocab_size, word_dim, hidden_dim, layers,
                 dropout_prob, device, alpha=0, bidirectional=False):
        super(LSTMCRF, self).__init__()

        self.device = device
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.lstm_layers = layers
        self.dropout_prob = dropout_prob
        self.alpha = alpha

        self.crf = crf
        self.bidirectional = bidirectional
        self.n_labels = n_labels = self.crf.n_labels
        self.embeddings = nn.ModuleList([nn.Embedding(vocab_size, word_dim).to(self.device)]).to(self.device)

        self.output_hidden_dim = self.hidden_dim
        if bidirectional:
            self.output_hidden_dim *= 2

        self.tanh = nn.Tanh()
        self.input_layer = nn.Linear(self.word_dim, hidden_dim)
        self.output_layer = nn.Linear(self.output_hidden_dim, n_labels)
        self.lstm = nn.LSTM(input_size=hidden_dim,
                            hidden_size=hidden_dim,
                            num_layers=layers,
                            bidirectional=bidirectional,
                            dropout=dropout_prob,
                            batch_first=True)

    def reset_parameters(self):
        for emb in self.embeddings:
            nn.init.xavier_normal_(emb.weight.data)

        nn.init.xavier_normal_(self.input_layer.weight.data)
        nn.init.xavier_normal_(self.output_layer.weight.data)
        self.crf.reset_parameters()
        self.lstm.reset_parameters()

    def update_embedding(self, vocab_size):
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size, self.word_dim).to(self.device)]
        ).to(self.device)

    def _run_rnn_packed(self, cell, sequence_batch, sequence_batch_lens, h=None):
        sequence_batch_packed = nn.utils.rnn.pack_padded_sequence(sequence_batch, sequence_batch_lens.data.tolist(),
                                                  batch_first=True)

        if h is not None:
            output, h = cell(sequence_batch_packed, h)
        else:
            output, h = cell(sequence_batch_packed)

        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        return output, h

    def _embeddings(self, sequences_batch):
        embedded_sequence_combination = self.embeddings[0].to(self.device)(sequences_batch[0].to(self.device))
        return embedded_sequence_combination

    def _forward_bilstm(self, sequences_batch, lens):
        n_feats, batch_size, seq_len = sequences_batch.size()

        embedded_sequences_batch = self._embeddings(sequences_batch)
        embedded_sequences_batch = embedded_sequences_batch.view(-1, self.word_dim)
        embedded_sequences_batch = self.tanh(self.input_layer(embedded_sequences_batch))
        embedded_sequences_batch = embedded_sequences_batch.view(batch_size, seq_len, self.hidden_dim)

        output, h = self._run_rnn_packed(self.lstm, embedded_sequences_batch, lens)

        output = output.contiguous()
        output = output.view(-1, self.output_hidden_dim)
        output = self.tanh(self.output_layer(output))
        output = output.view(batch_size, seq_len, self.n_labels)

        return output

    def _bilstm_score(self, logits, sequence_labels, lens):
        sequence_labels_expanded = sequence_labels.unsqueeze(-1)
        scores = torch.gather(logits, 2, sequence_labels_expanded).squeeze(-1)
        mask = sequence_mask(lens, self.device).float()
        scores = scores * mask
        score = scores.sum(1).squeeze(-1)
        return score

    def score(self, sequences, sequence_labels, lens, logits=None):
        if logits is None:
            logits = self._forward_bilstm(sequences, lens)

        transition_score = self.crf.transition_score(sequence_labels, lens)
        bilstm_score = self._bilstm_score(logits, sequence_labels, lens)

        score = transition_score + bilstm_score

        return score

    def predict(self, sequences, lens):
        logits = self._forward_bilstm(sequences, lens)
        scores, preds = self.crf.viterbi_decode(logits, lens)

        return preds, scores, logits

    def loglik(self, sequences, sequence_labels, lens, return_logits=False):
        logits = self._forward_bilstm(sequences, lens)
        norm_score = self.crf(logits, lens)
        sequence_score = self.score(sequences, sequence_labels, lens, logits=logits)
        loglik = sequence_score - norm_score - self.alpha*self.crf.transitions[2:-2, 2:-2].pow(2).sum()

        if return_logits:
            return loglik, logits
        else:
            return loglik

def sequence_mask(lens, device, max_len=None):
    batch_size = lens.size(0)

    if max_len is None:
        max_len = lens.max().data

    ranges = torch.arange(0, max_len).long().to(device)
    ranges = ranges.unsqueeze(0).expand(batch_size, max_len)
    ranges = Variable(ranges)

    lens_expanded = lens.unsqueeze(1).expand_as(ranges)
    mask = ranges < lens_expanded

    return mask