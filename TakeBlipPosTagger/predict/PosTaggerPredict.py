import torch
import pickle
import logging
from TakeBlipPosTagger import utils
from TakeBlipPosTagger import vocab
from TakeBlipPosTagger import data


class PosTaggerPredict:
    def __init__(self, model, label_path, embedding, save_dir=None,
                 encoding=None, separator=None):
        self.model = model
        self.label_path = label_path
        self.label_vocab = self.read_label_vocab()
        self.save_dir = save_dir
        self.encoding = encoding
        self.separator = separator
        self.fasttext = embedding

    def create_input_vocab(self, input_sentence):
        vocabulary = vocab.Vocabulary()
        vocabulary.add('<pad>')
        vocabulary.add("<unk>")
        vocab.populate_vocab(input_sentence, vocabulary)
        return vocabulary

    def read_label_vocab(self):
        with open(self.label_path, 'rb') as f:
            label_vocab = pickle.load(f)
        return label_vocab

    def load_embedding(self, input_vocab):
        self.model.embeddings[0].weight.data = torch.from_numpy(self.fasttext[input_vocab.i2f.values()])
        self.model.embeddings[0].weight.requires_grad = False

    def predict_line(self, line, use_pre_processing=True):
        self.model.train(False)
        if use_pre_processing:
            line = utils.pre_process(line)
        if len(line) == 0:
            return '', ''
        split_line = line.split()
        input_vocab = self.create_input_vocab([split_line])
        self.model.device = 'cpu'
        self.model.update_embedding(len(input_vocab))
        self.load_embedding(input_vocab)
        unlexicalized_line = utils.unlexicalize(split_line, input_vocab)
        with torch.no_grad():
            line_len = torch.LongTensor([len(split_line)])
            unlexicalized_line = torch.cat(
                [torch.LongTensor([unlexicalized_line]).unsqueeze(0)])
            predicted_line, _, _ = self.model.predict(unlexicalized_line,
                                                      line_len)
        preds = [' '.join(k) for k in
                 [utils.lexicalize(sequence, self.label_vocab)
                  for sequence in predicted_line.data.tolist()]]
        return line, preds[0]

    def predict_batch(self, filepath, sentence_column, pad_string, unk_string,
                      batch_size, shuffle, use_pre_processing, output_lstm,
                      sentences=None):
        self.model.train(False)

        input_vocab = vocab.create_vocabulary(
            input_path=filepath,
            column_name=sentence_column,
            pad_string=pad_string,
            unk_string=unk_string,
            encoding=self.encoding,
            separator=self.separator,
            use_pre_processing=use_pre_processing,
            sentences=sentences)

        logging.info('Loading dataset')

        if sentences:
            dataset = data.MultiSentWordBatchDataset(
                use_pre_processing=use_pre_processing,
                sentences=sentences)
            predictions = []
            use_index = True
        else:
            dataset = data.MultiSentWordDataset(
                path=filepath,
                label_column=None,
                encoding=self.encoding,
                separator=self.separator,
                use_pre_processing=use_pre_processing)
            utils.create_save_file(self.save_dir, output_lstm)
            use_index = False

        data_load = data.MultiSentWordDataLoader(
            dataset=dataset,
            vocabs=[input_vocab],
            pad_string=pad_string,
            unk_string=unk_string,
            batch_size=batch_size,
            shuffle=shuffle,
            tensor_lens=True,
            use_index=use_index)

        logging.info('Updating embedding')

        self.model.device = 'cpu'
        self.model.update_embedding(len(input_vocab))
        self.load_embedding(input_vocab)

        global_step = 0
        logging.info('Embedding updated')
        
        for i_idx, (batch, lens, index) in enumerate(data_load):
            batch_size = batch[0].size(0)
            global_step += batch_size
            sequence_batch = batch
            with torch.no_grad():
                sequence_batch_wrapped, index_wrapped, lens_s = utils.prepare_batch(sequence_batch, index, lens)
                preds, _, logits = self.model.predict(sequence_batch_wrapped, lens_s)

            preds = preds.data.tolist()
            sequence_batch = sequence_batch_wrapped.data.tolist()[0]

            preds_all = utils.lexicalize_sequence(preds, lens_s, self.label_vocab)

            sequence_batch_all = utils.lexicalize_sequence(sequence_batch,
                                                           lens_s, input_vocab)

            lstm_all = None

            if output_lstm:
                lstm_preds = logits.max(2)[1]
                lstm_preds_list = lstm_preds.data.tolist()
                lstm_all = utils.lexicalize_sequence(lstm_preds_list, lens_s,
                                                     self.label_vocab)

            if sentences:
                index_all = index_wrapped.data.tolist()
                predictions += [{'id': id,
                                 'processed_sentence': ' '.join(sentence),
                                 'tags': ' '.join(pred)}
                                for id, sentence, pred in zip(index_all,
                                                              sequence_batch_all,
                                                              preds_all)]
            else:
                utils.save_predict(self.save_dir, sequence_batch_all, preds_all, lstm_all)
            
            logging.info(f"iteration={global_step}")
       
        if sentences:
            return predictions
