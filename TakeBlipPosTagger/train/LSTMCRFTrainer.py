import os
import logging

import mlflow
import numpy as np
import torch
import torch.optim as optim

import TakeBlipPosTagger.model as model
import TakeBlipPosTagger.utils as utils
import TakeBlipPosTagger.data as data
import TakeBlipPosTagger.logger as logger

class LSTMCRFTrainer(object):
    def __init__(self, bilstmcrf_model: model.LSTMCRF, input_vocab, input_path,
                 label_vocab, save_dir, ckpt_period, val, val_period, samples,
                 pad_string, unk_string, batch_size, shuffle, label_column,
                 encoding, separator, use_pre_processing, learning_rate,
                 learning_rate_decay, max_patience, max_decay_num,
                 patience_threshold, run_id, val_path=None, epochs=5,
                 optimizer=optim.Adam, n_iter=5, azureml_logger=None):
        self.model = bilstmcrf_model
        self.epochs = epochs
        self.optimizer_cls = optimizer
        self.optimizer = optimizer(self.model.parameters(), learning_rate)
        self.save_dir = save_dir
        self.input_vocab = input_vocab
        self.label_vocab = label_vocab
        self.input_path = input_path
        self.val_path = val_path
        self.val = val
        self.writer = None
        self.samples = samples
        self.encoding = encoding
        self.label_column = label_column
        self.separator = separator
        self.use_pre_processing = use_pre_processing
        self.pad_string = pad_string
        self.unk_string = unk_string
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_iter = n_iter
        self.min_dev_loss = float('inf')
        self.patience = 0
        self.patience_threshold = patience_threshold
        self.max_patience = max_patience
        self.decay_num = 0
        self.max_decay_num = max_decay_num
        self.learning_rate_decay = learning_rate_decay
        self.logic_break = False
        self.azureml_logger = azureml_logger
        # self.azureml_modelsave = azureml_modelsave
        self.validation_number = 0
        self.run_id = run_id
        
        self.repeatables = {}

        if self.val:
            self.repeatables[val_period] = self.validate

    @staticmethod
    def get_longest_word(vocab):
        lens = [(len(w), w) for w in vocab.f2i]
        return max(lens, key=lambda x: x[0])[1]

    @staticmethod
    def display_row(items, widths):
        assert len(items) == len(widths)

        padded = ['{{:>{}s}}'.format(w).format(item)
                  for item, w in zip(items, widths)]
        logging.info(' '.join(padded))

    def load_data(self, train):
        path = self.input_path if train else self.val_path

        dataset = data.MultiSentWordDataset(
            path=path,
            label_column=self.label_column,
            encoding=self.encoding,
            separator=self.separator,
            use_pre_processing=self.use_pre_processing)

        dataloader = data.MultiSentWordDataLoader(
            dataset=dataset,
            vocabs=[self.input_vocab]+[self.label_vocab],
            pad_string=self.pad_string,
            unk_string=self.unk_string,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            tensor_lens=True)

        return dataloader

    def display_samples(self, inputs, targets, lstms, crfs):
        inputs = [[utils.lexicalize(s, self.input_vocab) for s in input_item]
                  for input_item in inputs]
        targets = [utils.lexicalize(s, self.label_vocab) for s in targets]
        lstms = [utils.lexicalize(s, self.label_vocab) for s in lstms]
        crfs = [utils.lexicalize(s, self.label_vocab) for s in crfs]
        transposed = list(zip(*(inputs + [lstms, crfs, targets])))
        col_names = [f'INPUT{i + 1:02d}' for i in range(len(inputs))] + \
                    ['LSTM', 'CRF', 'TARGETS']
        vocabs = [self.input_vocab] + [self.label_vocab] * 3
        col_widths = [max(len(self.get_longest_word(v)), len(c))
                      for v, c in zip(vocabs, col_names)]

        for i, sample in enumerate(transposed):
            rows = list(zip(*sample))

            logging.info('')
            logging.info(f'SAMPLE #{i + 1}')
            self.display_row(col_names, col_widths)
            for row in rows:
                self.display_row(row, col_widths)

    @staticmethod
    def random_idx(max_count, subset=None):
        idx = np.random.permutation(np.arange(max_count))

        if subset is not None:
            return idx[:subset]

        return idx

    @staticmethod
    def gather(lst, idx):
        return [lst[i] for i in idx]

    def save_model(self, steps=None):
        save_path = os.path.join(self.save_dir, 'model.pkl')
        torch.save(self.model, save_path)
        # if self.azureml_logger and self.azureml_modelsave:
        #     try:
        #         self.azureml_modelsave(self.save_dir, self.azureml_logger)
        #     except Exception as e:
        #         logging.error('Error saving model', e)

    def on_epoch_complete(self, epoch_idx, global_iter, global_step):
        print("repeatables")
        print(self.repeatables)
        print(type(self.repeatables))
        print(self.repeatables.items())
        for period_checker, func in self.repeatables.items():
            if period_checker(epochs=epoch_idx,
                              iters=global_iter,
                              steps=global_step):
                func(steps=global_step)
                self.model.train(True)

    def train(self):
        self.model.train(True)
        global_step = 0
        global_iter = 0

        for epoch_idx in range(self.epochs):
            epoch_idx += 1
            train_data = self.load_data(train=True)
            for iteration_idx, (batch, lens, _) in enumerate(train_data):
                batch_size = batch[0].size(0)
                iteration_idx += 1
                global_step += batch_size
                global_iter += 1

                self.model.zero_grad()

                batch_sentences, batch_labels = batch[:-1], batch[-1]
                batch_sentences, batch_labels_var, lens_s = utils.prepare_batch(batch_sentences, batch_labels, lens)
                loglik = self.model.loglik(batch_sentences, batch_labels_var, lens_s)
                negative_loglik = -loglik.mean()
                print("negative_loglik", negative_loglik.item())
                mlflow.log_metric("negative_loglik", negative_loglik.item())
                #self.azureml_logger.save_negative_loglik(negative_loglik)
                negative_loglik.backward()
                self.optimizer.step()

                if global_step % (self.batch_size * self.n_iter) == 0:
                    # logging.info(f'iteration = {global_iter}  negative_loglik={negative_loglik:.4f}')
                    if self.logic_break:
                        # logging.info('Early stopping')
                        # logging.info('Saving model ...')
                        # self.save_model()
                        # mlflow.log_metric("Final iteration", global_iter)
                        #mlflow.log_metric("Final negative_loglik", negative_loglik.item())   
                        break
            train_data.dataset.close_file()
            self.on_epoch_complete(epoch_idx, global_iter, global_step)
            if self.logic_break:
                # logging.info('Early stopping')
                # logging.info('Saving model ...')
                # self.save_model()
                # mlflow.log_metric("Final iteration", global_iter)
                #mlflow.log_metric("Final negative_loglik", negative_loglik.item())   
                break
        # logging.info(f'iteration ={global_iter}')
        # logging.info(f'negative_loglik={negative_loglik:.4f}')
        # logging.info('Saving model ...')
        # self.save_model()
        mlflow.log_metric("Final iteration", global_iter)
        return self.model

    def check_loss(self, loss):
        if loss < self.min_dev_loss * self.patience_threshold:
            self.min_dev_loss = loss
            self.patience = 0
        else:
            self.patience += 1
            if self.patience == self.max_patience:
                self.decay_num += 1
                if self.decay_num == self.max_decay_num:
                    self.logic_break = True
                learning_rate = self.optimizer.param_groups[0]['lr'] * self.learning_rate_decay
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = learning_rate
                self.patience = 0

    def validate(self, steps=None):
        if not self.val:
            return

        path = os.path.join(self.save_dir, 'predict_validation.csv')
        utils.create_save_file(path)

        logging.info('...............................')
        logging.info('Validating...')
        self.model.train(False)
        data_size = 0
        sampled = False

        val_data = self.load_data(train=False)

        nb_classes = len(self.label_vocab)

        confusion_matrix = torch.zeros(nb_classes, nb_classes)

        loss = 0

        for i_idx, (batch, lens, _) in enumerate(val_data):
            i_idx += 1
            batch_size = batch.size(1)
            data_size += batch_size

            sequence_batch, label_batch = batch[:-1], batch[-1]
            with torch.no_grad():
                sequence_batch_wrapped, label_batch_wrapped, lens_s = utils.prepare_batch(sequence_batch, label_batch, lens)
                loglik, logits = self.model.loglik(sequence_batch_wrapped, label_batch_wrapped, lens_s, return_logits=True)
                preds, _, _ = self.model.predict(sequence_batch_wrapped, lens_s)

            loss += -loglik.sum()
            pred_list = preds.data.tolist()
            targets = label_batch_wrapped.data.tolist()
            lens_s = lens_s.data.tolist()
            sequence_batch = sequence_batch_wrapped.data.tolist()[0]

            preds_all = utils.lexicalize_sequence(pred_list, lens_s, self.label_vocab)
            sequence_batch_all = utils.lexicalize_sequence(sequence_batch, lens_s, self.input_vocab)

            utils.save_predict(path, sequence_batch_all, preds_all)

            for t, p in zip(label_batch_wrapped.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

            if not sampled and self.samples > 0:
                sample_idx = self.random_idx(batch_size, self.samples)
                sequence_batch = sequence_batch_wrapped.data.tolist()
                sequence_batch = [utils.tighten(x, lens_s) for x in sequence_batch]
                lstm = logits.max(2)[1]
                lstm = lstm.data.tolist()
                lstm = utils.tighten(lstm, lens_s)

                sequence_batch_smp = [self.gather(x, sample_idx) for x in sequence_batch]
                label_batch_smp = self.gather(targets, sample_idx)
                crf_smp = self.gather(pred_list, sample_idx)
                lstm_smp = self.gather(lstm, sample_idx)

                self.display_samples(sequence_batch_smp, label_batch_smp, lstm_smp, crf_smp)
                del sequence_batch, lstm, sequence_batch_smp, label_batch_smp, crf_smp, lstm_smp

                sampled = True

        val_data.dataset.close_file()
        loss = loss/data_size

        self.check_loss(loss)

        confusion_matrix = confusion_matrix[2:, 2:]
        print(confusion_matrix)
        # logging.info(f'precision: {confusion_matrix.diag() / confusion_matrix.sum(dim=0)}')
        # logging.info(f'recall: {confusion_matrix.diag() / confusion_matrix.sum(dim=1)}')
        # logging.info(f'negative_loglik validate {loss:.4f}')
        mlflow.log_metric("negative_loglik validate", loss.item())
        
        if self.azureml_logger: 
            self.azureml_logger.save_confusion_matrix_from_tensor(
                confusion_matrix=confusion_matrix,
                labels=list(self.label_vocab.f2i.keys())[2:],
                current_epoch=self.validation_number
            )
            self.azureml_logger.save_metrics(
                confusion_matrix=confusion_matrix,
                labels=list(self.label_vocab.f2i.keys())[2:]
            )
            #self.azureml_logger.save_validation_metrics(loss)
        else:
          logger.MLflowLogger.save_confusion_matrix_from_tensor(
                  confusion_matrix=confusion_matrix,
                  labels=list(self.label_vocab.f2i.keys())[2:],
                  current_epoch=self.validation_number,
                  run_id=self.run_id,
                  save_dir=self.save_dir
              )
        self.validation_number += 1

