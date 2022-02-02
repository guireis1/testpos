import os
import argparse
import logging
import pickle
import torch
import yaap
import numpy as np

from azureml.core import Model
from azureml.core import Run

import TakeBlipPosTagger.vocab as vocab
import TakeBlipPosTagger.model as model
import TakeBlipPosTagger.utils as utils
from TakeBlipPosTagger.logger import AzuremlLogger
from TakeBlipPosTagger.train import LSTMCRFTrainer

parser = yaap.ArgParser(
    allow_config=True,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

group = parser.add_group('Basic Options')
group.add('--input-path', type=str, required=True,
          help='Path to input file that contains sequences of tokens '
               'separated by spaces.')
group.add('--separator', type=str, default='|',
          help='Input file column separator')
group.add('--encoding', type=str, default='utf-8',
          help='Input file encoding')
group.add('--sentence_column', type=str, required=True,
          help='String with the name of the column of sentences to be read from input file.')
group.add('--label_column', type=str, required=True,
          help='String with the name of the column of labels to be read from input file.')
group.add('--save-dir', type=str, required=True,
          help='Directory to save outputs (checkpoints, vocabs, etc.)')
group.add('--use_pre_processing', action='store_true', default=False,
          help='Whether to pre process input data.')

group = parser.add_group('Word Embedding Options')
group.add('--wordembed-path', type=str,
          help='Name of a pre-trained word embeddings.')
group.add('--embedding_model_name', default=None,
          help='Registered embedding model name in Azure ML.')
group.add('--embedding_model_version', type=int, default=None,
          help='Registered embedding model version in Azure ML.')

group = parser.add_group('Training Options')
group.add('--epochs', type=int, default=1,
          help='Number of training epochs.')
group.add('--dropout-prob', type=float, default=0.05,
          help='Probability in dropout layers.')
group.add('--batch-size', type=int, default=32,
          help='Mini-batch size.')
group.add('--shuffle', action='store_true', default=False,
          help='Whether to shuffle the dataset.')
group.add('--learning-rate', type=float, default=0.001,
          help='Learning rate parameter')
group.add('--learning-rate-decay', type=float, default=0.01,
          help='Learning rate decay')
group.add('--max-patience', type=int, default=5,
          help='Max patience')
group.add('--max-decay-num', type=int, default=5,
          help='Max decay')
group.add('--patience-threshold', type=float, default=0.98,
          help='Threshold of loss for patience count')
group.add('--ckpt-period', type=utils.PeriodChecker, default='1e',
          help='Period to wait until a model checkpoint is '
               'saved to the disk. '
               'Periods are specified by an integer and a unit ("e": '
               'epoch, "i": iteration, "s": global step).')

group = parser.add_group('Validation Options')
group.add('--val', action='store_true', default=False,
          help='Whether to perform validation.')
group.add('--val-path', type=str,
          help='Validation file path.')
group.add('--val-period', type=utils.PeriodChecker, default='100i',
          help='Period to wait until a validation is performed. '
               'Periods are specified by an integer and a unit ("e": '
               'epoch, "i": iteration, "s": global step).')
group.add('--samples', type=int, default=10,
          help='Number of output samples to display at each iteration.')

group = parser.add_group('Model Parameters')
group.add('--word-dim', type=int, default=300,
          help='Dimensions of word embeddings. Must be specified for each '
               'input. Defaults to 300 if none is specified.')
group.add('--lstm-dim', type=int, default=300,
          help='Dimensions of lstm cells. This determines the hidden '
               'state and cell state sizes.')
group.add('--lstm-layers', type=int, default=1,
          help='Layers of lstm cells.')
group.add('--bidirectional', action='store_true', default=False,
          help='Whether lstm cells are bidirectional.')
group.add('--alpha', type=float, default=0,
          help='L2 penalization parameter')

group = parser.add_group('Azure variable Parameters')
group.add('--train_ds_name', type=str, help='Train dataset variable name')
group.add('--val_ds_name', type=str, help='Validation dataset variable name')


def check_arguments(args):
    assert args.input_path is not None, 'At least one input file must be specified.'
    assert args.separator in {'|', ';', ','}, 'Specify a valid separator.'
    assert args.encoding in {'utf-8', 'utf8',
                             'latin-1'}, 'Specify a valid encoding.'
    if args.val:
        assert args.val_path is not None
    os.makedirs(args.save_dir, exist_ok=True)


def get_path_from_workspace(run, model_file_name,
                            model_name, model_version=None):
    ws = run.experiment.workspace
    model_path = Model(ws,
                       name=model_name).get_model_path(model_name=model_name,
                                                       version=model_version,
                                                       _workspace=ws)
    print('Workspace model path', model_path)
    return os.path.join(model_path, model_file_name)


def main(run, args):
    logging.basicConfig(level=logging.INFO)
    check_arguments(args)

    pad_string = '<pad>'
    unk_string = '<unk>'

    azml_logger = AzuremlLogger(run, torch.__version__)
    logging.info('AzuremlLogger active')

    input_path = run.input_datasets[args.train_ds_name]
    val_path = run.input_datasets[args.val_ds_name] if args.val else None
    wordembed_path = get_path_from_workspace(run=run,
                                             model_file_name=args.wordembed_path,
                                             model_name=args.embedding_model_name,
                                             model_version=args.embedding_model_version)
    logging.info('Creating vocabulary...')

    input_vocab = vocab.create_vocabulary(
        input_path=input_path,
        column_name=args.sentence_column,
        pad_string=pad_string,
        unk_string=unk_string,
        encoding=args.encoding,
        separator=args.separator,
        use_pre_processing=args.use_pre_processing)

    label_vocab = vocab.create_vocabulary(
        input_path=input_path,
        column_name=args.label_column,
        pad_string=pad_string,
        unk_string=unk_string,
        encoding=args.encoding,
        separator=args.separator,
        is_label=True)

    logging.info(
        f'Input: {input_path}\nValidation: {val_path}\nWE: {wordembed_path}')
    logging.info('Vocabulary and labels created')

    # Copying configuration file to save directory if config file is specified.
    if args.config:
        config_path = os.path.join(args.save_dir,
                                   os.path.basename(args.config))
        os.system(f'cp {args.config} {config_path}')

    if val_path:
        sentences = vocab.read_sentences(
            path=val_path,
            column=args.sentence_column,
            encoding=args.encoding,
            separator=args.separator,
            use_pre_processing=args.use_pre_processing)
        vocab.populate_vocab(sentences, input_vocab)

    vocab_path = os.path.join(args.save_dir, f'vocab-input.pkl')
    pickle.dump(input_vocab, open(vocab_path, 'wb'))
    vocab_label_path = os.path.join(args.save_dir, f'vocab-label.pkl')
    pickle.dump(label_vocab, open(vocab_label_path, 'wb'))

    logging.info('Initializing model...')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    crf = model.CRF(
        vocab_size=len(label_vocab),
        pad_idx=input_vocab.f2i[pad_string],
        unk_idx=input_vocab.f2i[unk_string],
        device=device).to(device)
    bilstmcrf_model = model.LSTMCRF(device=device,
                                   crf=crf,
                                   vocab_size=len(input_vocab),
                                   word_dim=args.word_dim,
                                   hidden_dim=args.lstm_dim,
                                   layers=args.lstm_layers,
                                   dropout_prob=args.dropout_prob,
                                   bidirectional=args.bidirectional,
                                   alpha=args.alpha).to(device)
    bilstmcrf_model.reset_parameters()
    params = sum(np.prod(p.size()) for p in bilstmcrf_model.parameters())
    logging.info(f'Number of parameters: {params}')

    logging.info('Loading word embeddings...')

    fasttext = utils.load_fasttext_embeddings(wordembed_path, pad_string)
    bilstmcrf_model.embeddings[0].weight.data = torch.from_numpy(
        fasttext[input_vocab.i2f.values()])
    bilstmcrf_model.embeddings[0].weight.requires_grad = False

    logging.info('Beginning training...')
    trainer = LSTMCRFTrainer(
        bilstmcrf_model=bilstmcrf_model,
        epochs=args.epochs,
        input_vocab=input_vocab,
        input_path=input_path,
        label_vocab=label_vocab,
        save_dir=args.save_dir,
        ckpt_period=args.ckpt_period,
        val=args.val,
        val_period=args.val_period,
        samples=args.samples,
        pad_string=pad_string,
        unk_string=unk_string,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        label_column=args.label_column,
        encoding=args.encoding,
        separator=args.separator,
        use_pre_processing=args.use_pre_processing,
        learning_rate=args.learning_rate,
        learning_rate_decay=args.learning_rate_decay,
        max_patience=args.max_patience,
        max_decay_num=args.max_decay_num,
        patience_threshold=args.patience_threshold,
        val_path=val_path,
        azureml_logger=azml_logger
    )
    trainer.train()
    logging.info('Done!')


if __name__ == '__main__':
    run = Run.get_context()
    args = parser.parse_args()
    main(run, args)
    run.complete()
