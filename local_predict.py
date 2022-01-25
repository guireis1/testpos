import os
import sys
import torch
import pickle
import logging
import yaap
import argparse
import TakeBlipPosTagger.utils as utils
import TakeBlipPosTagger.data as data
import TakeBlipPosTagger.vocab as vocab
from TakeBlipPosTagger.predict import PosTaggerPredict


parser = yaap.ArgParser(
    allow_config=True,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

group = parser.add_group("Basic Options")
group.add("--model-path", type=str, required=True,
          help="Path to trained model.")

group.add("--input-sentence", type=str, required=False,
          help="String containing sentence to be predicted")

group.add("--input-path", type=str, required=False,
          help="Path to the input file containing the sentences to be predicted")

group.add("--sentence_column", type=str, required=False,
          help="String with the name of the column of sentences to be read from input file.")

group.add("--label-vocab", type=str, required=True,
          help="Path to input file that contains the label vocab")

group.add("--save-dir", type=str, default='./pred.csv',
          help="Directory to save predict")

group.add("--wordembed-path", type=str, required=True,
          help="Path to pre-trained word embeddings. ")

group.add("--separator", type=str, default='|',
          help="Input file column separator")

group.add("--encoding", type=str, default='utf-8',
          help="Input file encoding")

group.add("--shuffle", action="store_true", default=False,
          help="Whether to shuffle the dataset.")

group.add("--batch-size", type=int, default=32,
          help="Mini-batch size.")

group.add("--use-pre-processing", action="store_true", default=False,
          help="Whether to pre process input data.")

group.add("--use-lstm-output", action='store_true', default=False,
          help='Whether to give the output for BiLSTM')


def check_arguments(args):
    assert args.input_path is not None or args.input_sentence is not None, "At least one input file must be specified."

    if args.input_path is not None:
        assert args.sentence_column is not None, "When reading from file the column to be read must be specified."


def main(args):

    check_arguments(args)

    sys.path.insert(0, os.path.dirname(args.model_path))
    logging.basicConfig(level=logging.INFO)
    logging.info("Loading Model...")
    bilstmcrf = torch.load(args.model_path)

    embedding = utils.load_fasttext_embeddings(args.wordembed_path, '<pad>')

    postagger_predicter = PosTaggerPredict(
        model=bilstmcrf,
        label_path=args.label_vocab,
        embedding=embedding,
        save_dir=args.save_dir,
        encoding=args.encoding,
        separator=args.separator)

    if args.input_sentence is not None:
        processed_sentence, tags = postagger_predicter.predict_line(args.input_sentence)
        logging.info('Sentence: "{}" Predicted: {}'.format(processed_sentence, tags))
        return [processed_sentence, tags]
    else:
        pad_string = '<pad>'
        unk_string = '<unk>'
        postagger_predicter.predict_batch(
            filepath=args.input_path,
            sentence_column=args.sentence_column,
            pad_string=pad_string,
            unk_string=unk_string,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            use_pre_processing=args.use_pre_processing,
            output_lstm=args.use_lstm_output)
        return None

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
