# TakeBlipPosTagging Package
_Data & Analytics Research_

## Overview
The Part Of Speech Tagging (POSTagging) is the process of labeling a word in a text (corpus) with it's corresponding particular part of speech.
This implementation uses BiLSTM-CRF for solving the POSTagging task utilizing PyTorch framework for training a supervised model and predicting in CPU. 
For training it receives a pre-trained FastText Gensim embedding and a .csv file. It outputs three pickle files: model, word vocabulary and label vocabulary. 

Here are presented these content:

* [Installation](#installation)
* [File Format](#fileformat)
* [Configure](#configure)
* [Run](#run)


## Installation

This version works in:

* PyTorch: 1.7.1
* Python: 3.6

## Requirements ##

Install all required packages (other than pytorch) from `requirements.txt`

    pip install -r requirements.txt

## Training ##

Prepare data first. Data must be supplied in one csv file where the first column contain the sentences and the second one the respective labels for that sentence. File might be prepared as follows:

    (sample.csv)
	MessageProcessed,		                Tags
    quero o meu boleto,	                        VERB ART PRON SUBS
    não consegui contato por telefone,		ADV VERB SUBS PREP SUBS
    ...,						...
    
Then the above input is provided to `train.py` using `--input-path` and the column name for the sentences and the labels using `--sentence_column` and `--label_column`.

    python train.py --input-path *.csv --sentence_column MessageProcessed --label_column Tags ...

You might need to setup several more parameters in order to make it work. 

A few parameters available on training are:

* `--batch-size`: number of sentences in each batch.
*  `--epochs`: number of epochs
* `--learning_rate`: learning rate parameter value

And parameters for validation and early stopping. 

## Our Training ##
For local execution run command:

	python train.py --input-path *.csv --separator , --sentence_column MessageProcessed --label_column Tags --save-dir * --wordembed-path *.kv --epochs 5

	python train.py --input-path *.csv --separator , --sentence_column MessageProcessed --label_column Tags --save-dir * --wordembed-path f*.kv --epochs 5 --val --val-path *.csv --bidirectional --val-period 1e
    
    python train.py --input-path *.csv --separator , --sentence_column MessageProcessed --label_column Tags --save-dir * --wordembed-path *.kv --epochs 5 --val --val-path *.csv --bidirectional --val-period 10i --max-decay-num 2 --max-patience 2 --learning-rate-decay 0.1 --patience-threshold 0.98
 


## Prediction ##
For local execution run command for one line predict:

	python predict.py --model-path *.pkl --input-sentence "eu quero prever essa frase" --label-vocab *.pkl --save-dir *.csv --wordembed-path *.kv

For local execution run command for batch predict:

	python predict.py --model-path *.pkl --input-path *.csv --sentence_column MessageProcessed --label-vocab *.pkl --save-dir *.csv --wordembed-path *.kv
	
	python predict.py --model-path *.pkl --input-path *.csv --sentence_column MessageProcessed --label-vocab *.pkl --save-dir *.csv --wordembed-path *.kv --use-lstm-output

Data must be supplied in one csv file with one column which contain the sentences. File might be prepared as follows:

    (sample.csv)
	MessageProcessed
    quero o meu boleto
    não consegui contato por telefone
    ...,	

