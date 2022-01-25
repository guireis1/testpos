# Part Of Speech Tagging using BiLSTM-CRF on PyTorch #

An efficient BiLSTM-CRF implementation for solving Part Of Speech Tagging (POSTagging) problem.

This version of code works in:

* pytorch: versão
* python: versão

## Requirements ##

Install all required packages (other than pytorch) from `requirements.txt`

    pip install -r requirements.txt

## Training ##

Prepare data first. Data must be supplied in one csv file where the first column contain the sentences and the second one the respective labels for that sentence. File might be prepared as follows:

    (sample.csv)
	MessageProcessed,					Tags
    the fat rat sat on a mat,	det adj noun verb prep det noun
    the cat sat on a mat,		det noun verb prep det noun
    ...,						...
    
Then the above input is provided to `train.py` using `--input-path` and the column name for the sentences and the labels using `--sentence_column` and `--label_column`.

    python train.py --input-path files/input/sample.csv --sentence_column MessageProcessed --label_column Tags ...

You might need to setup several more parameters in order to make it work. 

A few parameters available on training are:

* `--batch-size`: number of sentences in each batch.
*  `--epochs`: number of epochs
* `--learning_rate`: learning rate parameter value

And parameters for validation and early stopping. 

## Our Training ##
For local execution run command:

	python train.py --input-path files/input/sample.csv --separator , --sentence_column MessageProcessed --label_column Tags --save-dir files/output/ --wordembed-path files/input/Titan_v2_titan_v2_after_correction_fasttext_window4_mincount20_cbow.kv --epochs 5

	python train.py --input-path files/input/sample.csv --separator , --sentence_column MessageProcessed --label_column Tags --save-dir files/output/ --wordembed-path files/input/Titan_v2_titan_v2_after_correction_fasttext_window4_mincount20_cbow.kv --epochs 5 --val --val-path files/input/sample_validation.csv --bidirectional --val-period 1e
    
    python train.py --input-path files/input/sample.csv --separator , --sentence_column MessageProcessed --label_column Tags --save-dir files/output/ --wordembed-path files/input/Titan_v2_titan_v2_after_correction_fasttext_window4_mincount20_cbow.kv --epochs 5 --val --val-path files/input/sample_validation.csv --bidirectional --val-period 10i --max-decay-num 2 --max-patience 2 --learning-rate-decay 0.1 --patience-threshold 0.98
 
    	
For running on Google Colab:

	!python train.py --input-path files/input/sample.csv --separator , --sentence_column MessageProcessed --label_column Tags --save-dir files/output/ --wordembed-path '/content/gdrive/Shared drives/Data & Analytics/D&A Research/TKS/Modelos/Embedding/FastText/kv/Titan_v2_titan_v2_after_correction_fasttext_window4_mincount20_cbow.kv' --epochs 5

	!python train.py --input-path files/input/sample.csv --separator , --sentence_column MessageProcessed --label_column Tags --save-dir files/output/ --wordembed-path '/content/gdrive/Shared drives/Data & Analytics/D&A Research/TKS/Modelos/Embedding/FastText/kv/Titan_v2_titan_v2_after_correction_fasttext_window4_mincount20_cbow.kv' --epochs 5 --val --val-path files/input/sample_validation.csv --bidirectional --val-period 1e
	
	!python train.py --input-path files/input/sample.csv --separator , --sentence_column MessageProcessed --label_column Tags --save-dir files/output/ --wordembed-path '/content/gdrive/Shared drives/Data & Analytics/D&A Research/TKS/Modelos/Embedding/FastText/kv/Titan_v2_titan_v2_after_correction_fasttext_window4_mincount20_cbow.kv' --epochs 5 --val --val-path files/input/sample_validation.csv --bidirectional --val-period 10i --max-decay-num 2 --max-patience 2 --learning-rate-decay 0.1 --patience-threshold 0.98

## Prediction ##
For local execution run command for one line predict:

	python predict.py --model-path files/output/model.pkl --input-sentence "eu quero prever essa frase" --label-vocab files/output/vocab-label.pkl --save-dir files/output/pred.csv --wordembed-path files/input/Titan_v2_titan_v2_after_correction_fasttext_window4_mincount20_cbow.kv

For local execution run command for batch predict:

	python predict.py --model-path files/output/model.pkl --input-path files/input/sample_predict.csv --sentence_column MessageProcessed --label-vocab files/output/vocab-label.pkl --save-dir files/output/pred.csv --wordembed-path files/input/Titan_v2_titan_v2_after_correction_fasttext_window4_mincount20_cbow.kv
	
	python predict.py --model-path files/output/model.pkl --input-path files/input/sample_predict.csv --sentence_column MessageProcessed --label-vocab files/output/vocab-label.pkl --save-dir files/output/pred.csv --wordembed-path files/input/Titan_v2_titan_v2_after_correction_fasttext_window4_mincount20_cbow.kv --use-lstm-output

For running on Google Colab for one line predict:

	!python predict.py --model-path files/output/model.pkl --input-sentence "eu quero prever essa frase" --label-vocab files/output/vocab-label.pkl --save-dir files/output/pred.csv --wordembed-path '/content/gdrive/Shared drives/Data & Analytics/D&A Research/TKS/Modelos/Embedding/FastText/kv/Titan_v2_titan_v2_after_correction_fasttext_window4_mincount20_cbow.kv'
	
	!python predict.py --model-path files/output/model.pkl --input-sentence "eu quero prever essa frase" --label-vocab files/output/vocab-label.pkl --save-dir files/output/pred.csv --wordembed-path '/content/gdrive/Shared drives/Data & Analytics/D&A Research/TKS/Modelos/Embedding/FastText/kv/Titan_v2_titan_v2_after_correction_fasttext_window4_mincount20_cbow.kv' --use-lstm-output

For running on Google Colab for batch predict:

	!python predict.py --model-path files/output/model.pkl --input-path files/input/sample_predict.csv --sentence_column MessageProcessed --label-vocab files/output/vocab-label.pkl --save-dir files/output/pred.csv --wordembed-path '/content/gdrive/Shared drives/Data & Analytics/D&A Research/TKS/Modelos/Embedding/FastText/kv/Titan_v2_titan_v2_after_correction_fasttext_window4_mincount20_cbow.kv'

Data must be supplied in one csv file with one column which contain the sentences. File might be prepared as follows:

    (sample.csv)
	MessageProcessed
    the fat rat sat on a mat
    the cat sat on a mat
    ...,		
