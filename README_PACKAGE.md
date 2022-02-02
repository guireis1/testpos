# TakeBlipPosTagging Package
_Data & Analytics Research_

## Overview
The Part Of Speech Tagging (POSTagging) is the process of labeling a word in a text (corpus) with it's corresponding particular part of speech.
This implementation uses BiLSTM-CRF for solving the POSTagging task utilizing PyTorch framework for training a supervised model and predicting in CPU. 
For training, it receives a pre-trained FastText Gensim embedding and a .csv file. It outputs three pickle files: model, word vocabulary and label vocabulary. 

Here are presented these content:

## Training PosTagging Model
To train you own PosTagging model using this package, some steps should 
be made before:
1) Import main packages
2) Initialize file variables: embedding, train and validation .csv files;
3) Initialize PosTagging parameters;
4) Instantiate vocabulary and label vocabulary objects;
5) Save vocabulary and label models;
6) Initialize BiLSTM-CRF model and set embedding;
7) Initialize LSTMCRFTrainer object;
8) Train the model.

An example of the above steps could be found in the python code below:

1) Import main packages:
 ```
   import os
   import torch
   import pickle
    
   import TakeBlipPosTagger.utils as utils
   import TakeBlipPosTagger.vocab as vocab
   import TakeBlipPosTagger.model as model
   from TakeBlipPosTagger.train import LSTMCRFTrainer
 ```
2) Initialize file variables:
```
wordembed_path = '*.kv'
save_dir = '*'
input_path = '*.csv'
val_path = '*csv'
```
3) Initialize PosTagging parameters

In order to train a model, the following variables should be created:

- **sentence_column**: String with sentence column name in train file
- **unknown_string**: String which represents unknown token;
- **padding_string**: String which represents the pad token;
- **batch_size**: Number of sentences Number of samples that will be propagated through the network;
- **shuffle**: Boolean representing whether the dataset is shuffled.
- **use_pre_processing**: Boolean indicating whether the sentence will be preprocessed
- **separator**: String with file separator (for batch prediction);
- **encoding**: String with the encoding used in sentence;
- **save_dir**: String with directory to save outputs (checkpoints, vocabs, etc.);
- **device**: String where train will occur (cpu or gpu);
- **word_dim**: Integer with dimensions of word embeddings;
- **lstm_dim**: Integer with dimensions of lstm cells. This determines the hidden state and cell state sizes;
- **lstm_layers**: Integer with layers of lstm cells;
- **dropout_prob**: Float with probability in dropout layers;
- **bidirectional**: Boolean whether lstm cells are bidirectional;
- **alpha**: Float representing L2 penalization parameter;
- **epochs**: Integer with number of training epochs;
- **ckpt_period**: Period to wait until a model checkpoint is saved to the disk. Periods are specified by an integer and a unit ("e": 'epoch, "i": iteration, "s": global step);
- **val**: Integer whether to perform validation;
- **val_period**: Period to wait until a validation is performed. Periods are specified by an integer and a unit ("e": epoch, "i": iteration, "s": global step);
- **samples**: Integer with number of output samples to display at each validation;
- **learning_rate**: Float representing learning rate parameter;
- **learning_rate_decay**: Float representing learning rate decay parameter;
- **max_patience**: Integer with max patience parameter;
- **max_decay_num**: Integer with max decay parameter;
- **patience_threshold**: Float representing threshold of loss for patience count.


Example of parameters creation:
```
sentence_column = 'Message'
label_column = 'Tags'
unknown_string = '<unk>'
padding_string = '<pad>'
batch_size = 64
shuffle = False
use_pre_processing = True
separator = '|'
encoding = 'utf-8'
device = 'cpu'
word_dim = 300
lstm_dim = 300
lstm_layers = 1
dropout_prob = 0.05
bidirectional = False
alpha = 0.5
epochs = 10
ckpt_period = '1e'
val = True
val_period = '1e'
samples = 10
learning_rate = 0.001
learning_rate_decay = 0.01
max_patience = 5
max_decay_num = 5
patience_threshold = 0.98
```
4) Instantiate vocabulary and label vocabulary objects:
```
input_vocab = vocab.create_vocabulary(
    input_path=input_path,
    column_name=sentence_column,
    pad_string=padding_string,
    unk_string=unknown_string,
    encoding=encoding,
    separator=separator,
    use_pre_processing=use_pre_processing)

label_vocab = vocab.create_vocabulary(
    input_path = input_path,
    column_name =label_column,
    pad_string = padding_string,
    unk_string = unknown_string,
    encoding = encoding,
    separator = separator,
    is_label = True)
```   
5) Save vocabulary and label models:
```
vocab_path = os.path.join(save_dir, f'vocab-input.pkl')
pickle.dump(input_vocab, open(vocab_path, 'wb'))
vocab_label_path = os.path.join(save_dir, f'vocab-label.pkl')
pickle.dump(label_vocab, open(vocab_label_path, 'wb'))
```
6) Initialize BiLSTM-CRF model and set embedding:
```
crf = model.CRF(
        vocab_size=len(label_vocab),
        pad_idx=input_vocab.f2i[padding_string],
        unk_idx=input_vocab.f2i[unknown_string],
        device=device).to(device)

bilstmcr_model = model.LSTMCRF(
        device=device,
        crf=crf,
        vocab_size=len(input_vocab),
        word_dim=word_dim,
        hidden_dim=lstm_dim,
        layers=lstm_layers,
        dropout_prob=dropout_prob,
        bidirectional=bidirectional,
        alpha=alpha
    ).to(device)

bilstmcr_model.reset_parameters()

fasttext = utils.load_fasttext_embeddings(wordembed_path, padding_string)
bilstmcr_model.embeddings[0].weight.data = torch.from_numpy(fasttext[input_vocab.i2f.values()])
bilstmcr_model.embeddings[0].weight.requires_grad = False
```
7) Initialize LSTMCRFTrainer object:
```
trainer = LSTMCRFTrainer(
        bilstmcrf_model=bilstmcr_model,
        epochs=epochs,
        input_vocab=input_vocab,
        input_path=input_path,
        label_vocab=label_vocab,
        save_dir=save_dir,
        ckpt_period=utils.PeriodChecker(ckpt_period),
        val=val,
        val_period=utils.PeriodChecker(val_period),
        samples=samples,
        pad_string=padding_string,
        unk_string=unknown_string,
        batch_size=batch_size,
        shuffle=shuffle,
        label_column=label_column,
        encoding=encoding,
        separator=separator,
        use_pre_processing=use_pre_processing,
        learning_rate=learning_rate,
        learning_rate_decay=learning_rate_decay,
        max_patience=max_patience,
        max_decay_num=max_decay_num,
        patience_threshold=patience_threshold,
        val_path=val_path)
```
8) Train the model:
```
trainer.train()
```

## Prediction

The prediction could be done in two ways, with a single sentence or 
a batch of sentences.

### Single Prediction

To predict a single sentence, the method **predict_line** should be used. 
Example of initialization e usage:

**Important**: before label some sentence, it's needed to make some steps:
1) Import main packages;
2) Initialize model variables;
3) Read PosTagging model and embedding model;
4) Initialize and usage.

An example of the above steps could be found in the python code below:

1) Import main packages:
```
  import torch
  
  import TakeBlipPosTagger.utils as utils
  from TakeBlipPosTagger.predict import PosTaggerPredict
```
2) Initialize model variables:

In order to predict the sentences tags, the following variables should be
created:
- **model_path**: string with the path of PosTagging pickle model;
- **wordembed_path**: string with FastText embedding files
- **label_vocab**: string with the path of PosTagging pickle labels;
- **save_dir**: string with path and file name which will be used to
  save predicted sentences (for batch prediction);
- **padding_string**: string which represents the pad token;
- **encoding**: string with the encoding used in sentence;
- **separator**: string with file separator (for batch prediction);
- **sentence**: string with sentence to be labeled.

Example of variables creation:
```
model_path = '*.pkl'
wordembed_path = '*.kv'
label_vocab = '*.pkl'
save_dir = '*.csv'
padding_string = '<pad>'
encoding = 'utf-8'
separator = '|'
sentence = 'SENTENCE EXAMPLE TO PREDICT'
```
3) Read PosTagging model and embedding model:
```
bilstmcrf = torch.load(model_path)
embedding = utils.load_fasttext_embeddings(wordembed_path, padding)
```
4) Initialize and usage:

```
postagger_predicter = PosTaggerPredict(
        model=bilstmcrf,
        label_path=label_vocab,
        embedding=embedding,
        save_dir=save_dir,
        encoding=encoding,
        separator=separator)

print(postagger_predicter.predict_line(sentence))
```

### Batch Prediction

To predict a batch of sentences in a .csv file, another set of variables should
be created and passed to **predict_batch** method. The variables are the following:
- **input_path**: a string with path of the .csv file;
- **sentence_column**: a string with column name of .csv file;
- **unknown_string**: a string which represents unknown token;
- **batch_size**: number of sentences which will be predicted at the same time;
- **shuffle**: a boolean representing if the dataset is shuffled;
- **use_pre_processing**: a boolean indicating if sentence will be preprocessed;
- **output_lstm**: a boolean indicating if LSTM prediction will be saved.

Example of initialization e usage of **predict_batch** method:
```
input_path = '*.csv'
sentence_column = '*'
unknown = '<unk>'
batch_size = 64
shuffle = False
use_pre_processing = True
output_lstm = True

postagger_predicter.predict_batch(
            filepath=input_path,
            sentence_column=sentence_column,
            pad_string=padding_string,
            unk_string=unknown_string,
            batch_size=batch_size,
            shuffle=shuffle,
            use_pre_processing=use_pre_processing,
            output_lstm=output_lstm)
```
The batch sentences prediction will be saved in the given **save_dir** path.