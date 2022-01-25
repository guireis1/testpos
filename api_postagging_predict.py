import torch
import logging
import os
import json
from glob import glob
from shutil import copyfile
import TakeBlipPosTagger.utils as utils
from TakeBlipPosTagger.predict import PosTaggerPredict

class AzuremlPosTaggingPredict():
    def __init__(self):
        self.pad_string = '<pad>'
        self.unk_string = '<unk>'
        try:
            self.set_postagging_predict()
        except Exception as e:
            logging.error('Error while setting postagging predict', e)

    def set_postagging_predict(self):
        embedding_model = self.get_embedding_model(
            embedding_container = 'spellcheckedembedding', 
            embedding_model_registry_name = 'EmbeddingModel',
            embedding_name = 'titan_v2_after_correction_fasttext_window4_mincount20_cbow.kv'
        )
        postagging_model = self.get_postagging_model(
            postagging_container = 'postagging_files_from_training',
            postagging_model_registry_name = 'PostaggingModel'
        )
        postagging_label_path = self.get_postagging_label_path(
            postagging_container='postagging_files_from_training',
            postagging_label_registry_name='PostaggingLabel'
        )
        self.postagger_predicter = PosTaggerPredict(
            model=postagging_model,
            label_path=postagging_label_path,
            embedding=embedding_model
        )
    def predict(self, input_sentence):
        return self.postagger_predicter.predict_line(input_sentence)
    
    def predict_batch(self, batch_size, shuffle, use_pre_processing, output_lstm, input_sentences):
        return self.postagger_predicter.predict_batch('', '', self.pad_string, self.unk_string,
         batch_size, shuffle, use_pre_processing, output_lstm, input_sentences)

    def get_embedding_model(self, embedding_container, embedding_model_registry_name, embedding_name):
        logging.info('Getting embedding model...')
        embedding_path = self.__get_model_path(embedding_container, embedding_model_registry_name, embedding_name)
        return utils.load_fasttext_embeddings(embedding_path, self.pad_string)

    def get_postagging_model(self, postagging_container, postagging_model_registry_name):
        logging.info('Started reading model ...')
        self.__copy_python_file(postagging_container, postagging_model_registry_name, 'model.py')
        postag_model_path = self.__get_model_path(postagging_container, postagging_model_registry_name, 'model.pkl')
        postagging_model = torch.load(postag_model_path)
        logging.info('Loaded postagging model ...')
        return postagging_model

    def get_postagging_label_path(self, postagging_container, postagging_label_registry_name):
        logging.info('Getting labels path...')
        self.__copy_python_file(postagging_container, postagging_label_registry_name, 'vocab.py')
        return self.__get_model_path(postagging_container, postagging_label_registry_name, 'vocab-label.pkl')

    def __copy_python_file(self, postagging_container, postagging_registry_name, file_name):
        postagging_code_path = self.__get_model_path(postagging_container, postagging_registry_name, file_name)
        copyfile(postagging_code_path, os.path.join(os.getcwd() , file_name))

    def __get_model_path(self, container, model_registry_name, model_file_name):
        azure_ml_dir = os.getenv('AZUREML_MODEL_DIR')
        logging.info('Azuredir list')
        logging.info(azure_ml_dir)
        logging.info('OS list azuredir')
        logging.info(os.listdir(azure_ml_dir))
        version_container_dir = os.path.join('*', container) 
        azure_models_path = os.path.join(azure_ml_dir, '{}', version_container_dir, '{}')
        logging.info('Azure Path Test')
        logging.info(azure_models_path.format(model_registry_name, model_file_name))
        model_path = glob(azure_models_path.format(model_registry_name, model_file_name))[0]
        logging.info(model_path)
        return model_path