import torch
import logging
import os
from glob import glob

import TakeBlipPosTagger.utils as utils
from TakeBlipPosTagger.predict import PosTaggerPredict


def get_model_path(container, model_registry_name, model_file_name):
    azure_ml_dir = os.getenv('AZUREML_MODEL_DIR')
    logging.info('Azuredir list')
    logging.info(azure_ml_dir)
    logging.info('OS list azuredir')
    logging.info(os.listdir(azure_ml_dir))
    version_container_dir = os.path.join('*', container)
    azure_models_path = os.path.join(azure_ml_dir, '{}',
                                     version_container_dir, '{}')
    logging.info('Azure Path Test')
    logging.info(
        azure_models_path.format(model_registry_name, model_file_name))
    model_path = glob(azure_models_path.format(model_registry_name,
                                               model_file_name))
    logging.info(model_path[0])
    return model_path[0]


class AzuremlPosTaggingPredict:
    def __init__(self):
        self.pad_string = '<pad>'
        self.unk_string = '<unk>'
        self.model_folder = 'outputs'
        try:
            self.set_postagging_predict()
        except Exception as e:
            logging.error('Error while setting postagging predict', e)

    def set_postagging_predict(self):
        embedding_model = self.get_embedding_model()

        postagging_model = self.get_postagging_model(self.model_folder)

        postagging_label_path = self.get_postagging_label_path(self.model_folder)

        self.postagger_predicter = PosTaggerPredict(
            model=postagging_model,
            label_path=postagging_label_path,
            embedding=embedding_model
        )

    def predict(self, input_sentence):
        return self.postagger_predicter.predict_line(input_sentence)
    
    def predict_batch(self, batch_size, shuffle, use_pre_processing,
                      output_lstm, input_sentences):
        return self.postagger_predicter.predict_batch('',
                                                      '',
                                                      self.pad_string,
                                                      self.unk_string,
                                                      batch_size,
                                                      shuffle,
                                                      use_pre_processing,
                                                      output_lstm,
                                                      input_sentences)

    def get_embedding_model(self):
        logging.info('Getting embedding model...')
        embedding_path = get_model_path(self.model_folder,
                                        'EmbeddingModel',
                                        'embedding.kv')
        return utils.load_fasttext_embeddings(embedding_path, self.pad_string)

    @staticmethod
    def get_postagging_model(model_folder):
        logging.info('Started reading model ...')
        postag_model_path = get_model_path(model_folder,
                                           'PostaggingModel',
                                           'model.pkl')
        postagging_model = torch.load(postag_model_path)
        return postagging_model

    @staticmethod
    def get_postagging_label_path(model_folder):
        logging.info('Getting labels path...')
        return get_model_path(model_folder,
                              'PostaggingModel',
                              'vocab-label.pkl')
