from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.standard_py_parameter_type import StandardPythonParameterType
from api_postagging_predict import AzuremlPosTaggingPredict
import logging

def init():
    logging.info('Init start')
    global azureml_postagging
    azureml_postagging =  AzuremlPosTaggingPredict()
    logging.info('Init function finalized! All models were read!')

@input_schema('sentence', StandardPythonParameterType('Quero o meu boleto'))
@output_schema(StandardPythonParameterType(['quero o meu boleto', 'VERB ART PRON SUBS']))
def run(sentence):
    processed_sequence, predicted = azureml_postagging.predict(sentence)
    return [processed_sequence, predicted]