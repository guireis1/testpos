from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.standard_py_parameter_type import StandardPythonParameterType
from api_postagging_predict import AzuremlPosTaggingPredict
import logging


def init():
    global azureml_postagging
    try:
        azureml_postagging = AzuremlPosTaggingPredict()
    except Exception as e:
        print('error', e)
    logging.info('Batch init function finalized! All models were read!')

@input_schema('sentences', StandardPythonParameterType([{'id':1,'sentence':'Quero o meu boleto'}, {'id':2,'sentence':'ñ consegui contato por telefone'}]))
@output_schema(StandardPythonParameterType([{'id':'1','processed_sentence': 'quero o meu boleto', 'tags': 'VERB ART PRON SUBS'}, {'id':'2','processed_sentence':'não consegui contato por telefone','tags':'ADV VERB SUBS PREP SUBS'}]))
def run(sentences):
    predictions_list = azureml_postagging.predict_batch(
        batch_size=50,
        shuffle=True,
        use_pre_processing=True,
        output_lstm='',
        input_sentences=sentences)
    logging.info('Run function finalized! All sentences were predicted!')
    return predictions_list