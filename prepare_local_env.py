print("Starting files setup to run local predict")

import os
from azureml.core import Workspace
from azureml.core.model import Model

ws_name = 'hmg-research-machine-learning'
subs_id = '86d359e5-24fc-4007-9040-6b4de4727f31'
rg_name = 'hmg-blip-dataanalytics'

print("Using azureml-sdk to get Workspace '{}' inside Resource Group '{}'".format(ws_name, rg_name))

ws = Workspace.get(name=ws_name, 
        subscription_id=subs_id, 
        resource_group=rg_name)

print("Successfully get workspace {} ".format(ws.name))

needed_models_list = [('EmbeddingModel',2), ('PostaggingModel',24), ('PostaggingLabel',3)]
heavy_folder = 'files\\input\\heavy'
output_folder = os.getcwd() + '\\' + heavy_folder

print("Starting models download")
for model_name in needed_models_list:
    print("\tGetting model", model_name[0])
    model = Model(workspace=ws, name=model_name[0], version=model_name[1])
    print("\tOk... Starting download of {} in {}".format(model.id, output_folder))
    try:
        m = model.download(target_dir=output_folder, exist_ok=False)
        print("\tDownload finished\n")
    except Exception as ex:
        if 'File already exists' in ex.args[0]:
            print ('\tpass')
        else:
            raise ex

print("End of setup. Now you can run the following command:\n\n")
print('python predict.py --model-path {} --input-sentence "hoje, eu quero prever essa frase" --label-vocab {} --save-dir files/output/pred.csv --wordembed-path {}'.format(
    heavy_folder+'\\postagging_files_from_training\\model.pkl',
    heavy_folder+'\\postagging_files_from_training\\vocab-label.pkl',
    heavy_folder+'\\spellcheckedembedding\\titan_v2_after_correction_fasttext_window4_mincount20_cbow.kv'
))
