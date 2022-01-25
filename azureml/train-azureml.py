import os
import azureml._restclient.snapshots_client
from azure.core.exceptions import ResourceNotFoundError, ResourceExistsError
from azureml.core.compute import ComputeTarget
from azureml.core.runconfig import RunConfiguration
from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig, Datastore, Dataset
from azureml.data.datapath import DataPath
from azureml.pipeline.core import Pipeline, PipelineData, PipelineParameter
from azureml.train.estimator import Estimator
from azureml.pipeline.steps import PythonScriptStep, EstimatorStep, ParallelRunStep, ParallelRunConfig
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import time
import timeit
from datetime import datetime
import shutil

def get_env_list(workspace):
    envs = Environment.list(ws)
    keys_to_remove = [env_key for env_key in envs if env_key.startswith("Azure")]
    for key_to_remove in keys_to_remove:
        del envs[key_to_remove]
    return [env for env in envs]

def get_or_create_env(env_name, env_file, workspace, force=False):
    env = None
    if force:
        return Environment.from_conda_specification(name=env_name, file_path=env_file)
    try:
        env = Environment.get(workspace, env_name)
    except Exception as e:
        error_404 = 'Error retrieving the environment definition. Code: 404' in e.args[0]
        if not error_404:
            raise e
        env = Environment.from_conda_specification(name=env_name, file_path=env_file)
    
    return env

def get_run_config(cluster_name, env, workspace):
    pipeline_run_config = RunConfiguration()
    pipeline_run_config.target = ComputeTarget(workspace=ws, name=cluster_name)
    pipeline_run_config.environment = env
    return pipeline_run_config

##################################
##################################
##################################

print('Train pipeline start - Setting configs up!')

env_name = 'hmg_takeblip_postagger_env'
env_file = 'files\\conda_env\\postagging-predict-linux.yml'
env_update = True

cluster_name = 'training-cluster'

storage_of_model = {
    'name': 'dardatasets',
    'conn_string': '',
    'container': 'postaggingfiles'
}

storage_of_embedding = {
    'name': 'darmodels',
    'conn_string': '',
    'container': 'spellcheckedembedding'
}

datastore_model_name = 'postaggingfiles'
datastore_embedding_name = 'fasttext'

experiment_name = 'hmg-postagging'

subscription_id = '86d359e5-24fc-4007-9040-6b4de4727f31'
resource_group = 'hmg-blip-dataanalytics'
workspace_name = 'hmg-research-machine-learning'

azureml._restclient.snapshots_client.SNAPSHOT_MAX_SIZE_BYTES = 3e+9

print('Getting workspace')
ws = Workspace.get(name = workspace_name, subscription_id = subscription_id, resource_group = resource_group)
print('Getting workspace - DONE!')

print('Setting environment and run config')
azureml_env = get_or_create_env(env_name, env_file, ws, False)
if env_update:
    env_register = azureml_env.register(workspace=ws)
run_config = get_run_config(cluster_name, azureml_env, ws)
print('Setting environment and run config - DONE!')

print('Getting datastores')
model_datastore = Datastore.get(ws, datastore_model_name)
embedding_datastore = Datastore.get(ws, datastore_embedding_name)
#https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.data.dataset_factory.filedatasetfactory?view=azure-ml-py#from-files-path--validate-true-
datastore_embedding_path = [
       DataPath(embedding_datastore, 'titan_v2_after_correction_fasttext_window4_mincount20_cbow.kv'),
       DataPath(embedding_datastore, 'titan_v2_after_correction_fasttext_window4_mincount20_cbow.kv.vectors_ngrams.npy'),
       DataPath(embedding_datastore, 'titan_v2_after_correction_fasttext_window4_mincount20_cbow.kv.vectors_vocab.npy'),
       DataPath(embedding_datastore, 'titan_v2_after_correction_fasttext_window4_mincount20_cbow.kv.vectors.npy')
   ]

datastore_input_path = [
       DataPath(model_datastore, 'sample.csv'),
       DataPath(model_datastore, 'sample_validation.csv'),
   ]

input_dataset = Dataset.File.from_files(path=datastore_input_path)
embedding_dataset = Dataset.File.from_files(path=datastore_embedding_path)

model_pipeline_container_folder = PipelineData('postagging_files_from_training', datastore=model_datastore)
print('Getting datastores - DONE!')

print('Setting Train Pipeline')
input_path = 'postaggingfiles/sample.csv'
separator = ','
sentence_column = 'MessageProcessed'
label_column = 'Tags'
wordembedding_path = 'fasttext/titan_v2_after_correction_fasttext_window4_mincount20_cbow.kv'
n_epochs = 10
validation_path = 'postaggingfiles/sample_validation.csv'
validation_period = '100i' 
batch_size = 64 
learning_rate = 0.001
learning_rate_decay = 0.1
max_patience = 5 
patience_threshold = 0.98
max_decay_num = 5

input_path_param = PipelineParameter(name = 'input_path', default_value = input_path)
separator_param = PipelineParameter(name = 'separator', default_value = separator)
sentence_column_param = PipelineParameter(name = 'sentence_column', default_value = sentence_column)
label_column_param = PipelineParameter(name = 'label_column', default_value = label_column)
wordembedding_param = PipelineParameter(name = 'wordembedding', default_value = wordembedding_path)
n_epochs_param = PipelineParameter(name = 'n_epochs', default_value = n_epochs)
validation_path_param = PipelineParameter(name = 'validation_path', default_value = validation_path)
validation_period_param = PipelineParameter(name = 'validation_period', default_value = validation_period)
batch_size_param = PipelineParameter(name = 'batch_size', default_value = batch_size)
learning_rate_param = PipelineParameter(name = 'learning_rate', default_value = learning_rate)
learning_rate_decay_param = PipelineParameter(name = 'learning_rate_decay', default_value = learning_rate_decay)
max_patience_param = PipelineParameter(name = 'max_patience', default_value = max_patience)
max_decay_num_param = PipelineParameter(name = 'max_decay_num', default_value = max_decay_num)
patience_threshold_param = PipelineParameter(name = 'patience_threshold', default_value = patience_threshold)
dropout_prob_param = PipelineParameter(name = 'dropout_prob', default_value = 0.05)
ckpt_period_param = PipelineParameter(name = 'ckpt_period', default_value = '1e')
lstm_layers_param = PipelineParameter(name = 'lstm_layers', default_value = 1)
alpha_param = PipelineParameter(name = 'alpha', default_value = 1)

postagging_script_params = [
  '--input-path', input_path_param,
  '--separator', separator_param,
  '--sentence_column', sentence_column_param,
  '--label_column', label_column_param,
  '--dropout-prob', dropout_prob_param,
  '--save-dir', model_pipeline_container_folder,
  '--wordembed-path', wordembedding_param,                                    
  '--epochs', n_epochs_param,
  '--val-path', validation_path_param,
  '--val-period', validation_period_param,
  '--batch-size', batch_size_param, 
  '--learning-rate', learning_rate_param, 
  '--learning-rate-decay', learning_rate_decay_param,
  '--max-patience', max_patience_param,
  '--max-decay-num', max_decay_num_param,
  '--patience-threshold', patience_threshold_param,
  '--ckpt-period', ckpt_period_param,
  '--input-data-ref', input_dataset.as_mount(),
  '--wordembed-data-reference', embedding_dataset.as_mount(),
  '--lstm-layers', lstm_layers_param,
  '--alpha', alpha_param,
  '--val',
  '--bidirectional']

print('Creating Temp Dir')
temp_dir = datetime.now().strftime("%Y%m%d%H%M%S%f")
os.mkdir(temp_dir)
files_to_copy = [('','mytrain.py'),('TakeBlipPosTagger','vocab.py'),('TakeBlipPosTagger','model.py')]
for file_tuple in files_to_copy:
    file_source = os.path.join(file_tuple[0], file_tuple[1])
    file_dest = os.path.join(temp_dir, file_tuple[1])
    print((file_source,file_dest))
    shutil.copyfile(file_source, file_dest)

print('Creating Temp Dir - DONE!')
train_step = PythonScriptStep(
    name="MyLocalTrain",
    script_name='mytrain.py',
    source_directory=temp_dir,
    arguments=postagging_script_params,
    outputs=[model_pipeline_container_folder],
    compute_target=run_config.target,
    runconfig=run_config,
    allow_reuse=True
)

pipeline_steps = [train_step]
pipeline = Pipeline(workspace = ws, steps = pipeline_steps)
print('Setting Train Pipeline - Pipeline Built - DONE!')
experiment = Experiment(workspace=ws, name=experiment_name)
pipeline_run = experiment.submit(pipeline, regenerate_outputs=True)
print('Pipeline submitted for execution.')
print(pipeline_run.get_portal_url())
print('Waiting train pipeline to finish')
shutil.rmtree(temp_dir)
print('Remove Temp Dir')
pipeline_run.wait_for_completion()
print('Train pipeline finish successfully!')