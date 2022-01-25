import os
import shutil
from datetime import datetime
import azureml._restclient.snapshots_client
from azureml.core import Model, Run
from azureml.core import Workspace, Experiment, ScriptRunConfig, Datastore
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice, Webservice, LocalWebservice
from azureml.core.environment import Environment
from azureml.core.conda_dependencies import CondaDependencies

#TODO: parametrizar as variaveis abaixo e as variaveis de modelo para permitir a customização do deploy

subscription_id = ''
resource_group = ''
workspace_name = ''
postagging_deploy_env_name = "hmg_takeblip_postagger_env"
postagging_deploy_env_file = '../files/conda_env/postagging-predict-linux.yml'
service_name = 'hmg-postagging-line-aci'

def get_aml_workspace(sid, rgn, wsn):
    ws = Workspace.get(name=wsn, 
                   subscription_id=sid, 
                   resource_group=rgn)
    return ws

def main():
    print("Start deploy")
    ws = get_aml_workspace(subscription_id, resource_group, workspace_name)
    print("AML Workspace - OK")
    try:
        postagging_env = Environment.get(workspace=ws, name=postagging_deploy_env_name)
        print("Env {} getted".format(postagging_env.name))
    except Exception:
        postagging_env = Environment.from_conda_specification(name=postagging_deploy_env_name, file_path=postagging_deploy_env_file)
        print("Env {} created".format(postagging_env.name))
        postagging_env.register(workspace=ws)
        print("Env {} registered at {}".format(postagging_env.name, ws.name))
    
    print("Searching for models...")

    postagging_model = Model(ws, name='PostaggingModel', version=37)
    postagging_label = Model(ws, name='PostaggingLabel', version=18)
    embedding_model = Model(ws, name='EmbeddingModel', version=1)

    print("Using models: {}.{}, {}.{} and {}.{} for deploy PosTagging Predict Line".format(
        postagging_model.name, postagging_model.version
        ,postagging_label.name, postagging_label.version
        ,embedding_model.name, embedding_model.version
        ))

    print('Creating Temp Dir')
    temp_dir = datetime.now().strftime("%Y%m%d%H%M%S%f")
    os.mkdir(temp_dir)
    files_to_copy = [('','azureml_predict.py'),('','api_postagging_predict.py')]
    for file_tuple in files_to_copy:
        file_source = os.path.join(file_tuple[0], file_tuple[1])
        file_dest = os.path.join(temp_dir, file_tuple[1])
        print((file_source,file_dest))
        shutil.copyfile(file_source, file_dest)

    entry_point = 'azureml_predict.py'

    print("Define entry point: {}".format(entry_point))
    inference_config = InferenceConfig(entry_script=entry_point, 
                                   environment=postagging_env, 
                                   source_directory=temp_dir)

    print("Define deployment config")
    deployment_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=4)
    #deployment_config = LocalWebservice.deploy_configuration()
    

    print("Deploying service '{}' now!".format(service_name))
    service = Model.deploy(ws, name=service_name, 
                       models=[postagging_model, postagging_label, embedding_model], 
                       inference_config=inference_config, 
                       deployment_config=deployment_config,
                       overwrite=True)

    print('Remove Temp Dir')
    shutil.rmtree(temp_dir)    
    print("Waiting for deploy finish")
    try:
        service.wait_for_deployment(show_output=True)
    except Exception as ex:
        print('Error',ex)

    print(service.get_logs())
    print("Done", service.state)


if __name__ == '__main__':
    main()