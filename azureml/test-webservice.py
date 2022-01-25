import json
import requests
from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig, Datastore, Dataset
from azureml.core.webservice import AciWebservice, Webservice

subscription_id = ''
resource_group = ''
workspace_name = ''
service_name = 'hmg-postagging-line-aci'
ws = Workspace.get(name = workspace_name, subscription_id = subscription_id, resource_group = resource_group)
service = Webservice(name=service_name, workspace=ws)
headers = {'Content-Type': 'application/json'}
test_sample = json.dumps({'sentence': 'quero meu boleto agora, pode ser'})
response = requests.post(service.scoring_uri, data=test_sample, headers=headers)

for sentence in response.json():
  print(sentence)