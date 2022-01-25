import os
import mlflow
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def save_confusion_matrix(self, targets_all, preds_all, current_epoch):
  image_file_name = 'confusion_matrix_epoch_{}.png'.format(current_epoch)
  labels = list(set(targets_all))
  labels.sort()
  cm = confusion_matrix(targets_all, preds_all)
  plt.figure(figsize=(16, 10))
  sns.heatmap(cm, annot=True, cmap=plt.cm.Blues, xticklabels=labels,
              yticklabels=labels, fmt='d')
  plt.yticks(rotation=0)
  plt.savefig(image_file_name)
  #self.run.log_image(image_file_name, plot=plt)
  mlflow.log_artifact(run_id, image_file_name)
  
def save_confusion_matrix_from_tensor(confusion_matrix, labels,
                                    current_epoch, run_id, save_dir):
  image_file_name = 'confusion_matrix_validation_{}.png'.format(
      current_epoch)
  plt.figure(figsize=(16, 10))
  matrix = sns.heatmap(confusion_matrix.long().numpy(), annot=True,
              cmap=plt.cm.Blues, xticklabels=labels, yticklabels=labels,
              fmt='d')
  plt.yticks(rotation=0)
  plt.savefig(os.path.join(save_dir, image_file_name))
  #self.run.log_image(image_file_name, plot=plt)
  mlflow.log_artifact(os.path.join(save_dir, image_file_name), artifact_path="data")

# COMMAND ----------


