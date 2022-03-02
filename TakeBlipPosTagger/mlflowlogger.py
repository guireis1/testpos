import os
import psutil
import mlflow
import seaborn as sns
import matplotlib.pyplot as plt


def save_confusion_matrix_from_tensor(confusion_matrix, labels,
                                      current_epoch, save_dir):
    """Receive a confusion matrix from tensor, generate a image 
    with seaborn and save as .png in mlflow experiment

    Args:
        confusion_matrix (torch.Tensor): Tensor of confusion matrix
        labels (list): Classification labels
        current_epoch (int): Current epoch number
        save_dir (str): Directory to save
    """
    image_file_name = 'confusion_matrix_validation_{}.png'.format(
        current_epoch)
    plt.figure(figsize=(16, 10))
    matrix = sns.heatmap(confusion_matrix.long().numpy(), annot=True,
                         cmap=plt.cm.Blues, xticklabels=labels, yticklabels=labels,
                         fmt='d')
    plt.yticks(rotation=0)
    plt.savefig(os.path.join(save_dir, image_file_name))
    mlflow.log_artifact(os.path.join(
        save_dir, image_file_name), artifact_path="images")


def save_metrics(confusion_matrix, labels):
    """Receive a confusion matrix from tensor, 
    calculates desirable metrics and log in mlflow experiment

    Args:
        confusion_matrix (torch.Tensor): Tensor of confusion matrix
        labels (list): Classification labels
    """
    precision = confusion_matrix.diag() / confusion_matrix.sum(dim=0)
    recall = confusion_matrix.diag() / confusion_matrix.sum(dim=1)
    f1_score = 2*(precision*recall / (precision + recall))

    for index, label in enumerate(labels):
        mlflow.log_metric(label + ' F1-score', f1_score[index].numpy().item())

    mlflow.log_metric('Model Precision',
                      precision[precision >= 0].mean().numpy().item())
    mlflow.log_metric(
        'Model Recall', recall[recall >= 0].mean().numpy().item())
    mlflow.log_metric('Model F1-score',
                      f1_score[f1_score >= 0].mean().numpy().item())


def save_report(report):
    """Receive a metric report and log in mlflow experiment

    Args:
        report (dict): Dictionary of calculated metrics
    """
    mlflow.log_metric('Accuracy', report['accuracy'])
    mlflow.log_metric('Precision - Macro Avg',
                      report['macro avg']['precision'])
    mlflow.log_metric('Recall - Macro Avg', report['macro avg']['recall'])
    mlflow.log_metric('F1-score - Macro Avg', report['macro avg']['f1-score'])
    mlflow.log_metric('Precision - Weighted Avg',
                      report['weighted avg']['precision'])
    mlflow.log_metric('Recall - Weighted Avg',
                      report['weighted avg']['recall'])
    mlflow.log_metric('F1-score - Weighted Avg',
                      report['weighted avg']['f1-score'])


def save_system_metrics():
    """Log system metrics in mlflow experiment
    """
    mlflow.log_metric('cpu_percent', float(psutil.cpu_percent()))
    mlflow.log_metric('memory_percent', float(psutil.virtual_memory().percent))


def save_model(model, model_name):
    """Save model as artifact in mlflow experiment

    Args:
        model: Trained LSTM model on pytorch
        model_name (str): Name of the saved model
    """
    mlflow.sklearn.log_model(model, artifact_path='models',
                             registered_model_name=model_name)


def save_predict(save_dir, file_name):
    """Save predict file as artifact in mlflow experiment

    Args:
        save_dir (str): Directory of the file
        file_name (str): File name
    """
    mlflow.log_artifact(os.path.join(save_dir, file_name),
                        artifact_path="data")


def save_param(name, variable):
    """Save parameter in mlflow experiment

    Args:
        name (str): Name to log on mlflow
        variable (Union[int, float]): Variable that is going to be logged
    """
    mlflow.log_param(name, variable)


def save_metric(name, variable):
    """Save metric in mlflow experiment

    Args:
        name (str): Name to log on mlflow
        variable (Union[int, float]): Variable that is going to be logged
    """
    mlflow.log_metric(name, variable)
