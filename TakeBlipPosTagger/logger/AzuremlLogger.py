import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


class AzuremlLogger(object):
    def __init__(self, run, torch_version):
        self.run = run
        self.torch_version = torch_version

    def save_validation_metrics(self, metric):
        self.run.log('negative_loglik', metric.item())

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
        self.run.log_image(image_file_name, plot=plt)

    def save_confusion_matrix_from_tensor(self, confusion_matrix, labels,
                                          current_epoch):
        image_file_name = 'confusion_matrix_validation_{}.png'.format(
            current_epoch)
        plt.figure(figsize=(16, 10))
        sns.heatmap(confusion_matrix.long().numpy(), annot=True,
                    cmap=plt.cm.Blues, xticklabels=labels, yticklabels=labels,
                    fmt='d')
        plt.yticks(rotation=0)
        plt.savefig(image_file_name)
        self.run.log_image(image_file_name, plot=plt)

    def save_loss_convergence(self, loss):
        image_file_name = 'validation_loss.png'
        fig = plt.figure()
        plt.plot(loss, color='blue')
        plt.legend(['Validation Loss'], loc='upper right')
        plt.xlabel('Number of validating examples')
        plt.ylabel('Negative log likelihood loss')
        fig.savefig(image_file_name)
        self.run.log_image(image_file_name, plot=fig)

    def save_report(self, report):
        self.run.log('Accuracy', report['accuracy'])
        self.run.log('Precision - Macro Avg', report['macro avg']['precision'])
        self.run.log('Recall - Macro Avg', report['macro avg']['recall'])
        self.run.log('F1-score - Macro Avg', report['macro avg']['f1-score'])
        self.run.log('Precision - Weighted Avg',
                     report['weighted avg']['precision'])
        self.run.log('Recall - Weighted Avg', report['weighted avg']['recall'])
        self.run.log('F1-score - Weighted Avg',
                     report['weighted avg']['f1-score'])

    def save_negative_loglik(self, negative_loglik):
        self.run.log('negative_loglik_train', negative_loglik.item())

    def save_metrics(self, confusion_matrix, labels):
        precision = confusion_matrix.diag() / confusion_matrix.sum(dim=0)
        recall = confusion_matrix.diag() / confusion_matrix.sum(dim=1)
        f1_score = 2 * (precision * recall / (precision + recall))

        for index, label in enumerate(labels):
            self.run.log(label + ' Precision', precision[index].numpy().item())
            self.run.log(label + ' Recall', recall[index].numpy().item())
            self.run.log(label + ' F1-score', f1_score[index].numpy().item())

        self.run.log('Model Precision',
                     precision[precision >= 0].mean().numpy().item())
        self.run.log('Model Recall', recall[recall >= 0].mean().numpy().item())
        self.run.log('Model F1-score',
                     f1_score[f1_score >= 0].mean().numpy().item())
