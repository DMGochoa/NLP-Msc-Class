from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from .CustomLogger import CustomLogger
import numpy as np


class ModelMetrics:
    def __init__(self):
        self.logger = CustomLogger('ModelMetrics')

    def accuracy(self, y_true, y_pred):
        """Compute the accuracy.

        Args:
            y_true: True target values
            y_pred: Estimated targets as returned by a classifier

        Returns:
            Accuracy of the classifier
        """
        self.logger.info('Calculating accuracy')
        return accuracy_score(y_true, y_pred)

    def precision(self, y_true, y_pred, average='micro'):
        """Compute the precision.

        Args:
            y_true: True target values
            y_pred: Estimated targets as returned by a classifier

        Returns:
            Precision of the classifier
        """
        self.logger.info('Calculating precision')
        return precision_score(y_true, y_pred, average=average)

    def recall(self, y_true, y_pred, average='micro'):
        """Compute the recall/sensitivity.

        Args:
            y_true: True target values
            y_pred: Estimated targets as returned by a classifier

        Returns:
            Recall of the classifier
        """
        self.logger.info('Calculating recall')
        return recall_score(y_true, y_pred,average=average)

    def specificity(self, y_true, y_pred):
        """Compute the specificity for multiclass classification.
        
        Computes specificity for each class and returns the average.
        
        Args:
            y_true: True target values
            y_pred: Estimated targets as returned by a classifier

        Returns:
            Average specificity of the classifier
        """
        self.logger.info('Calculating specificity')
        cm = confusion_matrix(y_true, y_pred)
        fp = cm.sum(axis=0) - np.diag(cm)
        fn = cm.sum(axis=1) - np.diag(cm)
        tp = np.diag(cm)
        tn = cm.sum() - (fp + fn + tp)

        # Sensitivity, hit rate, recall, or true positive rate
        tpr = tp / (tp + fn)
        # Specificity or true negative rate
        tnr = tn / (tn + fp)
        
        return np.mean(tnr)  # Returning the average specificity

