from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, f_classif
from .CustomLogger import CustomLogger
import numpy as np

class TermSelecting:
    def __init__(self, k=1000):
        self.k = k
        self.logger = CustomLogger('Term Selecting')

    def select_terms_chi2(self, X, y):
        """Use Chi-Squared test to select terms.

        This technique selects the terms that have the highest chi-squared values relative to the response.

        Args:
            X: Training input samples
            y: Target values

        Returns:
            Selected features
        """
        self.logger.info('Starting Chi-Squared term selection')
        selector = SelectKBest(chi2, k=self.k)
        return selector.fit_transform(X, y)


    def select_terms_mutual_info(self, X, y):
        """Use Mutual Information to select terms.

        This technique selects the terms that provide the most mutual information with the target.

        Args:
            X: Training input samples
            y: Target values

        Returns:
            Selected features
        """
        self.logger.info('Starting Mutual Information term selection')
        selector = SelectKBest(mutual_info_classif, k=self.k)
        return selector.fit_transform(X, y)

    def select_terms_anova(self, X, y):
        """Use Analysis of Variance (ANOVA) to select terms.

        This technique selects the terms that have the highest variance between classes.

        Args:
            X: Training input samples
            y: Target values

        Returns:
            Selected features
        """
        self.logger.info('Starting ANOVA term selection')
        selector = SelectKBest(f_classif, k=self.k)
        return selector.fit_transform(X, y)
