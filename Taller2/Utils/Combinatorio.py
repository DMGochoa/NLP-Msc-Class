import itertools
import random
from .Clasificadores import Classifier
from .SeleccionDeTerminos import TermSelecting
from .PonderadoDeTerminos import TermWeighting
from .MetricasModelos import ModelMetrics
from .CustomLogger import CustomLogger


class CombinationOfMethods():
    """
    Class for running combinations of different term weighting, term selecting, 
    and classification methods on a given dataset.

    Attributes
    ----------
    classifiers : dict
        A dictionary with classification method names as keys and corresponding
        method functions as values.
    term_selecting : dict
        A dictionary with term selecting method names as keys and corresponding
        method functions as values.
    term_weight : dict
        A dictionary with term weighting method names as keys and corresponding
        method functions as values.
    random_selection : int, optional
        The number of method combinations to randomly select for execution.
    """

    def __init__(self, random_selection=None):
        """
        Initialize CombinationOfMethods object.

        Args:
            random_selection: int, optional
                The number of method combinations to randomly select for execution.
        """
        classifier = Classifier()
        term_selecting = TermSelecting()
        term_weighting = TermWeighting()
        self.classifiers = {
            'SVM': classifier.svm,
            'RANDOM_FOREST': classifier.random_forest,
            'NAIVE_BAYES': classifier.naive_bayes,
            'LOGISTIC_REGRESSION': classifier.logistic_regression,
            'GRADIENT_BOOSTING': classifier.gradient_boosting
        }
        self.term_selecting = {
            'ANOVA': term_selecting.select_terms_anova,
            'CHI2': term_selecting.select_terms_chi2,
            'MUTUAL_INFO': term_selecting.select_terms_mutual_info
        }
        self.term_weight = {
            'BINARY_IDF': term_weighting.compute_binary_idf,
            'TFIDF': term_weighting.compute_tfidf,
            'EUCLIDEAN_DISTANCE': term_weighting.normalize_by_euclidean_distance,
            'LENGTH': term_weighting.normalize_by_length
        }
        self.random_selection = random_selection
        self.logger = CustomLogger('Combinations of methods')

    def __combinations(self):
        """
        Generate combinations of methods for term weighting, term selecting, and 
        classification. If random_selection attribute is set, a subset of combinations
        is randomly selected.

        Returns:
            list: Combinations of term weighting, term selecting, and classification methods.
        """
        combinations = list(itertools.product(self.term_weight.keys(),
                                              self.term_selecting.keys(),
                                              self.classifiers.keys()))
        if self.random_selection is not None:
            combinations = random.sample(combinations, self.random_selection)
        return combinations

    def excute_combinations(self, X, y):
        """
        Execute the combinations of methods on the given dataset. For each combination,
        it performs term weighting, term selection, training of the classifier, and 
        prediction of the test set. It then calculates and stores the precision, accuracy, 
        sensitivity, and specificity of each method combination.

        Args:
            X : array-like of shape (n_samples, n_features)
                Training input samples.
            y : array-like of shape (n_samples,)
                Target values.

        Returns:
            dict: Results with method combination and the corresponding precision, 
            accuracy, sensitivity, and specificity.
        """
        self.results = {
            'term_weight': [],
            'term_selecting': [],
            'classifier': [],
            'precision': [],
            'accuracy': [],
            'sensitivity': [],
            'specificity': []
        }
        for selected_methods in self.__combinations():
            self.results['term_weight'].append(selected_methods[0])
            self.results['term_selecting'].append(selected_methods[1])
            self.results['classifier'].append(selected_methods[2])

            # Apply term weighting method on the input data.
            weighted_X = self.term_weight[selected_methods[0]](X)
            self.logger.debug(
                f'The type of the X element through term_weight: {type(weighted_X)}')
            print(weighted_X)
            # Apply term selection method on the weighted input data.
            selected_X = self.term_selecting[selected_methods[1]](
                weighted_X, y)
            self.logger.debug(
                f'The type of the X element through term_weight: {type(selected_X)}')

            # Split the selected input data into training and test sets.
            X_train, X_test, y_train, y_test = Classifier().split_data(selected_X, y)

            # Train the classifier on the training set.
            selected_classifier = self.classifiers[selected_methods[2]](
                X_train, y_train)

            # Predict the target values for the test set.
            y_pred = selected_classifier.predict(X_test)

            # Calculate and store the precision, accuracy, sensitivity, and specificity of the method combination.
            precision = ModelMetrics().precision(y_test, y_pred)
            self.results['precision'].append(precision)
            accuracy = ModelMetrics().accuracy(y_test, y_pred)
            self.results['accuracy'].append(accuracy)
            sensibility = ModelMetrics().recall(y_test, y_pred)
            self.results['sensitivity'].append(sensibility)
            specificity = ModelMetrics().specificity(y_test, y_pred)
            self.results['specificity'].append(specificity)
            self.logger.debug(f'Los datos de precision son: {precision}')
            self.logger.debug(f'Los datos de accuracy son: {accuracy}')
            self.logger.debug(f'Los datos de sensibility son: {sensibility}')
            self.logger.debug(f'Los datos de specificity son: {specificity}')

        # Return the results.
        return self.results
