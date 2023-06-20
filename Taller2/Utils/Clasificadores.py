from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from .CustomLogger import CustomLogger

class Classifier:
    def __init__(self):
        self.logger = CustomLogger('Classifier')
        
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into training and testing sets.

        Args:
            X: Input data
            y: Target values
            test_size: Proportion of the dataset to include in the test split (default is 0.2)
            random_state: Controls the shuffling applied to the data before applying the split (default is 42)

        Returns:
            Split data: X_train, X_test, y_train, y_test
        """
        self.logger.info('Splitting data into training and testing sets')
        return train_test_split(X, y, test_size=test_size, random_state=random_state)


    def logistic_regression(self, X, y):
        """Classify using Logistic Regression.

        Args:
            X: Training input samples
            y: Target values

        Returns:
            Fitted classifier
        """
        self.logger.info('Starting Logistic Regression Classification')
        classifier = LogisticRegression()
        return classifier.fit(X, y)

    def svm(self, X, y):
        """Classify using Support Vector Machines (SVM).

        Args:
            X: Training input samples
            y: Target values

        Returns:
            Fitted classifier
        """
        self.logger.info('Starting SVM Classification')
        classifier = SVC()
        return classifier.fit(X, y)

    def random_forest(self, X, y):
        """Classify using Random Forests.

        Args:
            X: Training input samples
            y: Target values

        Returns:
            Fitted classifier
        """
        self.logger.info('Starting Random Forests Classification')
        classifier = RandomForestClassifier()
        return classifier.fit(X, y)

    def gradient_boosting(self, X, y):
        """Classify using Gradient Boosting.

        Args:
            X: Training input samples
            y: Target values

        Returns:
            Fitted classifier
        """
        self.logger.info('Starting Gradient Boosting Classification')
        classifier = GradientBoostingClassifier()
        return classifier.fit(X, y)

    def naive_bayes(self, X, y):
        """Classify using Naive Bayes.

        Args:
            X: Training input samples
            y: Target values

        Returns:
            Fitted classifier
        """
        self.logger.info('Starting Naive Bayes Classification')
        classifier = MultinomialNB()
        return classifier.fit(X, y)
