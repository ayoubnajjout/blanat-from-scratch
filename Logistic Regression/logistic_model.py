import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = []
        self.biases = []
        self.classes = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_samples, n_features = X.shape
        self.weights = np.zeros((len(self.classes), n_features))
        self.biases = np.zeros(len(self.classes))

        for idx, c in enumerate(self.classes):

            y_binary = np.where(y == c, 1, 0)

            weights = np.zeros(n_features)
            bias = 0

            for _ in range(self.n_iterations):
                linear_model = np.dot(X, weights) + bias
                y_predicted = self.sigmoid(linear_model)

                dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y_binary))
                db = (1 / n_samples) * np.sum(y_predicted - y_binary)

                weights -= self.learning_rate * dw
                bias -= self.learning_rate * db

            self.weights[idx, :] = weights
            self.biases[idx] = bias

    def predict(self, X):
        linear_models = np.dot(X, self.weights.T) + self.biases
        y_predicted = self.sigmoid(linear_models)
        return self.classes[np.argmax(y_predicted, axis=1)]

    def score(self, X, y):
        accuracy = np.mean(X == y)
        return accuracy
