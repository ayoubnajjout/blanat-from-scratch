import numpy as np

class GaussianNB:
    def __init__(self):
        self.classes = None
        self.parameters = {}

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y)
        self.classes = np.unique(y)
        
        for c in self.classes:
            X_c = X[y == c]
            mean = np.mean(X_c, axis=0)
            var = np.var(X_c, axis=0) 
            prior = X_c.shape[0] / X.shape[0]
            self.parameters[c] = {"mean": mean, "var": var, "prior": prior}

    def __likelihood(self, mean, var, x):
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def predict(self, X):
        X = np.array(X, dtype=float)
        y_pred = []

        for x in X:
            posteriors = []
            
            for c in self.classes:
                params = self.parameters[c]
                prior = np.log(params["prior"])
                likelihood = np.sum(np.log(self.__likelihood(params["mean"], params["var"], x)))
                posterior = prior + likelihood
                posteriors.append(posterior)
            
            y_pred.append(self.classes[np.argmax(posteriors)])
        
        return np.array(y_pred)

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

