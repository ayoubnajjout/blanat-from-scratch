import numpy as np

class DecisionTreeC45:
    def __init__(self, max_depth=None):
        self.tree = {}
        self.max_depth = max_depth

    def __entropy_class(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities))
    
    def __entropy_info(self,x, y):
        _, counts = np.unique(x, return_counts=True)
        probabilities = counts / len(x)
        entropy = 0
        for i in range(len(probabilities)):
            entropy += probabilities[i] * self.__entropy_class(y[x == i])
        return entropy
    
    def __information_gain(self, x, y):
        return self.__entropy_class(y) - self.__entropy_info(x, y)
    
    def __split_info(self, x):
        _, counts = np.unique(x, return_counts=True)
        probabilities = counts / len(x)
        return -np.sum(probabilities * np.log2(probabilities))
    
    def __gain_ratio(self, x, y):
        return self.__information_gain(x, y) / self.__split_info(x)
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.tree = self.__fit(X, y, depth=0)

    def __fit(self, X, y, depth):
        if len(np.unique(y)) == 1:
            return y[0]
        if X.shape[1] == 0 or (self.max_depth is not None and depth >= self.max_depth):
            return np.argmax(np.bincount(y))
        
        gains = np.array([self.__gain_ratio(X[:, i], y) for i in range(X.shape[1])])
        best_feature = np.argmax(gains)
        tree = {best_feature: {}}
        for value in np.unique(X[:, best_feature]):
            index = X[:, best_feature] == value
            tree[best_feature][value] = self.__fit(X[index], y[index], depth + 1)
        return tree
    
    def show_tree(self):
        print(self.tree)

    def predict(self, X):
        X = np.array(X)
        return np.array([self.__predict(x) for x in X])
    
    def __predict(self, x):
        tree = self.tree
        while isinstance(tree, dict):
            feature = list(tree.keys())[0]
            tree = tree[feature]
            value = x[feature]
            if value not in tree:
                return np.argmax(np.bincount(tree.values()))
            tree = tree[value]
        return tree
