import numpy as np

class Node:
    def __init__(self, data):
        self.data = data
        self.children = []
        self.parent = None

    def add_child(self, child):
        child.parent = self
        self.children.append(child)

class DecisionTree:
    def __init__(self):
        pass

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.root = self.__build_tree(X, y)

    def __build_tree(self, X, y):
        if len(np.unique(y)) == 1:
            return Node(data={"label": y[0]})

        chi_index, chi_values = self.__chi_square(X, y)
        best_feature_idx = chi_index[np.argmax(chi_values)]
        best_feature_name = best_feature_idx

        node = Node(data={"feature": best_feature_name})
        unique_values = np.unique(X[:, best_feature_idx])

        for value in unique_values:
            sub_X = X[X[:, best_feature_idx] == value]
            sub_y = y[X[:, best_feature_idx] == value]
            child = self.__build_tree(sub_X, sub_y)
            child.data["condition"] = f"{best_feature_name} = {value}"
            node.add_child(child)

        return node

    def __chi_square(self,X, y):
        X = np.array(X)
        y = np.array(y)
        chi_outputs = np.empty((X.shape[1]), dtype=float)
        chi_index = np.empty((X.shape[1]), dtype=object)
        for i in range(X.shape[1]):
            y_features = np.unique(y)
            x_features = np.unique(X[:, i])
            chi_square_value = 0
            chi_index[i] = i 
            for j in x_features:
                count_0 = np.sum((X[:, i] == j) & (y == y_features[0]))
                count_1 = np.sum((X[:, i] == j) & (y == y_features[1]))
                total = count_0 + count_1
                e = total / 2
                chi_square_value += ((e - count_0) ** 2 / e) ** 0.5
            chi_outputs[i] = chi_square_value
        return chi_index, chi_outputs


    # prediction function for later :3