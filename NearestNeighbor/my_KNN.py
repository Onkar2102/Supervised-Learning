import pandas as pd
import numpy as np
from collections import Counter

class my_KNN:

    def __init__(self, n_neighbors=5, metric="minkowski", p=2):
    # def __init__(self, n_neighbors=5, metric="euclidean"):
        # metric = {"minkowski", "euclidean", "manhattan", "cosine"}
        # p value only matters when metric = "minkowski"
        # notice that for "cosine", 1 is closest and -1 is furthest
        # therefore usually cosine_dist = 1- cosine(x,y)
        self.n_neighbors = int(n_neighbors)
        self.metric = metric
        # self.p = p
        self.p = p

    def fit(self, X, y):
        # X: pd.DataFrame, independent variables, float
        # y: list, np.array or pd.Series, dependent variables, int or str
        
        # we are trying to fit the KNN classifier to the training data
        
        # to store the train data
        self.X_train = X
        self.y_train = y
        
        # list of unique class labels
        self.classes_ = list(set(list(y)))
        
        return self
    
    def calculate_distance(self, x1, x2):
        # Calculate the distance between two data points based on the selected metric
        if self.metric == "minkowski":
            # implementating Minkowski distance formula with the specified power parameter (p)
            # https://en.wikipedia.org/wiki/Minkowski_distance -- for formula reference
            return np.power(np.sum(np.power(np.abs(x1 - x2), self.p)), 1 / self.p)
        elif self.metric == "euclidean":
            # implementing Euclidean distance formula
            return np.sqrt(np.sum(np.square(x1 - x2)))
        elif self.metric == "manhattan":
            # implementing Manhattan distance formula
            return np.sum(np.abs(x1 - x2))
        elif self.metric == "cosine":
            # implementing Cosine similarity (1 - cosine similarity as distance)
            dot_product = sum(x1_i * x2_i for x1_i, x2_i in zip(x1, x2))
            norm_x1 = sum(x1_i ** 2 for x1_i in x1) ** 0.5
            norm_x2 = sum(x2_i ** 2 for x2_i in x2) ** 0.5

            cosine_sim = dot_product / (norm_x1 * norm_x2)
            cosine_distance = 1 - cosine_sim
            return cosine_distance
        else:
            raise ValueError("Invalid distance metric")
            

    def predict(self, X):
        # X: pd.DataFrame, independent variables, float
        # return predictions: list
        # write your code below
        
        # here, we are trying to predict the class labels for a set of independent variables
        
        predictions = []

        for _, row in X.iterrows():
            # compute distances between the test point and all training points
            distances = []

            for _, train_row in self.X_train.iterrows():
                dist = self.calculate_distance(row, train_row)
                distances.append(dist)

            # getting indices of the k-nearest neighbors
            # refer google for the following implementation of code for indices, 
            # it was suggesting the use of argsort function which is implemented in easy steps after understanding it's functionality (code -1)
            k_nearest_indices = []
            for _ in range(self.n_neighbors):
                min_distance = float('inf')
                min_index = -1

                # for loop through distances to find the minimum
                for i, distance in enumerate(distances):
                    if distance < min_distance:
                        min_distance = distance
                        min_index = i

                k_nearest_indices.append(min_index)
                distances[min_index] = float('inf')  # setting minimum distance to infinity

            # getting corresponding labels of the k-nearest neighbors
            k_nearest_labels = [self.y_train.iloc[i] for i in k_nearest_indices]

            # finding most common class label among the k-nearest neighbors
            most_common = Counter(k_nearest_labels).most_common(1)
            predicted_label = most_common[0][0]

            predictions.append(predicted_label)

        return predictions


    def predict_proba(self, X):
        # X: pd.DataFrame, independent variables, float
        # prob is a dict of prediction probabilities belonging to each categories
        # return probs = pd.DataFrame(list of prob, columns = self.classes_)
        # write your code below

        # here, we are calculating the class probabilities for a set of independent variables
        
        probs = []

        for _, row in X.iterrows():
            # compute distances between the test point and all training points
            distances = []

            for _, train_row in self.X_train.iterrows():
                dist = self.calculate_distance(row, train_row)
                distances.append(dist)

            # getting indices of the k-nearest neighbors
            # refer google for the following implementation of code for indices, 
            # it was suggesting the use of argsort function which is implemented in easy steps after understanding it's functionality (code-2)
            k_nearest_indices = sorted(range(len(distances)), key=lambda i: distances[i])[:self.n_neighbors]

            # getting corresponding labels of the k-nearest neighbors
            k_nearest_labels = [self.y_train.iloc[i] for i in k_nearest_indices]

            # to compute class probabilities based on the k-nearest neighbors
            class_probs = {label: k_nearest_labels.count(label) / self.n_neighbors for label in self.classes_}
            probs.append(class_probs)

        # converting list of dictionaries into a DataFrame
        probs = pd.DataFrame(probs, columns=self.classes_)

        return probs



