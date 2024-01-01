import pandas as pd
import numpy as np
from collections import Counter

class my_NB:

    def __init__(self, alpha=1):
        # Initialize the Naive Bayes classifier with a smoothing factor (Laplace smoothing)
        self.alpha = alpha

    def fit(self, X, y):
        # Fit the Naive Bayes classifier to the training data.
        # X: pd.DataFrame, independent variables, each value is a category of str type
        # y: list, np.array or pd.Series, dependent variables, each value is a category of int or str type

        # list of unique classes in the dependent variable.
        self.unique_classes = list(set(list(y)))

        # Calculate P(yj) and P(xi|yj) with Laplace smoothing
        self.class_probabilities = Counter(y)
        self.feature_probabilities = {}

        for label in self.unique_classes:
            self.feature_probabilities[label] = {}
            # Loop for feature in independent variables
            for feature_name in X.columns:
                self.feature_probabilities[label][feature_name] = {}
                unique_feature_values = X[feature_name].unique()
                num_unique_values = len(unique_feature_values)

                # for loop for unique value of the feature
                for value in unique_feature_values:
                    # count occurrences of the value in the feature for the current class
                    count_xi = np.sum((X[feature_name] == value) & (y == label))
                    # calculating the conditional probability P(xi|yj) with Laplace smoothing
                    self.feature_probabilities[label][feature_name][value] = (count_xi + self.alpha) / (np.sum(y == label) + num_unique_values * self.alpha)

        return self

    def predict(self, X):
        # Predict the class labels for a set of independent variables.
        # X: pd.DataFrame, independent variables, each value is a category of str type
        # return predictions: list
        probabilities = self.predict_proba(X)
        predictions = []

        # for loop on row in the probability DataFrame
        for _, row in probabilities.iterrows(): # used google searched for iterrows()
            # finding the class label with the maximum probability for each row
            max_probability_class = row.idxmax()   # used google searched for this idxmax() function - link referred https://www.geeksforgeeks.org/python-pandas-dataframe-idxmax/  
            predictions.append(max_probability_class)

        return predictions
    
    def predict_proba(self, X):
        # Calculate the class probabilities and predict probabilities for a set of independent variables.
        # X: pd.DataFrame, independent variables, each value is a category of str type
        # prob is a dict of prediction probabilities belonging to each category
        # return probs = pd.DataFrame(list of prob, columns = self.unique_classes)

        # storing prediction probabilities
        probabilities_dict = {label: [] for label in self.unique_classes}

        # for over each row in the independent variables DataFrame
        for _, row in X.iterrows():
            for label in self.unique_classes:
                probability = np.log(self.class_probabilities[label])
                for feature_name in X.columns:
                    value = row[feature_name]
                    if value in self.feature_probabilities[label][feature_name]:
                        # calculating the log probability for each feature and sum them up
                        probability += np.log(self.feature_probabilities[label][feature_name][value])
                probabilities_dict[label].append(probability)

        # converting the log probabilities to probabilities and normalizing them
        probabilities_df = pd.DataFrame(probabilities_dict, columns=self.unique_classes)
        probabilities_df = np.exp(probabilities_df)
        probabilities_df = probabilities_df.div(probabilities_df.sum(axis=1), axis=0)

        return probabilities_df
