import numpy as np
from scipy.linalg import svd
from copy import deepcopy
from collections import Counter
from pdb import set_trace
import pandas as pd

class my_normalizer:
     
    def __init__(self, norm="Min-Max", axis = 1):
        # set the normalization method and axis of operation
        #     norm = {"L1", "L2", "Min-Max", "Standard_Score"}
        #     axis = 0: normalize rows
        #     axis = 1: normalize columns
        self.norm = norm    
        self.axis = axis      

    def fit(self, X):
        #     X: input matrix
        #     Calculate offsets and scalers which are used in transform()
        X_array  = np.asarray(X)
        m, n = X_array.shape
        self.offsets = []   # List to store offsets
        self.scalers = []   # List to store scalers
        
        if self.axis == 1:
            # Normalize columns
            for col in range(n):
                offset, scaler = self.vector_norm(X_array[:, col])
                self.offsets.append(offset)
                self.scalers.append(scaler)
                
        elif self.axis == 0:
            # Normalize rows
            for row in range(m):
                offset, scaler = self.vector_norm(X_array[row])
                self.offsets.append(offset)
                self.scalers.append(scaler)
        else:
            raise Exception("Unknown axis.")

    def transform(self, X):
        # apply normalization to the input matrix X
        X_norm = deepcopy(np.asarray(X))
        m, n = X_norm.shape
        
        if self.axis == 1:
            # Normalize columns
            for col in range(n):
                X_norm[:, col] = (X_norm[:, col]-self.offsets[col])/self.scalers[col]
        
        elif self.axis == 0:
            # Normalize rows
            for row in range(m):
                X_norm[row] = (X_norm[row]-self.offsets[row])/self.scalers[row]
        else:
            raise Exception("Unknown axis.")
        return X_norm

    def fit_transform(self, X):
        # fit the normalization parameters and apply normalization
        self.fit(X)
        return self.transform(X)

    def vector_norm(self, x):
        # Calculate the offset and scaler for input vector x
        if self.norm == "Min-Max":
            # Write your own code below
            # Min-Max normalization
            offset = np.min(x)
            scaler = np.max(x) - np.min(x)

        elif self.norm == "L1":
            # Write your own code below
            # L1 normalization
            offset = 0
            scaler = np.sum(np.abs(x))

        elif self.norm == "L2":
            # Write your own code below
            # L2 normalization
            offset = 0
            scaler = np.sqrt(np.sum(x**2))

        elif self.norm == "Standard_Score":
            # Write your own code below
            # Standard Score normalization
            offset = np.mean(x)
            scaler = np.std(x)

        else:
            raise Exception("Unknown normlization.")
        return offset, scaler

class my_pca:
    def __init__(self, n_components = 5):
        # initializing the PCA model with a number of principal component to keep
        #     n_components: number of principal components to keep
        self.n_components = n_components
        self.principal_components = None    # Initialize the principal component

    def fit(self, X):
        #  Use svd to perform PCA on X
        #  Inputs:
        #     X: input matrix
        #  Calculates:
        #     self.principal_components: the top n_components principal_components
        U, s, Vh = svd(X)
        # Write your own code below
        # Store the top n_components principal components
        self.principal_components = Vh[:self.n_components]

    def transform(self, X):
        # Transform the input data X into a PCA space
        #     X_pca = X.dot(self.principal_components)
        X_array = np.asarray(X)
        return X_array.dot(self.principal_components.T)

    def fit_transform(self, X):
        # fit the PCA model to the input data and then transform it
        self.fit(X)
        return self.transform(X)

def stratified_sampling(y, ratio, replace=True):
    #  Inputs:
    #     y: class labels
    #     0 < ratio < 1: len(sample) = len(y) * ratio
    #     replace = True: sample with replacement
    #     replace = False: sample without replacement
    #  Output:
    #     sample: indices of stratified sampled points
    #             (ratio is the same across each class,
    #             samples for each class = int(np.ceil(ratio * # data in each class)) )
    if ratio <= 0 or ratio >= 1:
        raise Exception("ratio must be 0 < ratio < 1.")

    # Create a DataFrame from the input 'y' to work with pandas
    df = pd.DataFrame({'y': y})
    grouped = df.groupby('y')
    sample_indices = []

    for label, group in grouped:
        # Calculate the number of samples to select from this group
        num_samples = int(np.ceil(ratio * len(group)))

        # Randomly select samples with or without replacement
        if replace:
            sampled_indices = np.random.choice(group.index, size=num_samples, replace=True)
        else:
            sampled_indices = np.random.choice(group.index, size=num_samples, replace=False)

        # Extend the list of sampled indices with the indices from this group
        sample_indices.extend(sampled_indices)

    # Convert the list of sampled indices to a NumPy array of integers and return it
    return np.array(sample_indices).astype(int)
 