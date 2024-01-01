# Name - Onkar Shelar
# Email ID - os9660@rit.edu

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import SGDClassifier
# from sklearn.pipeline import make_pipeline
# from sklearn.model_selection import RandomizedSearchCV, train_test_split
# from sklearn.metrics import f1_score
# from sklearn.model_selection import GridSearchCV
from gensim.parsing.preprocessing import STOPWORDS
import string
import re

from sklearn.svm import SVC

# Custom preprocessing function
def preprocess(data_frame):

    # print(data_frame.columns)
    # Splitting the location into city, state, and country and handling missing values
    location_split = data_frame['location'].str.split(',', expand=True)
    data_frame['city'] = location_split[0].str.strip().fillna('Not specified')
    data_frame['state'] = location_split[1].str.strip().fillna('Not specified')
    data_frame['country'] = location_split[2].str.strip().fillna('Not specified')

    # Handling other missing values
    data_frame.fillna('Not specified', inplace=True)

    # Removing HTML tags from text columns
    for column in ['title', 'description', 'requirements']:
        data_frame[column] = data_frame[column].str.replace(r'<[^>]*>', '')

    # Frequency encoding for 'country', 'state', and 'city'
    for col in ['country', 'state', 'city']:
        freq = data_frame[col].value_counts() / len(data_frame)
        data_frame[f'{col}_freq'] = data_frame[col].map(freq)

    # Ensure binary features are numeric
    binary_features = ['telecommuting', 'has_company_logo', 'has_questions']
    for feature in binary_features:
        data_frame[feature] = data_frame[feature].astype(int)

    # Combining text columns 
    data_frame['combined_text'] = data_frame['title'] + ' ' + data_frame['description'] + ' ' + data_frame['requirements']
    
    # Normalize text: convert to lowercase
    data_frame['combined_text'] = data_frame['combined_text'].str.lower()

    # Removing removing non-word characters and stop words
    data_frame['combined_text'] = data_frame['combined_text'].apply(lambda x: re.sub(r'\W', ' ', x))
    data_frame['combined_text'] = data_frame['combined_text'].apply(lambda text: " ".join([word for word in text.lower().split() if word not in STOPWORDS]))
    
    # Creating text length features
    data_frame['desc_length'] = data_frame['description'].apply(len)
    data_frame['req_length'] = data_frame['requirements'].apply(len)
    
    # print(data_frame.columns)

    # Dropping original text and location columns
    data_frame.drop(['title', 'description', 'requirements', 'location', 'city', 'state', 'country', 'telecommuting'], axis=1, inplace=True)

    # print(data_frame.columns)
    # print(data_frame)
    
    return data_frame

# Custom model class
class my_model():
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        # Initialize your classifier with the best parameters found
        self.classifier = RandomForestClassifier(
            n_estimators=600,
            max_depth=30,        
            min_samples_split=11,  
            criterion='entropy',        
            class_weight='balanced',
            random_state=42
        )
        # self.classifier = RandomForestClassifier(class_weight='balanced', random_state=42)

    def fit(self, X, y):
        
        X_preprocessed = preprocess(X.copy())
        
        # Separating text data for TF-IDF transformation
        text_data = X_preprocessed.pop('combined_text')
        
        text_features = self.tfidf_vectorizer.fit_transform(text_data)
        
        # Combining text features with other features
        X_combined = np.hstack((text_features.toarray(), X_preprocessed.values))
        
        self.classifier.fit(X_combined, y)
        
        # # Define a broad range of parameters for RandomizedSearchCV
        # rf_random_params = {
        #     'n_estimators': np.arange(100, 1001, 100),
        #     'max_depth': np.arange(10, 101, 10),
        #     'min_samples_split': np.arange(2, 11, 1),
        #     'criterion': ['gini', 'entropy']
        # }
        
        # # Randomized Search with Cross-Validation
        # self.rfc = RandomForestClassifier(class_weight="balanced", random_state=42)
        # random_search = RandomizedSearchCV(self.rfc, rf_random_params, n_iter=100, cv=5, scoring='f1', n_jobs=-1, random_state=42)
        # random_search.fit(X_combined, y)
        # print("Best parameters from RandomizedSearch: ", random_search.best_params_)

        # # Refine search with GridSearchCV around the best parameters found
        # best_params = random_search.best_params_
        # rf_grid_params = {
        #     'n_estimators': [best_params['n_estimators'] - 50, best_params['n_estimators'], best_params['n_estimators'] + 50],
        #     'max_depth': [best_params['max_depth'] - 10, best_params['max_depth'], best_params['max_depth'] + 10],
        #     'min_samples_split': [best_params['min_samples_split'] - 1, best_params['min_samples_split'], best_params['min_samples_split'] + 1],
        #     'criterion': [best_params['criterion']]
        # }
        # self.rscv = GridSearchCV(self.rfc, rf_grid_params, cv=5, scoring='f1', n_jobs=-1)
        # self.rscv.fit(X_combined, y)
        # print("Refined best parameters from GridSearchCV: ", self.rscv.best_params_)

    def predict(self, X):
        
        X_preprocessed = preprocess(X.copy())

        # Separating text data for TF-IDF transformation
        text_data = X_preprocessed.pop('combined_text')
        text_features = self.tfidf_vectorizer.transform(text_data)

        # Combining text features with other features
        X_combined = np.hstack((text_features.toarray(), X_preprocessed.values))

        return self.classifier.predict(X_combined)
