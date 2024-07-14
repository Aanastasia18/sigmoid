#!/usr/bin/env python
# coding: utf-8

# In[19]:


# import libraries
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from sklearn.pipeline import Pipeline


# In[11]:


class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    
    def fit(self, X: pd.DataFrame, y = None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if all(column in X.columns for column in self.columns):
            return X.drop(self.columns, axis=1)
        else:
            raise ValueError("DataFrame you sent has no such column")


class MultiColumnImputer(BaseEstimator, TransformerMixin):
    def __init__(self, columns, imputer_name="simple_imputer"):
        self.columns = columns
        self.imputer_name = imputer_name
        self.imputer = None  

    def fit(self, X, y=None):
        if self.imputer_name == "simple_imputer":
            self.imputer = SimpleImputer(strategy="mean")
        elif self.imputer_name == "KNNI":
            n_neighbors = max(1, len(X) // 5)  # Ensure at least one neighbor
            self.imputer = KNNImputer(n_neighbors=n_neighbors)
        elif self.imputer_name == "MICE":
            self.imputer = IterativeImputer()

        self.imputer.fit(X[self.columns])
        return self
    
    def transform(self, X):
        X = X.copy()
        X[self.columns] = self.imputer.transform(X[self.columns])
        return X


class FeatureEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, encoding_name="one hot encoding", columns=None, categories=None):
        self.encoding_name = encoding_name
        self.columns = columns
        self.categories = categories
        self.encoders = {}
    
    def fit(self, X, y=None):
        if self.columns is None:
            raise ValueError("Columns must be specified.")
        
        if self.encoding_name == "one hot encoding":
            for col in self.columns:
                encoder = OneHotEncoder(sparse=False, drop='first', handle_unknown='ignore')
                encoder.fit(X[[col]])
                self.encoders[col] = encoder
        elif self.encoding_name == "ordinary encoding":
            if self.categories is None or not isinstance(self.categories, list) or len(self.categories) != len(self.columns):
                raise ValueError("Categories must be a list of lists and match the length of columns for 'ordinary encoding'.")
            for col, cat in zip(self.columns, self.categories):
                encoder = OrdinalEncoder(categories=[cat], handle_unknown='use_encoded_value', unknown_value=-1)
                encoder.fit(X[[col]])
                self.encoders[col] = encoder
        else:
            raise ValueError(f"Unsupported encoding_name: {self.encoding_name}")
        
        return self
    
    def transform(self, X):
        if not self.encoders:
            raise ValueError("FeatureEncoder has not been fitted yet.")
        
        X_transformed = X.copy()
        
        for col in self.columns:
            encoder = self.encoders[col]
            encoded_values = encoder.transform(X[[col]])
            if self.encoding_name == "one hot encoding":
                feature_names = encoder.get_feature_names_out([col])
                encoded_df = pd.DataFrame(encoded_values, columns=feature_names, index=X.index)
                encoded_df = encoded_df.loc[:, encoded_df.columns != f"{col}_nan"]
                X_transformed = pd.concat([X_transformed.drop(columns=[col]), encoded_df], axis=1)
            elif self.encoding_name == "ordinary encoding":
                # Handle NaNs explicitly: replace -1 with NaN after transformation
                encoded_values[encoded_values == -1] = np.nan
                X_transformed[col] = encoded_values.flatten()
        
        return X_transformed




