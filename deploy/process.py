import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np

            
def num_pipeline_transformer(data):
    '''
    Function to process numerical transformations
    Argument:
        data: original dataframe 
    Returns:
        num_attrs: numerical dataframe
        num_pipeline: numerical pipeline object
        
    '''
    numerics = ['float64', 'int64']

    num_attrs = data.select_dtypes(include=numerics)

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attrs_adder', CustomAttrAdder()),
        ('std_scaler', StandardScaler()),
        ])
    return num_attrs, num_pipeline

def pipeline_transformer(data):
    '''
    Complete transformation pipeline for both
    nuerical and categorical data.
    
    Argument:
        data: original dataframe 
    Returns:
        prepared_data: transformed data, ready to use
    '''
    cat_attrs = ["Origin"]
    num_attrs, num_pipeline = num_pipeline_transformer(data)
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, list(num_attrs)),
        ("cat", OneHotEncoder(), cat_attrs),
        ])
    prepared_data = full_pipeline.fit_transform(data)
    return prepared_data

def preprocess_origin_cols(df):
    if df['Origin'].nunique() > 3:
        raise AssertionError('Value more than 3')
        return
    df["Origin"] = df["Origin"].map({1: "India", 2: "USA", 3: "Germany"})    
    return df

def get_idx(df):
    global acc_ix,hpower_ix,cyl_ix
    acc_ix=df.columns.get_loc('Acceleration')
    hpower_ix=df.columns.get_loc('Horsepower')
    cyl_ix=df.columns.get_loc('Cylinders')
    return acc_ix,hpower_ix,cyl_ix

def dict_to_df(dict_data):
    df = pd.DataFrame(dict_data)
    print(df.head())
    get_idx(df)
    df = preprocess_origin_cols(df)
    return df

class CustomAttrAdder(BaseEstimator, TransformerMixin):
    def __init__(self, acc_on_power=True):
        self.acc_on_power = acc_on_power  

    def fit(self, x_data, y=None):
        return self  
    
    def transform(self, x_data):
        acc_on_cyl = x_data[:, acc_ix] / x_data[:, cyl_ix] 
        if self.acc_on_power:
            acc_on_power = x_data[:, acc_ix] / x_data[:, hpower_ix]    
            return np.c_[x_data, acc_on_power, acc_on_cyl] 
        else:
            return np.c_[x_data,acc_on_cyl]


