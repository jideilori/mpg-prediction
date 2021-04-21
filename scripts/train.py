import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.compose import ColumnTransformer
import joblib
        

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


def load_data(file_path):
    cols = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                    'Acceleration', 'Model Year', 'Origin']
    # reading the .data file using pandas
    df = pd.read_csv(f'{file_path}', names=cols, na_values = "?",
                    comment = '\t',
                    sep= " ",
                    skipinitialspace=True)
    return df


def strat_split(df,target,test_size,seed):
    if not isinstance(target, str):
        raise TypeError("Please provide a string")
    split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    for train_index, test_index in split.split(df, df[f"{target}"]):
        strat_train_set = df.loc[train_index]
        strat_test_set = df.loc[test_index]
    return strat_train_set,strat_test_set

def get_feat_labels(df,target):
    features = df.drop(f'{target}',axis=1)
    labels = df[f'{target}'].copy()
    return features,labels


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

def preprocess_origin_cols(df):
    if df['Origin'].nunique() > 3:
        raise AssertionError('Value more than 3')
        return
    df["Origin"] = df["Origin"].map({1: "India", 2: "USA", 3: "Germany"})    
    return df

save_model = True
if __name__ == '__main__':

    data =  load_data('./data/raw/auto-mpg.data')

    train_df,test_df = strat_split(data,'Cylinders',0.2,11)

    features,labels = get_feat_labels(train_df,'MPG')

    acc_ix=features.columns.get_loc('Acceleration')
    hpower_ix=features.columns.get_loc('Horsepower')
    cyl_ix=features.columns.get_loc('Cylinders')

    preprocessed_df = preprocess_origin_cols(features)
    prepared_data = pipeline_transformer(preprocessed_df)

    forest_reg = RandomForestRegressor()
    scores = cross_val_score(forest_reg, 
                            prepared_data, 
                            labels, 
                            scoring="neg_mean_squared_error", 
                            cv = 10)

    forest_reg_rmse_scores = np.sqrt(-scores)
    display_scores(forest_reg_rmse_scores)

    param_grid = [
        {'n_estimators': [3, 10, 30], 
        'max_features': [2, 4, 6, 8,10],
        'max_depth': [4,8,10],
        'random_state':[11,42],
        'bootstrap': [False], 
        'n_estimators': [3, 10],
        'max_features': [2, 3, 4]},
    ]

    forest_reg = RandomForestRegressor()
    grid_search = GridSearchCV(forest_reg, param_grid,
                            scoring='neg_mean_squared_error',
                            return_train_score=True,
                            cv=10,
                            )

    grid_search.fit(prepared_data, labels)
    print(grid_search.best_params_)

    cvres = grid_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)


    final_model = grid_search.best_estimator_

    X_test = test_df.drop("MPG", axis=1)
    y_test = test_df["MPG"].copy()

    X_test_preprocessed = preprocess_origin_cols(X_test)
    X_test_prepared = pipeline_transformer(X_test_preprocessed)

    final_predictions = final_model.predict(X_test_prepared)
    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)

    print('RMSE\n',final_rmse)

    if save_model:
        joblib.dump(final_model, './model_checkpoints/rand_model.pkl')
        print('model saved')