from datetime import datetime
from typing import List, Dict, Optional, NamedTuple

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import precision_recall_curve

USER_ID_COL = 'ID'
DATE_COL = 'ActiveDate'
TARGET_COL = 'target'
RANDOM_STATE = 0

MODEL_PARAMS = {
    'max_depth': 5,
    'objective': 'binary',
    'metric': 'binary_logloss',
    'min_data_in_leaf': 30,
    'learning_rate': 0.02,
    'num_leaves': 50,
    'feature_fraction': 0.6,
    'bagging_fraction': 0.65,
    'num_boost_round': 500,
    'random_state': RANDOM_STATE
}

FIT_PARAMS = {
    'early_stopping_rounds':35,
    'verbose': 50
}



# customize data objects
class SplitedData(NamedTuple):
    train: Dict[str, pd.DataFrame]
    test: Dict[str, pd.DataFrame]
    validation: Dict[str, pd.DataFrame]
        

class ModelArtifacts(NamedTuple):
    estimator: lgb.LGBMClassifier
    splited_data: SplitedData
    train_cols: List[str]
    optimal_threshold: float
        
        
def data_spliting(full_data: pd.DataFrame, train_ratio_size: float = 0.80, 
                     test_ratio_size: float = 0.10):
    """
    """
    accumlate_percetage = (full_data.groupby(DATE_COL)[USER_ID_COL].count().cumsum()/full_data.index.size).rename('accumlate_percetage')
    
    max_train_date = accumlate_percetage[accumlate_percetage<=train_ratio_size].idxmax()
    max_test_date = accumlate_percetage[accumlate_percetage<=train_ratio_size+test_ratio_size].idxmax()
    
    train = full_data[full_data[DATE_COL] <= max_train_date]
    test = full_data[(full_data[DATE_COL] > max_train_date) & (full_data[DATE_COL] <= max_test_date)]
    validation = full_data[full_data[DATE_COL] > max_test_date]
    
    train_cols = train.columns.difference([TARGET_COL])
    
    return SplitedData(
        train = {'X': train[train_cols], 'y': train[TARGET_COL]},
        test = {'X': test[train_cols], 'y': test[TARGET_COL]},
        validation = {'X': validation[train_cols], 'y': validation[TARGET_COL]}
    )
    
    
def train_model(splited_data: SplitedData, train_cols: List[str],
                train_weigts_vector: Optional[List[float]] = None):
    """
    """
    # gen calssifier and train
    estimator = lgb.LGBMClassifier(**MODEL_PARAMS)
    
    # set fit parametrs
    FIT_PARAMS['eval_set'] = [[splited_data.test['X'][train_cols], splited_data.test['y']]]
    FIT_PARAMS['eval_metric']= ['logloss']
    
    if isinstance(train_weigts_vector, list):
        FIT_PARAMS['sample_weight'] = train_weigts_vector
        
    estimator.fit(splited_data.train['X'][train_cols], splited_data.train['y'], **FIT_PARAMS)
    
    return estimator


def extract_optimal_threshold(estimator: lgb.LGBMClassifier, splited_data: SplitedData,
                             train_cols: List[str]):
    """
    """
    y_test_proba = estimator.predict_proba(splited_data.test['X'][train_cols])[:,1]
    precision, recall, thresholds = precision_recall_curve(splited_data.test['y'], y_test_proba)
    numerator = 2 * recall * precision
    denom = recall + precision
    f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom!=0))
    optimal_threshold = thresholds[np.argmax(f1_scores)]    
    
    return optimal_threshold


def model_exe(full_data: pd.DataFrame, train_cols: List[str], 
              positve_weight: Optional[float] = None) -> ModelArtifacts:
    """
    """
    splited_data = data_spliting(full_data)
    if positve_weight:
        train_weigts_vector = list(np.where(splited_data.train['y']==1, positve_weight, 1))
    else:
        train_weigts_vector = None
        
    estimator = train_model(splited_data, train_cols, train_weigts_vector=train_weigts_vector) 
    
    optimal_threshold = extract_optimal_threshold(estimator, splited_data, train_cols)
    
    return ModelArtifacts(
        estimator=estimator, 
        splited_data=splited_data,
        train_cols=train_cols,
        optimal_threshold=optimal_threshold
    )
