import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier, plot_importance
from sklearn import cross_validation, metrics   
from sklearn.grid_search import GridSearchCV  
from matplotlib import pyplot
import gc

def xgb_pt(s):
    """
    to be implemented
    :param s:
    :return:
    """
    pass

def xgb_pt1(s):
    param_test1 = {
                   'max_depth':[1,2,3,4,5,6],
                   'min_child_weight':[0.5,0.7,1,1.2]
                   }

    gsearch1 = GridSearchCV(
        estimator = XGBClassifier(learning_rate=0.3,
                          n_estimators=1000,
                          objective='binary:logistic',
                          nthread=8,
                          max_depth=5,
                          min_child_weight=0,
                          subsample=0.9,
                          colsample_bytree=0.7,
                          reg_alpha=4,
                          scale_pos_weight=0.9,
                          seed=99), 
                          param_grid = param_test1,     
                          scoring='roc_auc', 
                          n_jobs=4,
                          iid=False, 
                          cv=5)

    gsearch1.fit(s.train_df,s.label_df)
    print('grid_score:', gsearch1.grid_scores_)
    print('best_params:', gsearch1.best_params_)  
    print('best_score:', gsearch1.best_score_)

def xgb_pt2(s):
    param_test2 = {
                   'gamma':[i/10.0 for i in range(0,5)]
                   }

    gsearch2 = GridSearchCV(
        estimator = XGBClassifier(learning_rate=0.3,
                          n_estimators=1000,
                          objective='binary:logistic',
                          nthread=8,
                          max_depth=5,
                          min_child_weight=0,
                          subsample=0.9,
                          colsample_bytree=0.7,
                          reg_alpha=4,
                          scale_pos_weight=0.9,
                          seed=99), 
                          param_grid = param_test2,     
                          scoring='roc_auc', 
                          n_jobs=4,
                          iid=False, 
                          cv=5)

    gsearch2.fit(s.train_df,s.label_df)
    print('grid_score:', gsearch2.grid_scores_)
    print('best_params:', gsearch2.best_params_)  
    print('best_score:', gsearch2.best_score_)

def xgb_pt3(s):
    param_test3 = {
                   'subsample':[i/10.0 for i in range(6,10)],
                   'colsample_bytree':[i/10.0 for i in range(6,10)]
                   }

    gsearch3 = GridSearchCV(
        estimator = XGBClassifier(learning_rate=0.3,
                          n_estimators=1000,
                          objective='binary:logistic',
                          nthread=8,
                          max_depth=5,
                          min_child_weight=0,
                          subsample=0.9,
                          colsample_bytree=0.7,
                          reg_alpha=4,
                          scale_pos_weight=0.9,
                          seed=99), 
                          param_grid = param_test3,     
                          scoring='roc_auc', 
                          n_jobs=4,
                          iid=False, 
                          cv=5)

    gsearch3.fit(s.train_df,s.label_df)
    print('grid_score:', gsearch3.grid_scores_)
    print('best_params:', gsearch3.best_params_)  
    print('best_score:', gsearch3.best_score_)

def xgb_pt4(s):
    param_test4 = {
                   'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
                  }

    gsearch4 = GridSearchCV(
        estimator = XGBClassifier(learning_rate=0.3,
                          n_estimators=1000,
                          objective='binary:logistic',
                          nthread=8,
                          max_depth=5,
                          min_child_weight=0,
                          subsample=0.9,
                          colsample_bytree=0.7,
                          reg_alpha=4,
                          scale_pos_weight=0.9,
                          seed=99), 
                          param_grid = param_test4,     
                          scoring='roc_auc', 
                          n_jobs=4,
                          iid=False, 
                          cv=5)

    gsearch4.fit(s.train_df,s.label_df)
    print('grid_score:', gsearch4.grid_scores_)
    print('best_params:', gsearch4.best_params_)  
    print('best_score:', gsearch4.best_score_)


