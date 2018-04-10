import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier, plot_importance
from sklearn import cross_validation, metrics   
from sklearn.grid_search import GridSearchCV  
from matplotlib import pyplot
import gc



################################################# XGBoost #################################################

def xgb_init(s):
    s.model =  XGBClassifier(learning_rate=0.3,
                          n_estimators=1000,
                          objective='binary:logistic',
                          nthread=8,
                          max_depth=5,
                          min_child_weight=0,
                          subsample=0.9,
                          colsample_bytree=0.7,
                          reg_alpha=4,
                          scale_pos_weight=0.9,
                          seed=99)


def xgb_pt(s):
    # to be implemented
        pass


def modelfit(model, train_df, label_df, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = model.get_xgb_params()
        dtrain = xgb.DMatrix(train_df.values, label=label_df.values)
        cvresult = xgb.cv(xgb_param, dtrain, num_boost_round=model.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds, verbose_eval=True)
        model.set_params(n_estimators=cvresult.shape[0])
        del dtrain, cvresult
        gc.collect()
    # Fit the algorithm on the data
    model.fit(train_df, label_df, eval_metric='auc')

def xgb_train(s,eval_metric='auc'):
    modelfit(s.model,s.train_df,s.label_df)


    s.train_result = s.model.predict_proba(s.train_df)[:,1]
    print("\nModel Report")
    print("AUC Score (Train):", metrics.roc_auc_score(s.label_df.values, s.train_result))


def xgb_save_fi(s):
    plot_importance(s.model)
    pyplot.savefig(s.output_path+ 'feature_importance.png')

def xgb_test(s):
    prediction = pd.DataFrame()
    prediction['is_attributed'] = s.model.predict_proba(s.test_df)[:,1]
    s.test_result = prediction


################################ LightGBM ############################################

def lgbm_init(s):
    pass

def lgbm_pt(s,cv_folds=5, early_stopping_rounds=50):
    pass

def lgbm_train(s,eval_metric='auc') :
    pass

def lgbm_save_fi(s):
    pass

def lgbm_test(s,test_df):
    pass




