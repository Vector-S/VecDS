import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier, plot_importance
from sklearn import cross_validation, metrics   
from sklearn.grid_search import GridSearchCV  
from matplotlib import pyplot
import gc




################################ XGBoost ############################################

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


def xgb_pt(s,cv_folds=5, early_stopping_rounds=50):
    xgb_param = s.model.get_xgb_params()
    dtrain = xgb.DMatrix(s.train_df.values, label=s.label_df.values)
    cvresult = xgb.cv(xgb_param, dtrain, num_boost_round=s.model.get_params()['n_estimators'], nfold=cv_folds,
                      metrics='auc', early_stopping_rounds=early_stopping_rounds, verbose_eval=True)
    s.model.set_params(n_estimators=cvresult.shape[0])
    del dtrain, cvresult
    gc.collect()



def xgb_train(s,eval_metric='auc'):
    s.model.fit(s.train_df, s.label_df, eval_metric='auc')
    s.train_result = s.model.predict_proba(s.train_df)[:,1]
    print("\nModel Report")
    print("AUC Score (Train):", metrics.roc_auc_score(s.label_df.values, s.train_result))

def xgb_save_fi(s):
    plot_importance(s.model)
    pyplot.savefig(s.ouput_path+ s.file_name)

def xgb_test(s,test_df):
    prediction = pd.DataFrame()
    prediction['is_attributed'] = s.model.predict(s.test_df)
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






