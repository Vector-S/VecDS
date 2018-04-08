import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier, plot_importance
from sklearn import cross_validation, metrics   
from sklearn.grid_search import GridSearchCV  
from matplotlib import pyplot
import gc


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

    # Predict training set:
    train_predictions = model.predict(train_df)
    train_predprob = model.predict_proba(train_df)[:,1]
   

    # Print model report:
    print("\nModel Report")
    print("Accuracy:", metrics.accuracy_score(label_df.values, train_predictions))
    print("AUC Score (Train):", metrics.roc_auc_score(label_df.values, train_predprob))


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

    model = XGBClassifier(
                        max_depth=5,
                        learning_rate=0.1,
                        n_estimators=1000,
                        objective= 'binary:logistic',
                        n_jobs=4,
                        gamma=0.1,
                        min_child_weight=1,
                        subsample=0.8, 
                        colsample_bytree=0.8, 
                        scale_pos_weight=1, 
                        random_state=24)


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
    prediction['is_attributed'] = model.predict_proba(test_df)[:,1]
    return prediction



def opt_model1(train_df, label_df):
    param_test1 = {'max_depth':list(range(3,10,2)),
                   'min_child_weight':list(range(1,6,2))
                   }
    gsearch1 = GridSearchCV(
        estimator = XGBClassifier(
                                max_depth=5,
                                learning_rate=0.1,
                                n_estimators=1000,
                                objective= 'binary:logistic',
                                n_jobs=4,
                                gamma=0.1,
                                min_child_weight=1,
                                subsample=0.8, 
                                colsample_bytree=0.8, 
                                scale_pos_weight=1, 
                                random_state=24),
        param_grid = param_test1,     
        scoring='roc_auc',
        n_jobs=4,
        iid=False, 
        cv=5)

    gsearch1.fit(train_df, label_df)
    print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)     
