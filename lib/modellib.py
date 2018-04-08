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
            metrics='auc', early_stopping_rounds=early_stopping_rounds, verbose_eval=False)
        model.set_params(n_estimators=cvresult.shape[0])
        del dtrain, cvresult
        gc.collect()

    # Fit the algorithm on the data
    # model.fit(train_df, label_df, eval_set=, eval_metric='auc', early_stopping_rounds=early_stopping_rounds)
    model.fit(train_df, label_df, eval_metric='auc')

    # Predict training set:
    train_predictions = model.predict(train_df)
    train_predprob = model.predict_proba(train_df)[:,1]
   

    # Print model report:
    print("\nModel Report")
    print("AUC Score (Train):", metrics.roc_auc_score(label_df.values, train_predprob))

    ## Plot importance
    plot_importance(model)
    pyplot.show()

    return model


def xgb_train(train_df, label_df, useTrainCV=True):
    # Choose all predictors except target & IDcols

    model = XGBClassifier(
                        learning_rate =0.3,
                        n_estimators=1000,
                        objective= 'binary:logistic',
                        nthread=8,
                        max_depth=5,
                        min_child_weight=0,
                        subsample=0.9,
                        colsample_bytree=0.7,
                        reg_alpha=4,
                        scale_pos_weight=0.9,
                        seed=99)

    model = modelfit(model, train_df, label_df, useTrainCV)

    return model

def xgb_predict(model,test_df):
    prediction = pd.DataFrame()
    # prediction['is_attributed'] = model.predict(dtest, ntree_limit=model.best_ntree_limit)
    prediction['is_attributed'] = model.predict(test_df)
    return prediction

# def xgb_train(train_df, label_df, validate=False, params=None):
#     if not params:
#         params = {'eta': 0.3,
#                   'tree_method': "auto",
#                   'grow_policy': "lossguide",
#                   'max_leaves': 1400,
#                   'max_depth': 0,
#                   'subsample': 0.9,
#                   'colsample_bytree': 0.7,
#                   'colsample_bylevel': 0.7,
#                   'min_child_weight': 0,
#                   'alpha': 4,
#                   'objective': 'binary:logistic',
#                   'scale_pos_weight': 9,
#                   'eval_metric': 'auc',
#                   'nthread': 8,
#                   'random_state': 99,
#                   'silent': True}
#     if (validate == True):
#         # Get 10% of train dataset to use as validation
#         x1, x2, y1, y2 = train_test_split(train_df, label_df, test_size=0.1, random_state=99)
#         dtrain = xgb.DMatrix(x1, y1)
#         dvalid = xgb.DMatrix(x2, y2)
#         del x1, y1, x2, y2
#         gc.collect()
#         watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
#         model = xgb.train(params, dtrain, 200, watchlist, maximize=True, early_stopping_rounds=25, verbose_eval=5)
#         del dvalid
#     else:
#         dtrain = xgb.DMatrix(train_df, label_df)
#         del train_df, label_df
#         gc.collect()
#         watchlist = [(dtrain, 'train')]
#         model = xgb.train(params, dtrain, 30, watchlist, maximize=True, verbose_eval=1)
#     return model


# def xgb_predict(model,test_df):
#     dtest = xgb.DMatrix(test_df)
#     del test_df
#     gc.collect()
#     prediction = pd.DataFrame()
#     prediction['is_attributed'] = model.predict(dtest, ntree_limit=model.best_ntree_limit)
#     return prediction
