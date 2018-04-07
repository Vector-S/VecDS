import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   
from sklearn.grid_search import GridSearchCV  

import matplotlib.pylab as plt
# %matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4


def modelfit(model, Xtrain, ytrain, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = model.get_xgb_params()
        xgtrain = xgb.DMatrix(Xtrain.values, label=ytrain.values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=model.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds, show_progress=False)
        model.set_params(n_estimators=cvresult.shape[0])

    #Fit the algorithm on the data
    model.fit(Xtrain, ytrain, eval_metric='auc')

    #Predict training set:
    train_predictions = model.predict(Xtrain)
    train_predprob = model.predict_proba(Xtrain)[:,1]

    #Print model report:
    print("\nModel Report")
    print("Accuracy :", metrics.accuracy_score(ytrain.values, train_predictions))
    print("AUC Score (Train):", metrics.roc_auc_score(ytrain.values, train_predprob))

    feat_imp = pd.Series(model.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')

    return model


def xgb_train(train_df, label_df, useTrainCV=True):
    #Choose all predictors except target & IDcols

    model = XGBClassifier(
                        learning_rate =0.1,
                        n_estimators=1000,
                        max_depth=5,
                        min_child_weight=1,
                        gamma=0,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective= 'binary:logistic',
                        nthread=4,
                        scale_pos_weight=1,
                        seed=27)
    modelfit(model, train_df, label_df)

    return model








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


def xgb_predict(model,test_df):
    dtest = xgb.DMatrix(test_df)
    del test_df
    gc.collect()
    prediction = pd.DataFrame()
    prediction['is_attributed'] = model.predict(dtest, ntree_limit=model.best_ntree_limit)
    return prediction
