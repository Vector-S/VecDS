import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import train_test_split
import gc


def xgb_train(train_df,label_df,is_valid=False):
    params = {'eta': 0.3,
              'tree_method': "hist",
              'grow_policy': "lossguide",
              'max_leaves': 1400,
              'max_depth': 0,
              'subsample': 0.9,
              'colsample_bytree': 0.7,
              'colsample_bylevel': 0.7,
              'min_child_weight': 0,
              'alpha': 4,
              'objective': 'binary:logistic',
              'scale_pos_weight': 9,
              'eval_metric': 'auc',
              'nthread': 8,
              'random_state': 99,
              'silent': True}

    if (is_valid == True):
        # Get 10% of train dataset to use as validation
        x1, x2, y1, y2 = train_test_split(train_df, label_df, test_size=0.1, random_state=99)
        dtrain = xgb.DMatrix(x1, y1)
        dvalid = xgb.DMatrix(x2, y2)
        del x1, y1, x2, y2
        gc.collect()
        watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
        model = xgb.train(params, dtrain, 200, watchlist, maximize=True, early_stopping_rounds=25, verbose_eval=5)
        del dvalid
    else:
        dtrain = xgb.DMatrix(train_df, label_df)
        del train_df, label_df
        gc.collect()
        watchlist = [(dtrain, 'train')]
        model = xgb.train(params, dtrain, 30, watchlist, maximize=True, verbose_eval=1)
    return model


def xgb_predict(model,test_df):
    dtest = xgb.DMatrix(test_df)
    del test_df
    gc.collect()
    prediction = pd.DataFrame
    prediction['is_attributed'] = model.predict(dtest, ntree_limit=model.best_ntree_limit)
    return prediction
