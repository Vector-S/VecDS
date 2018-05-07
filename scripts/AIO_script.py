import time
import pickle
import json
import os
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier, plot_importance
from sklearn import cross_validation, metrics
from matplotlib import pyplot
import gc

####################################### Utils ########################################

def report(msg,tic=None,print_out=True):
    length = 100
    report = "\n"+">>>>  "+ msg + " "
    if tic:
        report += '-' * (100 - len(report))
        report +=" TC:{0:.4g}".format(time.time()-tic)
    report+="\n"
    if print_out:
        print(report)
    if not tic:
        return time.time()


def load_obj(filename):
    try:
        with open(filename,'rb') as f:
            obj =  pickle.load(f)
            return obj
    except Exception as e:
        print("Can't load obj:{0}".format(str(e)))

def save_obj(obj,filename):
    try:
        with open(filename,'wb') as f:
            pickle.dump(obj,f,pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print("Can't save obj:{0}".format(str(e)))


def load_json(filename):
    try:
        with open(filename,'r') as f:
            dic =  json.load(f)
        return dic
    except Exception as e:
        print("Can't load dict:{0}".format(str(e)))

def save_json(dic,filename):
    try:
        with open(filename,'w') as f:
            json.dump(dic,f)
    except Exception as e:
        print("Can't save dict:{0}".format(str(e)))


def check_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

class Report:
    def __init__(self):
        total_time_cost = 0

####################################### DataSet #######################################

class DataSet:
    train_file = 'train.csv'
    test_file = 'test.csv'
    label_name = 'is_attributed'
    train_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
    test_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id']
    skip_train_rows = None
    num_train_rows = None
    num_test_rows = None
    dtypes = {
        'ip': 'uint32',
        'app': 'uint16',
        'device': 'uint16',
        'os': 'uint16',
        'channel': 'uint16',
        'is_attributed': 'uint8',
        'click_id': 'uint32'
    }

    def __init__(self):
        pass

    def load_train(self, s):
        s.train_df = pd.read_csv(s.input_path + self.train_file, skiprows=self.skip_train_rows,
                                 nrows=self.num_train_rows, dtype=self.dtypes,
                                 usecols=self.train_cols)
        s.label_df = s.train_df[self.label_name]

    def load_test(self, s):
        s.test_df = pd.read_csv(s.input_path + self.test_file, dtype=self.dtypes, nrows=self.num_test_rows,
                                usecols=self.test_cols)


####################################### Feature #######################################
def build_features(df, feature_pipeline):
    feature_set=set()
    for fun in feature_pipeline:
        df, feature_set=fun(df, feature_set)
    df = df[list(feature_set)]
    gc.collect()
    return df, feature_set


def f_template(df, gb_dict):
    all_features = list(set(gb_dict['groupby'] + [gb_dict['select']]))
    ## name of new feature
    if gb_dict['agg']=='count':
        f_name = '{}_{}'.format('_'.join(gb_dict['groupby']), gb_dict['agg'])
    else:
        f_name = '{}_{}_{}'.format('_'.join(gb_dict['groupby']), gb_dict['agg'], gb_dict['select'])
    ## perfom the grouby
    gp = df[all_features]. \
        groupby(gb_dict['groupby'])[gb_dict['select']]. \
        agg(gb_dict['agg']). \
        reset_index(). \
        rename(index=str, columns={gb_dict['select']: f_name}).astype(gb_dict['type'])
    # Merge back to df
    df = df.merge(gp, on=gb_dict['groupby'], how='left')
    del gp
    gc.collect()
    return df,f_name

def f_base(df,fs):
    fs= fs | {'ip', 'app', 'device', 'os', 'channel'}
    return df, fs


def f_hour(df,fs):
    f_name = 'hour'
    df[f_name] = pd.to_datetime(df.click_time).dt.hour.astype('uint8')
    gc.collect()
    fs.add(f_name)
    return df, fs

def f_dfw(df,fs):
    f_name = 'dayofweek'
    df[f_name] = pd.to_datetime(df.click_time).dt.dayofweek.astype('uint8')
    gc.collect()
    fs.add(f_name)
    return df, fs


def f_count(df,fs):
    groupby_list = [['ip'],['os','device'],['os','device','hour'],['app','channel'],['ip','hour']]
    select = 'click_time'
    agg = 'count'
    type = 'uint32'
    for groupby in groupby_list:
        gb_dict = {'groupby': groupby, 'select': select, 'agg': agg, 'type': type}
        df, f_name = f_template(df, gb_dict)
        fs.add(f_name)
    return df,fs

def f_mean_hour(df,fs):
    # Count, for ip
    groupby_list = [['ip'],['os','device']]
    select = 'hour'
    agg = 'mean'
    type = 'uint32'
    for groupby in groupby_list:
        gb_dict = {'groupby': groupby, 'select': select, 'agg': agg, 'type': type}
        df, f_name = f_template(df, gb_dict)
        fs.add(f_name)
    fs.add(f_name)
    return df, fs


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
                          scale_pos_weight=9,
                          seed=99)




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
    s.test_result = pd.DataFrame()
    s.test_result['is_attributed'] = s.model.predict_proba(s.test_df)[:,1]
    s.test_result['click_id']=s.test_result.index
    #s.test_result = s.test_result['click_id','is_attributed']
    pass

########################################## Solution #####################################
class Solution:
    train_df = None
    label_df = None
    test_df = None
    f_ppl = []
    train_f_set = None
    test_f_set = None
    model = None
    train_result = None
    test_result = None
    log = None
    method = 'xgb'
    result_filename = 'submission.csv'
    input_path = '../input/'
    output_path = '../output/'
    data_set = None
    transductive =False
    para_tune_fcg = None

########################################## Macro Control Panel #####################################
TRAIN_FILE = 'train.csv'
TEST_FILE= 'test.csv'
MAX_TRAIN_ROWS = 184903889
MAX_TEST_ROWS =  18790468
NUM_TRAIN_ROWS=100000
NUM_TEST_ROWS=1000
SKIP_TRAIN_ROWS=range(1,MAX_TRAIN_ROWS-NUM_TRAIN_ROWS)

FEATURE_PPL = [f_base,f_hour,f_count,f_mean_hour]
METHOD = 'xgb'
TRANSDUCTIVE = True
INPUT_PATH = '../input/'
OUTPUT_PATH = '../output/'
PARA_TUNE_CFG = '../cfg/default.cfg'

##########################################         Setting up           #####################################

d = DataSet()
d.train_file=TRAIN_FILE
d.test_file=TEST_FILE
d.skip_train_rows=SKIP_TRAIN_ROWS
d.num_train_rows = NUM_TRAIN_ROWS
d.num_test_rows= NUM_TEST_ROWS

s = Solution()
s.data_set = d
s.method = METHOD
s.f_ppl= FEATURE_PPL
s.para_tune_fcg = PARA_TUNE_CFG
s.transductive = TRANSDUCTIVE
s.input_path =INPUT_PATH
s.output_path=OUTPUT_PATH
##########################################            Excution                ##################################

tic = report("Load Dataset Start")
if s.transductive:
    s.data_set.load_train(s)
    s.data_set.load_test(s)
else:
    s.data_set.load_train(s)
report("Load Dataset Done", tic)


tic = report("Build Feature Start")
if s.transductive:
    merge = pd.concat([s.train_df, s.test_df])
    merge, f_set = build_features(merge, s.f_ppl)
    s.train_f_set, s.test_f_set = f_set, f_set
    s.train_df, s.test_df = merge[:s.train_df.shape[0]], merge[s.train_df.shape[0]:].reset_index(
        drop=True)
else:
    s.train_df, s.train_f_set = build_features(s.train_df, s.f_ppl)
print("Feature Selected:\t{0}".format(','.join(s.train_f_set)))
report("Build Features Done", tic)


xgb_init(s)
tic = report("Model training Start")
xgb_train(s)
xgb_save_fi(s)
report("Model training Done", tic)

tic = report("Test Start")
if not s.transductive:
    d.load_test(s)
    s.test_df, s.test_f_set = build_features(s.test_df, s.f_ppl)
xgb_test(s)
report("Test Done", tic)
check_dir(s.output_path)
s.test_result.to_csv(s.output_path + s.result_filename, float_format='%.8f', index=False)








