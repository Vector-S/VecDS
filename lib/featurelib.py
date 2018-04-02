import pandas as pd
import gc

def f_template(df, gb_dict):
    
    all_features = list(set(gb_dict['groupby'] + [gb_dict['select']]))
    ## name of new feature
    new_feature = '{}_{}_{}'.format('_'.join(gb_dict['groupby']), gb_dict['agg'], gb_dict['select'])
    ## perfom the grouby
    gp = df[all_features]. \
        groupby(gb_dict['groupby'])[gb_dict['select']]. \
        agg(gb_dict['agg']). \
        reset_index(). \
        rename(index=str, columns={gb_dict['select']: new_feature}).astype(gb_dict['type'])
    # Merge back to df
    df = df.merge(gp, on=gb_dict['groupby'], how='left')
    print(df.columns.values.tolist())
    del gp
    gc.collect()

    feature_add =[new_feature]
    feature_drop =[]
    return feature_add, feature_drop

def f_base(df):
    feature_add =['ip', 'app', 'device', 'os']
    feature_drop =[]
    df.drop(['attributed_time', 'is_attributed'], axis=1, inplace=True)
    gc.collect()
    return feature_add,feature_drop


def f_1(df):
    feature_add =['hour','day']
    feature_drop =['click_time']
    df['hour'] = pd.to_datetime(df.click_time).dt.hour.astype('uint8')
    df['day'] = pd.to_datetime(df.click_time).dt.day.astype('uint8')
    df.drop(['click_time'], axis=1, inplace=True)
    gc.collect()
    return feature_add,feature_drop

def f_1_2(df):
    feature_add =['hour','day']
    feature_drop =['click_time']
    df['click_rnd']=df['click_time'].dt.round('H')
    df['hour'] = pd.to_datetime(df.click_rnd).dt.hour.astype('uint8')
    df['day'] = pd.to_datetime(df.click_rnd).dt.day.astype('uint8')
    df.drop(['click_time', 'click_rnd'], axis=1, inplace=True)
    gc.collect()
    return feature_add,feature_drop

def f_2(df):
    feature_add =['nip_day_test_hh']
    feature_drop =[]
    most_freq_hours_in_test_data = [4, 5, 9, 10, 13, 14]
    least_freq_hours_in_test_data = [6, 11, 15]
    df['in_test_hh'] = (3
                        - 2 * df['hour'].isin(most_freq_hours_in_test_data)
                        - 1 * df['hour'].isin(least_freq_hours_in_test_data)).astype('uint8')
    gp = df[['ip', 'day', 'in_test_hh', 'channel']].groupby(by=['ip', 'day', 'in_test_hh'])[
        ['channel']].count().reset_index().rename(index=str, columns={'channel': 'nip_day_test_hh'})
    df = df.merge(gp, on=['ip', 'day', 'in_test_hh'], how='left')
    df.drop(['in_test_hh'], axis=1, inplace=True)
    df['nip_day_test_hh'] = df['nip_day_test_hh'].astype('uint32')
    del gp
    gc.collect()
    return feature_add,feature_drop

def f_3(df):
    feature_add =['ip_count']
    feature_drop=[]
    gp = df[['ip', 'channel']].groupby(by=['ip'])[
        ['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_count'})
    df = df.merge(gp, on=['ip'], how='left')   
    del gp
    gc.collect()
    return feature_add,feature_drop


def f_4(df):
    feature_add =['app_count']
    feature_drop=[]
    gp = df[['app', 'channel']].groupby(by=['app'])[
        ['channel']].count().reset_index().rename(index=str, columns={'channel': 'app_count'})
    df = df.merge(gp, on=['app'], how='left')   
    del gp
    gc.collect()
    return feature_add,feature_drop

def f_5(df):
    feature_add =['dev_count']
    feature_drop=[]
    gp = df[['device', 'channel']].groupby(by=['device'])[
        ['channel']].count().reset_index().rename(index=str, columns={'channel': 'dev_count'})
    df = df.merge(gp, on=['device'], how='left')   
    del gp
    gc.collect()
    return feature_add,feature_drop

def f_6(df):
    feature_add =['os_count']
    feature_drop=[]
    gp = df[['os', 'channel']].groupby(by=['os'])[
        ['channel']].count().reset_index().rename(index=str, columns={'channel': 'os_count'})
    df = df.merge(gp, on=['os'], how='left')   
    del gp
    gc.collect()
    return feature_add,feature_drop


def f_7(df):
    feature_add =['chnl_count']
    feature_drop=[]
    gp = df[['os', 'channel']].groupby(by=['channel'])[
        ['os']].count().reset_index().rename(index=str, columns={'os': 'chnl_count'})
    df = df.merge(gp, on=['channel'], how='left')   
    del gp
    gc.collect()
    return feature_add,feature_drop


def f_8(df):
    feature_add =['ip_app_count']
    feature_drop=[]
    gp = df[['ip','app', 'channel']].groupby(by=['ip', 'app'])[
        ['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_count'})
    df = df.merge(gp, on=['ip','app'], how='left')   
    del gp
    gc.collect()
    return feature_add,feature_drop


if __name__ == "__main__":
    train_df = pd.read_csv('../input/train.csv', nrows=1000)
    # print(train_df.columns.values.tolist())

    ## test f_base
    f_a, f_d = f_base(train_df)
    assert f_a == ['ip', 'app', 'device', 'os']
    assert train_df.columns.values.tolist() == ['ip', 'app', 'device', 'os', 'channel', 'click_time']

    ## test f_1
    f_a, f_d = f_1(train_df)
    assert f_a == ['hour','day']
    assert f_d == ['click_time']
    assert train_df.columns.values.tolist() == ['ip', 'app', 'device', 'os', 'channel', 'hour', 'day']

    ## test f_2
    f_a, f_d = f_2(train_df)
    assert f_a == ['nip_day_test_hh']
    assert train_df.columns.values.tolist() == ['ip', 'app', 'device', 'os', 'channel', 'hour', 'day', 'nip_day_test_hh']

    # ## test f_template
    # gb_dict1 = {'groupby': ['ip', 'app'], 'select': 'channel', 'agg': 'count', 'type': 'uint32'}
    # gb_dict2 = {'groupby': ['ip','app','channel'], 'select': 'hour', 'agg': 'mean', 'type': 'float32', 'type': 'float32'}
    # f_a, f_d = f_template(train_df, gb_dict1)
    # # print(f_a)
    # print(train_df.columns.values.tolist())


    
