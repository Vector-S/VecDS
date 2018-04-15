import pandas as pd
import gc



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

#######################  feature definition zone #######################
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

def f_mean(df,fs):
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


def f_2(df,fs):
    """
    :param df:
    :param fs:
    :return:
    """
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
    fs |= {'nip_day_test_hh'}
    return df, fs

def f_1_2(df,fs):
    click_time= pd.to_datetime(df.click_time)
    df['click_rnd']=click_time.dt.round('H')
    df['hour'] = pd.to_datetime(df.click_rnd).dt.hour.astype('uint8')
    df['day'] = pd.to_datetime(df.click_rnd).dt.day.astype('uint8')
    df.drop('click_rnd',axis=1,inplace=True)
    fs =fs|{'hour','day'}
    gc.collect()
    return df, fs


if __name__ == "__main__":
    train_df = pd.read_csv('../input/train.csv', nrows=1000)

    ## test f_base
    train_df, fs = f_base(train_df, set())
    assert fs == {'ip', 'app', 'device', 'os', 'channel'}

    ## test f_template
    gb_dict1 = {'groupby': ['ip'], 'select': 'channel', 'agg': 'count', 'type': 'float32', 'type': 'float32'}
    gb_dict2 = {'groupby': ['ip', 'app'], 'select': 'channel', 'agg': 'mean', 'type': 'uint32'}

    train_df, new_feature = f_template(train_df, gb_dict1)
    assert new_feature == 'ip_count_channel'

    train_df, new_feature = f_template(train_df, gb_dict2)
    assert new_feature == 'ip_app_mean_channel'



    


    
