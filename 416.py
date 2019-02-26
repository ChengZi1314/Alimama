import time
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import log_loss
from sklearn import ensemble
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.linear_model import LogisticRegression 
from scipy import sparse
import xgboost as xgb
import operator


import numpy as np 
import datetime


# 由于时间是偏移的这里，做一下时间的特征
def timestamp_datetime(value):
    format = '%Y-%m-%d %H:%M:%S'
    value = time.localtime(value)
    dt = time.strftime(format, value)
    return dt
extract_feature = ['is_last']


# 数据预处理
def pre_process(data): 
    for i in range(3):
        data['category_%d'%(i)] = data['item_category_list'].apply(lambda x:x.split(";")[i] if len(x.split(";")) > i else " ")    
    del data['item_category_list']

    for i in range(3):
        data['property_%d'%(i)] = data['item_property_list'].apply(lambda x:x.split(";")[i] if len(x.split(";")) > i else " ")
    del data['item_property_list']

    for i in range(3):
        data['predict_category_%d'%(i)] = data['predict_category_property'].apply(lambda x:str(x.split(";")[i]).split(":")[0] if len(x.split(";")) > i else " ")

    # data['year'] = data['']
    data['time'] = data.context_timestamp.apply(timestamp_datetime)
    data['day'] = data.time.apply(lambda x: int(x[8:10]))
    data['hour'] = data.time.apply(lambda x: int(x[11:13]))
    
    return data


# # 贝叶斯平滑获取点击率
# def extract_bays_smooth_feature(data, feat_sum, feat_rate, rate_name):
#     #统计该属性的商品点击的次数
#     temp = data.groupby(feat_rate).size().reset_index().rename(columns={0: 'trade_sum'})
#     temp = temp[temp.is_trade == 1]
#
#     #统计该属性的商品展示的次数
#     temp1 = data.groupby(feat_sum).size().reset_index().rename(columns = {0: 'sum'})
#
#     #将数据按关键属性进行合并
#     rate = pd.merge(temp, temp1, on = feat_sum, how = 'left')
#
#     # 对数据进行beyes平滑，并计算出alpha值和beta值
#     bs = BayesianSmoothing(0.001, 0.001)
#     bs.update(rate['sum'].tolist(), rate['trade_sum'].tolist(), 10000, 0.00001)
#     alpha = bs.alpha
#     beta = bs.beta
#
#     #计算该属性情况下商品被点击概率
#     rate['sum'] = rate['sum'] + alpha + beta
#     rate['trade_sum'] = rate['trade_sum'] + alpha
#     rate[rate_name + '_rate'] = rate['trade_sum'] / rate['sum']
#     return rate


def extract_rate_feature(data, feat_rate, rate_name, train, test, fill_Value):
    feature_ = []
    feature_.extend(feat_rate)
    feature_.append(rate_name + '_count')
    feature_.append(rate_name + '_sum')
    feature_.append(rate_name + '_rate')
    extract_feature.append(rate_name + '_count')
    extract_feature.append(rate_name + '_sum')
    extract_feature.append(rate_name + '_rate')

    temp = data.groupby(feat_rate)['is_trade'].agg(
        {rate_name+'_count':'count',rate_name+'_sum' : 'sum'}).reset_index()
    temp[rate_name + '_rate'] = temp[rate_name+'_sum'] / temp[rate_name+'_count']
    train = pd.merge(train, temp[feature_] , on = feat_rate, how = 'left')
    test = pd.merge(test, temp[feature_] , on = feat_rate, how = 'left')
    train[rate_name+'_rate'] = train[rate_name+'_rate'].fillna(fill_Value)
    test[rate_name+'_rate'] = test[rate_name+'_rate'].fillna(fill_Value)
    train = train.fillna(0)
    test = test.fillna(0)
    return train, test


# 抽取特征
def extract_User_Item_Feature(data,train,test):
    # 用户职业跟价格的关系
    train,test = extract_rate_feature(data, ['user_occupation_id', 'item_price_level'], 'occup_prices',train,test,1)
    # 用统计量来表示职业与点击广告之间的关系
    train, test = extract_rate_feature(data, ['user_occupation_id'], 'occup', train, test, 1)
    # 页面与成交
    train, test = extract_rate_feature(data, ['context_page_id'], 'page', train, test, 1)
    # 展示次数与成交
    train, test = extract_rate_feature(data, ['item_pv_level'], 'pv',train,test,1)
    # 销量等级与成交
    train, test = extract_rate_feature(data, ['item_sales_level'],'sales_level',train,test,1)
    # 用户星级与价格
    train, test = extract_rate_feature(data, ['user_star_level','item_price_level'], 'star_price',train,test,1)
    # 用户年龄与页面
    train, test = extract_rate_feature(data, ['user_age_level', 'context_page_id'],'age_page', train, test, 1)
    # 用户职业与页面
    train, test = extract_rate_feature(data, ['user_occupation_id', 'context_page_id'], 'occup_page',train,test,1)
    # 用户性别跟页面
    train, test = extract_rate_feature(data, ['user_gender_id', 'context_page_id'], 'occip_gender_page',train,test,1)
    # 用统计量来表示职业与商品类目（类目一）之间的关系
    train, test = extract_rate_feature(data, ['user_occupation_id', 'category_1'], 'occup_cate',train,test,1)
    # 用统计量来表示职业与商铺品牌之间的关系
    train, test = extract_rate_feature(data, ['user_occupation_id', 'item_brand_id'], 'occup_brand',train,test,1)
    # 职业和商品城市之间的关系
    train, test = extract_rate_feature(data, ['user_occupation_id', 'item_city_id'], 'occup_city',train,test,1)
    # 职业和时间之间的关系, 主要考虑职业在时间上的空闲时间
    train, test = extract_rate_feature(data, ['user_occupation_id', 'hour'], 'occup_hour',train,test,1)
    # 考虑用户年龄跟商商品的类目可能有关系
    train, test = extract_rate_feature(data, ['user_age_level', 'category_1'], 'age_cate',train,test,1)
    # 考虑用户年龄和商铺品牌的关系
    train, test = extract_rate_feature(data, ['user_age_level', 'item_brand_id'], 'age_brand',train,test,1)
    # 考虑用户的年龄和时间的关系
    train, test = extract_rate_feature(data, ['user_age_level', 'hour'], 'age_hour',train,test,1)
    # 考虑城市id和类目1的关系
    train,test = extract_rate_feature(data, ['item_city_id', 'category_1'], 'city_cate1',train,test,1)
    train.to_csv('train.csv',index = None)
    test.to_csv('test.csv', index = None)
    return train, test


# 统计出现次数排名
def countTimesRank(data, feature_, name):
    temp = data.groupby(feature_).size().reset_index().rename(columns = {0: name + '_sum'})
    temp = temp.sort_values(feature_, ascending = False)
    temp[name + '_rank'] = range(1, len(temp)+1)
    return temp


def chengfa(data):
    data['good_review'] = data['shop_review_num_level']*data['shop_review_positive_rate']
    return  data


def extract_times_rank(data, train, test):
    item_times_rank = countTimesRank(data, ['item_id'], 'item_id')
    shop_id_times_rank = countTimesRank(data, ['shop_id'], 'shop_id')
    user_item_times = countTimesRank(data, ['user_id', 'item_id'], 'user_item')
    cate0_item_rank = countTimesRank(data,['category_0','item_id'],'cate0_item')
    train = pd.merge(train, item_times_rank, on = ['item_id'], how = 'left')
    train = pd.merge(train, shop_id_times_rank, on = ['shop_id'], how = 'left')
    train = pd.merge(train, user_item_times, on = ['user_id', 'item_id'], how = 'left')
    train = pd.merge(train,cate0_item_rank,on=['category_0','item_id'],how='left')
    test = pd.merge(test, item_times_rank, on = ['item_id'], how = 'left')
    test = pd.merge(test, shop_id_times_rank, on = ['shop_id'], how = 'left')
    test = pd.merge(test, user_item_times, on = ['user_id', 'item_id'], how = 'left')
    test = pd.merge(test, cate0_item_rank, on=['category_0', 'item_id'], how='left')
    train = train.fillna(0)
    test = test.fillna(0)
    train.to_csv('train.csv', index = None)
    test.to_csv('test.csv', index = None)
    return train, test

def creat_last_time_trick(train, test):
    x = train.groupby(['user_id', 'item_id'])['time'].max().reset_index().rename(
    columns = {'time': 'max_time'}) 
    train = pd.merge(train, x, on = ['user_id', 'item_id'], how = 'left')
    train['is_last'] = (train['time'] == train['max_time']).map(lambda x: 1 if x else 0)
    
    x = test.groupby(['user_id', 'item_id'])['time'].max().reset_index().rename(
    columns = {'time': 'max_time'}) 
    test = pd.merge(test, x, on = ['user_id', 'item_id'], how = 'left')
    test['is_last'] = (test['time'] == test['max_time']).map(lambda x: 1 if x else 0)

    return train, test


# 前后时间差
def extract_user_click_time(data):
    user_timestamp = data[['user_id', 'time']].drop_duplicates()
    # 按时间排序
    user_timestamp['time_rank'] = user_timestamp['time'].groupby(user_timestamp['user_id']).rank(ascending=True)
    # 把时间排序以后，然后在进行前后挪动
    user_timestamp_1 = user_timestamp.copy()
    user_timestamp_2 = user_timestamp.copy()
    user_timestamp_1['time_rank'] = user_timestamp_1['time_rank'] - 1
    user_timestamp_2['time_rank'] = user_timestamp_2['time_rank'] + 1
    user_timeall = pd.merge(user_timestamp_1, user_timestamp_2, on=['user_id', 'time_rank'], how='left')
    user_timeall = pd.merge(user_timeall, user_timestamp, on=['user_id', 'time_rank'], how='left')
    user_timeall = user_timeall.fillna('1900-01-01 00:00:00')
    user_timeall['diff1'] = user_timeall['time_x'].map(lambda x: datetime.datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S')) - user_timeall['time'].map(lambda x: datetime.datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S'))
    user_timeall['days_1'] = user_timeall['diff1'].map(lambda x: x.days)
    user_timeall['second_1'] = user_timeall['diff1'].map(lambda x: x.seconds)
    user_timeall['diff2'] = user_timeall['time'].map(
        lambda x: datetime.datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S')) - user_timeall['time_y'].map(
        lambda x: datetime.datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S'))
    user_timeall['days_2'] = user_timeall['diff2'].map(lambda x: x.days)
    user_timeall['second_2'] = user_timeall['diff2'].map(lambda x: x.seconds)
    user_timeall = user_timeall[['user_id', 'time', 'days_1', 'days_2', 'second_1', 'second_2', 'time_rank']]
    data = pd.merge(data, user_timeall, on=['user_id', 'time'], how='left')
    return data


def extract_frist_time(train,test):
    x = train.groupby(['user_id','item_id','day']).size().reset_index().rename(
        columns = {0:'dayclick_times'})
    train = pd.merge(train,x,on=['user_id','item_id','day'],how='left')

    x = test.groupby(['user_id','item_id','day']).size().reset_index().rename(
        columns = {0 :'dayclick_times'})
    test = pd.merge(test, x, on=['user_id','item_id','day'],how='left')

    x = train.groupby(['user_id','predict_category_1','day']).size().reset_index().rename(
        columns = {0:'user_day_cate_1'}
    )
    train = pd.merge(train,x,on=['user_id','predict_category_1','day'],how='left')
    x = test.groupby(['user_id', 'predict_category_1', 'day']).size().reset_index().rename(
        columns={0: 'user_day_cate_1'}
    )
    test = pd.merge(test, x, on=['user_id', 'predict_category_1', 'day'], how='left')
    x = train.groupby(['user_id', 'shop_id', 'day']).size().reset_index().rename(
        columns={0:'user_shop_day'}
    )

    train = pd.merge(train,x,on=['user_id','shop_id','day'],how='left')

    x = test.groupby(['user_id','shop_id','day']).size().reset_index().rename(
        columns = {0:'user_shop_day'}
    )
    test = pd.merge(test,x,on=['user_id','shop_id','day'],how='left')
    # 0410
    # 同一个商品一天内被多少不同的人浏览过
    x = train.groupby(['user_id', 'item_id', 'day']).size()
    f = x.groupby(['item_id', 'day']).size().reset_index().rename(
        columns = {0:'diff_user_item'}
    )
    train = pd.merge(train, f, on=['item_id','day'], how='left')
    x = test.groupby(['user_id', 'item_id', 'day']).size()
    f = x.groupby(['item_id','day']).size().reset_index().rename(
        columns = {0:'diff_user_item'}
    )
    test = pd.merge(test, f, on=['item_id','day'], how='left')
    x = train.groupby(['user_id','item_id']).size().reset_index().rename(
        columns = {0:'user_item_times'}
    )
    train = pd.merge(train,x,on=['user_id','item_id'],how='left')
    x = test.groupby(['user_id', 'item_id']).size().reset_index().rename(
        columns={0: 'user_item_times'}
    )
    test = pd.merge(test, x, on=['user_id', 'item_id'], how='left')

    x = train.groupby(['user_id','category_1','day']).size().reset_index().rename(
        columns = {0:'user_cate1_day'}
    )
    train = pd.merge(train,x,on=['user_id','category_1','day'],how='left')
    x = test.groupby(['user_id', 'category_1', 'day']).size().reset_index().rename(
        columns={0: 'user_cate1_day'}
    )
    test = pd.merge(test, x, on=['user_id', 'category_1', 'day'], how='left')

    x = train.groupby(['user_id','category_0','day']).size().reset_index().rename(
        columns = {0:'user_cate0_day'}
    )
    train = pd.merge(train,x,on=['user_id','category_0','day'],how='left')
    x = test.groupby(['user_id', 'category_0', 'day']).size().reset_index().rename(
        columns={0: 'user_cate0_day'}
    )
    test = pd.merge(test, x, on=['user_id', 'category_0', 'day'], how='left')

    x = train.groupby(['user_id','predict_category_0','day']).size().reset_index().rename(
        columns = {0:'user_day_cate_0'}
    )
    train = pd.merge(train,x,on=['user_id','predict_category_0','day'],how='left')
    x = test.groupby(['user_id', 'predict_category_0', 'day']).size().reset_index().rename(
        columns={0: 'user_day_cate_0'}
    )

    test = pd.merge(test, x, on=['user_id', 'predict_category_0', 'day'], how='left')
    x = train.groupby(['user_id','predict_category_1','day','predict_category_2']).size().reset_index().rename(
        columns = {0:'user_day_cate_12'}
    )
    train = pd.merge(train,x,on=['user_id','predict_category_1','day','predict_category_2'],how='left')
    x = test.groupby(['user_id','predict_category_1','day','predict_category_2']).size().reset_index().rename(
        columns={0: 'user_day_cate_12'}
    )
    test = pd.merge(test, x, on=['user_id','predict_category_1','day','predict_category_2'], how='left')
    # 用户当天点击了多少对应价格的
    x = train.groupby(['user_id', 'item_sales_level', 'day']).size().reset_index().rename(
        columns={0: 'user_day_page'}
    )
    train = pd.merge(train, x, on=['user_id', 'item_sales_level'], how='left')
    x = test.groupby(['user_id', 'item_sales_level','day']).size().reset_index().rename(
        columns={0: 'user_day_page'}
    )
    test = pd.merge(test, x, on=['user_id', 'item_sales_level'], how='left')



    return train,test

# def get_time_diff(date_after, date_before):
#     # 计算时间差
#     day_diff = int(date_after[-11:-9]) - int(date_before[-11:-9])
#     hour_diff = int(date_after[-8:-6]) - int(date_before[-8:-6])
#     min_diff = int(date_after[-5:-3]) - int(date_before[-5:-3])
#     second_diff = int(date_after[-2:])-int(date_before[-2:])
#     if day_diff == 0:
#         if hour_diff == 0 :
#             if min_diff == 0:
#                 return int(second_diff)
#             else:
#                 return int(second_diff)+min_diff*60
#         else:
#             if min_diff == 0:
#                 return int(second_diff)+hour_diff*3600
#             else:
#                 return int(second_diff)+hour_diff*3600+min_diff*60
#     else:
#         if hour_diff == 0:
#             if min_diff == 0:
#                 return int(second_diff)+day_diff*86400
#             else:
#                 return int(second_diff)+day_diff*86400+min_diff*60
#         else:
#             if min_diff == 0:
#                 return int(second_diff)+day_diff*86400+hour_diff*3600
#             else:
#                 return int(second_diff)+day_diff*86400+min_diff*60+hour_diff*3600


def get_mean(train, test, feature, feature_target, name):
    x = train.groupby(feature)[feature_target].mean().reset_index().rename(columns = {feature_target:name})
    train = pd.merge(train, x, on=feature, how='left')
    x = test.groupby(feature)[feature_target].mean().reset_index().rename(columns = {feature_target:name})
    test = pd.merge(test, x, on=feature, how='left')
    return train, test

# def findmax(train,test):
#     item_zhiye = train.groupby(['item_id', 'user_occupation_id'])['is_trade'].agg(
#         {'user_occupa_count': 'count'}).reset_index()
#     item_zhiye['occupa_rank'] = item_zhiye['user_occupa_count'].groupby(item_zhiye['item_id']).rank(ascending=False)
#     x = item_zhiye[item_zhiye.occupa_rank == 1].reset_index().rename(
#         columns={'user_occupation_id': 'new_user_occupation_id'})
#     train = pd.merge(train, x, on=['item_id'], how='left')
#     train.fillna(2005)
#     item_zhiye = test.groupby(['item_id', 'user_occupation_id'])['is_trade'].agg(
#         {'user_occupa_count': 'count'}).reset_index()
#     item_zhiye['occupa_rank'] = item_zhiye['user_occupa_count'].groupby(item_zhiye['item_id']).rank(ascending=False)
#     x = item_zhiye[item_zhiye.occupa_rank == 1].reset_index().rename(
#         columns={'user_occupation_id': 'new_user_occupation_id'})
#     test = pd.merge(test, x, on=['item_id'], how='left')
#     test.fillna(2005)
#     return train,test
def getavge(train,test):
    # 获取每个广告商品的平均价格
    train,test = get_mean(train,test,['item_id'],'item_price_level','mean_item_prices')
    # train, test = get_median(train, test, ['item_id'], 'item_price_level', 'median_item_prices')
    # 获取商铺平均价格
    train,test = get_mean(train,test,['shop_id'],'item_price_level','shop_mean_price')
    # train, test = get_median(train, test, ['shop_id'], 'item_price_level', 'shop_median_price')
    # 获取点击某商品的用户平均年龄
    train,test = get_mean(train,test,['item_id'],'user_age_level','mean_user_age')
    # train, test = get_median(train, test, ['item_id'], 'user_age_level', 'median_user_age')
    # 获取某品牌的平均价格
    train,test = get_mean(train,test,['item_brand_id'],'item_price_level','mean_brand_prices')
    # train, test = get_median(train, test, ['item_brand_id'], 'item_price_level', 'median_brand_prices')
    # 获取点击某价格等级商品用户平均年龄
    train,test = get_mean(train,test,['item_price_level'],'user_age_level','mean_price_age')
    # train, test = get_median(train, test, ['item_price_level'], 'user_age_level', 'median_price_age')
    # 获取点击某品牌的用户平均年龄
    train,test = get_mean(train,test,['item_brand_id'],'user_age_level','mean_brand_age')
    # train, test = get_median(train, test, ['item_brand_id'], 'user_age_level', 'median_brand_age')
    # #获取点击某店铺的用户平均年龄
    # train,test =  get_mean(train,test,['shop_id'],'user_age_level','mean_shop_age')
    # #获取某广告城市的平均年龄
    # train,test = get_mean(train,test,['item_city_id'],'user_age_level','mean_city_age')
    # 获取用户点击商品的平均销量等级
    # train,test = get_mean(train,test,['user_id'],'item_sales_level','user_mean_sales')
    return train,test


# 利用分词word2vec将商品名称, 商品属性等做一个特征向量，也叫嵌入式特征
def shop_mean(data, feature1, feature2, feature3, feature4):
    s1 = data.groupby(feature1)[feature2].apply(sum).reset_index().rename(
        columns={0: '_sum1'}
    )
    s2 = data.groupby(feature1).size().reset_index().rename(
        columns={0: '_count1'}
    )
    s11 = data.groupby(feature1)[feature3].apply(sum).reset_index().rename(
        columns={0: '_sum2'}
    )
    s22 = data.groupby(feature1).size().reset_index().rename(
        columns={0: '_count2'}
    )
    s111 = data.groupby(feature1)[feature4].apply(sum).reset_index().rename(
        columns={0: '_sum3'}
    )
    s222 = data.groupby(feature1).size().reset_index().rename(
        columns={0: '_count3'}
    )
    s1['mean1'] = s1[feature2] / s2['_count1']
    s1['mean2'] = s11[feature3] / s22['_count2']
    s1['mean3'] = s111[feature4] / s222['_count3']
    s1[feature3] = s11[feature3]
    # s1[feature2] =train[feature2]
    s1[feature4] = s111[feature4]
    for i in range(3):
        s1['is_low1'] = (s1['mean1'] <= s1[feature2]).map(lambda x: 1 if x else 0)
        s1['is_low2'] = (s1['mean2'] <= s1[feature3]).map(lambda x: 1 if x else 0)
        s1['is_low3'] = (s1['mean3'] <= s1[feature4]).map(lambda x: 1 if x else 0)
        s1['low_number'] = s1['is_low1'] + s1['is_low2'] + s1['is_low3']
    del s1[feature2]
    del s1[feature3]
    del s1[feature4]
    data = pd.merge(data, s1, on=feature1, how='left')
    return data



def getFeature():
    id_features = ['context_page_id','item_id', 'item_brand_id',
                   'item_city_id','user_gender_id',
                   'shop_id','user_id','category_0','category_1','category_2',
                   'property_0','property_1','property_2','predict_category_0',
                   'predict_category_1','predict_category_2']

    num_features = [ 'item_price_level', 'item_sales_level','user_star_level',
                'item_collected_level', 'item_pv_level', 'user_age_level',
                'shop_review_num_level', 'shop_star_level','shop_review_positive_rate',
                'shop_score_service', 'shop_score_delivery', 'shop_score_description','hour','mean_item_prices'
                     ,'shop_mean_price','mean_brand_prices','low_number'
                ]

    extract_feature = ['occup_rate','occup_cate_rate','occup_city_rate', 'occup_brand_rate', 'occup_hour_rate',
                        'age_cate_rate', 'age_brand_rate', 'item_id_rank',
                        'item_id_sum', 'shop_id_rank', 'shop_id_sum',
                        'age_hour_rate', 'is_last', 'user_item_sum','age_page_rate','occip_gender_page_rate','occup_page_rate','star_price_rate',
                       'page_rate','pv_rate','sales_level_rate','user_day_cate_1','user_shop_day','dayclick_times',
                       'occup_prices_rate','user_cate1_day','user_day_cate_12','diff_user_item','user_day_page','user_cate0_day'
                       ,'days_1', 'days_2', 'second_1', 'second_2','city_cate1_rate','mean_user_age','mean_price_age','mean_brand_age','max_occup'
                        ]
    x_columns = []
    for i in range(200):
        x_columns.append('item_id'+str(i))
    feature = []
    feature.extend(id_features)
    feature.extend(num_features)
    feature.extend(extract_feature)
    # feature.extend(x_columns)
    target = ['is_trade']
    return feature, id_features, num_features, target


def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()


def max_show(df):
    item_zhiye = df.groupby(['item_id', 'user_occupation_id']).size().reset_index().rename(
    columns = {0:'count'})
    item_zhiye['occupa_rank'] = item_zhiye['count'].groupby(item_zhiye['item_id']).rank(ascending=False)
    x = item_zhiye[item_zhiye.occupa_rank == 1]
    del x['count']
    del x['occupa_rank']
    names = ['item_id', 'max_occup']
    x.columns = names
    df = pd.merge(df, x, on=['item_id'],how='left')
    df['max_occup'].where(df['max_occup'].notnull(), 2005, inplace=True)
    return df


# 将数据切分为特征集，训练集，测试集
def split_feature_train_test(data, online):
    if not online:
        feature = data.loc[data.day < 22]  # 18,19,20,21,22,23,24
        train = data.loc[(data.day < 24) & (data.day > 21)]
        test = data.loc[data.day == 24]  # 暂时先使用第24天作为验证集
    elif online:
        feature = data.loc[data.day < 20]  # 18,19,20,21,22,23,24
        train = data.loc[(data.day < 25) & (data.day > 19)]
        test = pd.read_csv('round1_ijcai_18_test_a_20180301.txt', sep=' ')
        test = pre_process(test)
        test = extract_user_click_time(test)
        test = shop_mean(test, ['shop_id', 'category_0'], 'shop_score_service'
                        , 'shop_score_delivery', 'shop_score_description')
    return feature, train, test


def xgboost_model(train, test, features, target):

    params={'booster': 'gbtree',
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 7,
            'lambda': 100,
            'subsample': 0.7,
            'colsample_bytree': 0.5,
            'eta': 0.01,
            'seed': 1024,
            'nthread': 12,
            'silent': 1,
    }
    # ceate_feature_map(features)
    dtrain = xgb.DMatrix(train[features], label=train[target])
    dtest = xgb.DMatrix(test[features])
    clf = xgb.train(params, dtrain,num_boost_round=2000, early_stopping_rounds=30, verbose_eval=True)
    test['predicted_score'] = clf.predict(dtest)
    test.to_csv('test.csv', index=None)
    return test


if __name__ == "__main__":
    print("程序开始")
    # 这里用来标记是 线下验证 还是 在线提交
    online = True
    data = pd.read_csv('round1_ijcai_18_train_20180301.txt', sep=' ')
    data.drop_duplicates(inplace=True)
    data = pre_process(data)
    data = chengfa(data)
    data = extract_user_click_time(data)
    data = shop_mean(data, ['shop_id', 'category_0'], 'shop_score_service'
                            ,'shop_score_delivery', 'shop_score_description')
    data = max_show(data)
    f_train, train, test = split_feature_train_test(data, online)
    print("特征抽取开始")
    # train, test = create_w2v_feature(train, test)
    train, test = extract_times_rank(f_train, train, test)
    train, test = extract_User_Item_Feature(f_train, train, test)
    train, test = creat_last_time_trick(train, test)
    train, test = extract_frist_time(train, test)
    train, test = getavge(train, test)
    # train, test = findmax(train, test)
    print("特征抽取结束")

    train = train.fillna(1)
    test = test.fillna(1)
    features, id_features, num_features, target = getFeature()
    
    lb = LabelEncoder()
    for feat in id_features:
        tmp = lb.fit_transform((list(f_train[feat])+list(train[feat])+list(test[feat])))
        f_train[feat] = lb.transform(list(f_train[feat]))
        train[feat] = lb.transform(list(train[feat]))
        test[feat] = lb.transform(list(test[feat]))   

    # im_fea = xgboost_choose_feature(train, features)

    test = xgboost_model(train, test, features, target)
    # test = lgb_model(train, test, features, target)

    if online == False:
        print(log_loss(test[target], test['predicted_score']))
    else:
        test[['instance_id', 'predicted_score']].to_csv('414.txt', index=False,sep=' ')
        # 保存在线提交结果
