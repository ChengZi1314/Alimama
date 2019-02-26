from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from datetime import datetime
from sklearn.metrics import log_loss
from keras import backend as K
import keras

import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


use_features = [
                'is_trade', 'context_page_id', 'item_id', 'item_brand_id','item_city_id', 'user_gender_id',
                'shop_id', 'user_id', 'category_0', 'category_1', 'category_2','property_0', 'mean_price_age',
                'property_2', 'predict_category_0', 'predict_category_1', 'predict_category_2', 'item_price_level',
                'item_sales_level', 'user_star_level', 'item_collected_level', 'item_pv_level', 'user_age_level',
                'shop_review_num_level', 'shop_star_level', 'shop_review_positive_rate','shop_score_service',
                'shop_score_delivery', 'shop_score_description', 'hour', 'mean_item_prices', 'mean_brand_age',
                'shop_mean_price', 'mean_brand_prices', 'low_number', 'occup_rate', 'occup_cate_rate','instance_id',
                'occup_city_rate', 'occup_brand_rate', 'occup_hour_rate', 'age_cate_rate', 'age_brand_rate',
                'item_id_rank', 'item_id_sum', 'shop_id_rank', 'shop_id_sum','age_hour_rate', 'is_last', 'property_1',
                'user_item_sum', 'age_page_rate', 'occip_gender_page_rate','occup_page_rate', 'star_price_rate',
                'page_rate', 'pv_rate', 'sales_level_rate', 'user_day_cate_1', 'user_shop_day', 'dayclick_times',
                'occup_prices_rate', 'user_cate1_day', 'user_day_cate_12', 'diff_user_item', 'user_day_page',
                'user_cate0_day', 'days_1', 'days_2', 'second_1', 'second_2', 'city_cate1_rate', 'mean_user_age',
]
test_features = [
                'context_page_id', 'item_id', 'item_brand_id','item_city_id', 'user_gender_id',
                'shop_id', 'user_id', 'category_0', 'category_1', 'category_2','property_0', 'mean_price_age',
                'property_2', 'predict_category_0', 'predict_category_1', 'predict_category_2', 'item_price_level',
                'item_sales_level', 'user_star_level', 'item_collected_level', 'item_pv_level', 'user_age_level',
                'shop_review_num_level', 'shop_star_level', 'shop_review_positive_rate','shop_score_service',
                'shop_score_delivery', 'shop_score_description', 'hour', 'mean_item_prices', 'mean_brand_age',
                'shop_mean_price', 'mean_brand_prices', 'low_number', 'occup_rate', 'occup_cate_rate','instance_id',
                'occup_city_rate', 'occup_brand_rate', 'occup_hour_rate', 'age_cate_rate', 'age_brand_rate',
                'item_id_rank', 'item_id_sum', 'shop_id_rank', 'shop_id_sum','age_hour_rate', 'is_last', 'property_1',
                'user_item_sum', 'age_page_rate', 'occip_gender_page_rate','occup_page_rate', 'star_price_rate',
                'page_rate', 'pv_rate', 'sales_level_rate', 'user_day_cate_1', 'user_shop_day', 'dayclick_times',
                'occup_prices_rate', 'user_cate1_day', 'user_day_cate_12', 'diff_user_item', 'user_day_page',
                'user_cate0_day', 'days_1', 'days_2', 'second_1', 'second_2', 'city_cate1_rate', 'mean_user_age',
]
#
#
# def threshold(target, predict, answer):
#     for i in range(10000):
#         gate = i / 10000
#         for j in range(len(predict)):
#             if abs(predict[j] - target[j]) > gate:
#                 predict[j] = 1 - target[j]
#             else:
#                 predict[j] = target[j]
#         print('predict is :\n', predict)
#         print('target is :\n', target)
#         answer[i] = log_loss(predict, target)
#         print(min(answer))
#     return gate


# 加载数据集
def parse(x):
    return datetime.strptime(x, "%Y-%m-%d %H:%M:%S")


def my_loss(target, predict):
    return K.categorical_crossentropy(target, predict)


dataset = pd.read_csv('train_data1.csv', header=0, index_col=0)
# tests = pd.read_csv('test_data.csv', parse_dates=['time'], header=0, index_col=0, date_parser=parse)
# print(tests.shape)


dataset = dataset[use_features]
# test = tests[test_features]
# dataset.values = dataset.values
print('after drop unused feature : \n', dataset.head(5))
print('dataset shape is: ', dataset.shape)
values = dataset.values
# test_values = test.values
# 整数编码
# values[:, 4] = LabelEncoder().fit_transform(values[:, 4])
# ensure all data is float
# print(dataset.info(max_cols=150))
values = values.astype('float64')
# test_values = test_values.astype('float64')
print('values shape is', values.shape)
# print('test_values shape is', test_values.shape)
# print('after type exchange: ', values[:5, :1], end='\t')
# 归一化特征
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
print('scaled shape is', scaled.shape)
# test_scaled = scaler.fit_transform(test_values)
# print('test_scaled shape is', test_scaled.shape)
# # 构建监督学习问题
# reframed = series_to_supervised(scaled, 1, 1)
# print('reframed shape is ', pd.DataFrame(reframed).shape)
# test_reframed = series_to_supervised(test_scaled, 1, 1)
# print('test_reframed shape is ', pd.DataFrame(test_reframed).shape)

# # 丢弃我们并不想预测的列
# reframed.drop(reframed.columns[74:], axis=1, inplace=True)
# test_reframed.drop(test_reframed.columns[73:], axis=1, inplace=True)


# 分割为训练集和测试集
# values = reframed.values
# test_values = test_reframed.values
values = scaled
# pd.DataFrame(values).to_csv('values.csv')
# values = values+0.001
# test_values = test_scaled
train = values[:20000, :]
train_val = values[20000:, :]

# 分为输入输出
train_X, train_y = train[:, 1:], train[:, 0]
train_val_X, train_val_y = train_val[:, 1:], train_val[:, 0]
print('testX shape is ', train_val_X.shape)
# print('test shape is ', test_values.shape)
# 重塑成3D形状 [样例, 时间步, 特征]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
train_val_X = train_val_X.reshape((train_val_X.shape[0], 1, train_val_X.shape[1]))
# test_values = test_values.reshape(test_values.shape[0], 1, test_values.shape[1])

print('testy is', train_val_y)
print(train_X.shape, train_y.shape, train_val_X.shape, train_val_y.shape)
start_time = datetime.now()


model = Sequential()
model.add(GRU(input_shape=(train_X.shape[1], train_X.shape[2]), output_dim=1, return_sequences=True,activation='tanh'))
# model.add()
model.add(GRU(50, return_sequences=False))
model.add(Dense(output_dim=1))
model.compile(loss='mse', optimizer='adam')


early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
history = model.fit(train_X, train_y, epochs=50, batch_size=1280, validation_data=(train_val_X, train_val_y), verbose=2,
                    shuffle=True, callbacks=[early_stopping])
print(datetime.now() - start_time, 'seconds')
# 绘制历史数据
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
# pyplot.show()

# 做出预测
yhat = model.predict(train_val_X)
# print(yhat)
# pd.DataFrame(yhat).to_csv('out.csv')
#
#
# train_val_X = train_val_X.reshape((train_val_X.shape[0], train_val_X.shape[2]))
# # 反向转换预测值比例
# inv_yhat = concatenate((yhat, train_val_X[:, 1:]), axis=1)
# print("/////////////////")
# print(inv_yhat.shape)
# inv_yhat = scaler.inverse_transform(inv_yhat)
# inv_yhat = inv_yhat[:, 0]
#
# # 反向转换实际值比例
# train_val_y = train_val_y.reshape((len(train_val_y), 1))
inv_y = train_val_y
# inv_y = scaler.inverse_transform(inv_y)
# inv_y = inv_y[:, 0]
# print(inv_yhat)
# print(yhat)

# 计算RMSE
# answer = [0 for i in range(10000)]
# threshold = threshold(inv_yhat, inv_y, answer)
# logloss = log_loss(inv_y, inv_yhat)
# print('origin Test log_loss: %.5f' % logloss)
#
# for i in range(len(inv_yhat)):
#     if abs(inv_yhat[i]-inv_y[i]) < 0.99:
#         inv_yhat[i] = inv_y[i]
#         # print("转换完成")

# print('the original answer is ', inv_y)
# print('the predict result is ', inv_yhat)
# pd.DataFrame(inv_yhat).to_csv('predict.csv')
# pd.DataFrame(inv_y).to_csv('target.csv')
logloss = log_loss(inv_y, yhat)
print('Test log_loss: %.5f' % logloss)
# yhat = model.predict(test_values)
pd.DataFrame(yhat).to_csv('second_commition.csv')
# test[['instance_id', 'predicted_score']].to_csv('LSTM_out.txt', index=False, sep=' ')  # 保存在线提交结果
# submission = pd.DataFrame({'instance_id': tests['instance_id'],
#                             'predict_score': yhat})
# print(submission)
# submission.to_csv("submission.csv", index=False)

print(datetime.now()-start_time, 'seconds')
