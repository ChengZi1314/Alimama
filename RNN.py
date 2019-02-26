from math import sqrt
from numpy import concatenate
import numpy as np
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


def mean_abs_logarithmic_error(y_true, y_pred):
    first_log = K.log(K.clip(y_pred, K.epsilon(), np.inf))
    second_log = K.log(1-K.clip(y_pred, K.epsilon(), np.inf))
    return K.mean(K.abs(y_true*first_log + (1-y_true)*second_log), axis=-1)


dataset = pd.read_csv('new_train.csv', header=0, index_col=0)
tests = pd.read_csv('test_data.csv',header=0, index_col=0)
dataset = dataset[use_features]
test = tests[test_features]
print(dataset.info(max_cols=150))
print('after drop unused feature : \n', dataset.head(5))
print('dataset shape is: ', dataset.shape)
values = dataset.values
test_values = test.values
values = values.astype('float64')

# 归一化特征
scaler = MinMaxScaler(feature_range=(0, 1))
values = scaler.fit_transform(values)
test_values = scaler.fit_transform(test_values)

# 分训练集和验证集
train = values[:int(values.shape[0]*0.7), :]
train_val = values[int(values.shape[0]*0.7):, :]

# 分为输入输出
train_X, train_y = train[:, 1:], train[:, 0]
train_val_X, train_val_y = train_val[:, 1:], train_val[:, 0]

# 重塑成3D形状 [样例, 时间步, 特征]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
train_val_X = train_val_X.reshape((train_val_X.shape[0], 1, train_val_X.shape[1]))
test_values = test_values.reshape(test_values.shape[0], 1, test_values.shape[1])

# check一下这整个shape是否匹配
print(train_X.shape, train_y.shape, train_val_X.shape, train_val_y.shape)
start_time = datetime.now()


# 构建模型
model = Sequential()
model.add(GRU(input_shape=(train_X.shape[1], train_X.shape[2]), output_dim=1, return_sequences=True, activation='tanh'))
model.add(GRU(50, return_sequences=False,activation='tanh'))
model.add(Dense(output_dim=1))
model.compile(loss=mean_abs_logarithmic_error, optimizer='RMSprop')
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
history = model.fit(train_X, train_y, epochs=500, batch_size=1280, validation_data=(train_val_X, train_val_y), verbose=2,
                    shuffle=True, callbacks=[early_stopping])
print(datetime.now() - start_time, 'seconds')

# 绘制历史数据
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# 做出预测
yhat = model.predict(train_val_X)
# yhat = abs(yhat)

print(yhat)
print(train_val_y)
logloss = log_loss(train_val_y, yhat)
print(yhat)
print('Test log_loss: %.5f' % logloss)
yhat = model.predict(test_values)
pd.DataFrame(yhat).to_csv('new_commition.csv')


print(datetime.now()-start_time, 'seconds')
