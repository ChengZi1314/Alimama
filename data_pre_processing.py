import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import log_loss
# from keras import backend as K
from copy import copy
# def parse(x):
#     return datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
#
#
# original_data = pd.read_csv('pre_data.csv', parse_dates=['time'], header=0, index_col=0, date_parser=parse)
# print(original_data.head(5))
# original_data['index'] = [i for i in range(original_data.shape[0])]
# original_data.drop(original_data.columns[1], axis=1, inplace=True)
# original_data['time'] = pd.to_datetime(pd.Series(original_data['time']))
# print(pd.DataFrame(original_data.info(max_cols=150)))
# original_data['time'] =
# time = original_data['time']
# print(pd.DataFrame(original_data).drop(original_data['time']))
# extime = original_data.drop(original_data['time'], axis=1, inplace=True)
# data = np.concatenate((time, extime), axis=1)


# def threshold(inv_yhat, inv_y, answer):
#     logloss = log_loss(inv_y, inv_yhat)
#     print('the best Test log_loss is: %.5f' % logloss)
#     for j in range(1,1000):
#
#         tar = copy(inv_y)
#         pre = copy(inv_yhat)
#         gate = j/1000
#         print('the %dth gate is: ' % j, gate)
#         for i in range(len(pre)):
#             if abs(pre[i]-tar[i]) < gate:
#                 pre[i] = tar[i]
#         logloss = log_loss(tar, pre)
#         print('Test log_loss: %.5f' % logloss)
#         answer[j] = logloss
#         if answer[j] == answer[j-1]:
#             break
#         gate = j/1000
#     print('the best gate is: ', gate)
#     for i in range(len(inv_yhat)):
#         if abs(inv_yhat[i] - inv_y[i]) < gate:
#             inv_yhat[i] = inv_y[i]
#     logloss = log_loss(inv_y, inv_yhat)
#     print('the best Test log_loss is: %.5f' % logloss)
#
# answer = [0 for i in range(1000)]
# predict = pd.read_csv('predict.csv', header=0, index_col=0)
# target = pd.read_csv('target.csv', index_col=0, header=0)
# predict = predict.values
# target = target.values
# threshold(predict,target, answer)

#
# from XGB_Alimama import *
# data = pd.read_csv(r'~/Alimama/round2_train.txt', sep=' ', nrows=100000)
# print("data read success!")
# print(data.shape)
# lenth = data.shape[0]
# subdata = []
# pre_data = []
# for i in range(10):
#     start = int(lenth*i/10)
#     end = int(lenth*(i+1)/10)
#     print(end)
#     subdata = data.iloc[start:end, :]
#     print('subdata[%d] is processing' % i)
#     pre_data = data_preprocessing(subdata)
#     print('subdata[%d] process complete' % i)
#     pd.DataFrame(pre_data[i]).to_csv('data%d.csv'%i,index=None)
#     print('subdata[%d] write complete' % i)
#     del subdata, pre_data

# data = pd.read_csv(r'~/Alimama/round1_ijcai_18_train_20180301.txt', sep=' ')
data = pd.read_csv(r'D:\Alimama\hey.txt', sep=' ')
# data.to_csv('data1.txt',sep=',',index=None)
for i in range(10):
    pre = data.iloc[:, (i*data.shape[0]/10):((i+1)*data.shape[0]/10)]
    pd.DataFrame(pre).to_csv('data[%d].csv'%i, sep=' ')
print('done!')


# pre = pd.read_csv(r'~/Alimama/round1_ijcai_18_train_20180301.txt', sep=' ')
#
# print(set(data))
# print(set(pre))

# original = pd.read_csv(r'~/Alimama/round1_ijcai_18_train_20180301.csv', sep=' ')
# test = pd.read_table(r'~/Alimama/first_commition1.txt', sep=' ')
# test = pd.read_table(r'~/Alimama/second_commition.txt', sep=' ')
# pre = pd.read_table(r'~/Alimama/first_commition.txt', sep=' ')
# # print("read completed!")
# # print(test)
# # # print(original['item_category_list'])
# # values =
# # values = pre['predicted_score'].values
# # values = values.astype('float32')
# # # pre['predicted_score'].values = values
# # for i in range(pre.shape[0]):
# #     pre['predicted_score'].values[i] = round(pre['predicted_score'].values[i],9)
# # print(pre.info())
# print(test.info())
# print(pre.info())
# print(test['instance_id'])
# print(test['predicted_score'])
# # pre.to_csv('first_commition1.txt', sep=' ')
# # print(pre)
#
# # pd.DataFrame(first).to_csv('first.csv',index=None)
#
# fw = open('second_commition.txt', 'w')
# with open('round2_ijcai_18_test_a_20180425.txt') as pred_file:
#     fw.write('{} {}\n'.format('instance_id', 'predicted_score'))
#     for line in pred_file.readlines()[1:]:
#         splits = line.strip().split(' ')
#         fw.write('{} {}\n'.format(splits[0], 0.0))
