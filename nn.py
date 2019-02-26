#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 21:29:21 2018
@author: flyaway
"""
import tensorflow as tf
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder,StandardScaler,LabelEncoder
import numpy as np
from sklearn.metrics import accuracy_score,log_loss

import pandas as pd
import keras.backend as K


import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class DeepAndCross(object):
	def __init__(self, train, test,cate_feat, num_feat):
		self.train = train
		self.cate_feat = cate_feat
		self.num_feat = num_feat
		self.test = test
		self.all_data = pd.concat([train, test])
	
	# 产生embedding
	def embedding_input(self, inp, name, n_in, n_out):
		with tf.variable_scope(name) as scope:
			embeddings = tf.Variable(tf.random_uniform([n_in,n_out],-1.0,1.0))
		return inp,tf.nn.embedding_lookup(embeddings, inp, name = scope.name)

	# 产生embedding特征，针对类目特征产生embedding
	def embedding_feature_generate(self,cate_values,num_cate):
		embeddings_tensors = []
		# 获取类目特征个数
		col = cate_values.get_shape()[1]
		for i in range(col):
			layer_name = 'inp_' + str(i)
			nunique = num_cate[i]
			embed_dim = nunique if int(6 * np.power(nunique, 1/4)) > nunique else int(6 * np.power(nunique, 1/4))
			t_inp, t_build = self.embedding_input(cate_values[:, i], layer_name, nunique, embed_dim)
			embeddings_tensors.append((t_inp, t_build))
			del(t_inp, t_build)
		inp_embed = [et[1] for et in embeddings_tensors]
		return inp_embed

	def fclayer(self,x,output_dim,reluFlag,name):
		with tf.variable_scope(name) as scope:
			input_dim = x.get_shape().as_list()[1]
			W = tf.Variable(tf.random_normal([input_dim,output_dim], stddev=0.01))
			b = tf.Variable(tf.random_normal([output_dim], stddev=0.01))
			out = tf.nn.xw_plus_b(x, W, b, name=scope.name)
			if 'relu' == reluFlag:
				return tf.nn.relu(out)
			if 'exp' == reluFlag:
				return 1/(1+tf.exp(out, name=scope.name))
			else:
				return out

	def crosslayer(self, x, inp_embed, name):
		with tf.variable_scope(name) as scope:
			input_dim = x.get_shape().as_list()[1]
			w = tf.Variable(tf.random_normal([1, input_dim], stddev=0.01))
			b = tf.Variable(tf.random_normal([1, input_dim], stddev=0.01))
			tmp1 = K.batch_dot(K.reshape(x, (-1, input_dim, 1)), tf.reshape(inp_embed,(-1,1,input_dim)))
			tmp = K.sum(w * tmp1, 1, keepdims=True)
			tmp = tf.reshape(tmp, shape=(-1, input_dim))
			output = tf.add(tmp, b)
			output = tf.add(output, inp_embed)
			return output

	def build_model(self, X_cate, X_cont, num_cate):
		inp_embed = self.embedding_feature_generate(X_cate,num_cate)
		inp_embed = tf.concat(inp_embed, 1)
		input_dim = inp_embed.get_shape().as_list()
		inp_embed = tf.concat([inp_embed], 1)
		
		# 类目特征用两层交叉网络
		cross1 = self.crosslayer(inp_embed, inp_embed, name='cross1')
		cross2 = self.crosslayer(cross1, inp_embed, name='cross2')

		# 数值特征用全连接网络, 这里参数你可以自己修改,每层到底多少个元素
		# 还有就是每层的全连接可以采用dropout来降低, 如果不想用就将他注释掉
		inp_cond = tf.convert_to_tensor(X_cont)
		fc1 = self.fclayer(inp_cond, 272, reluFlag='relu',name = 'fc_1')
		fc1 = tf.nn.dropout(fc1, keep_prob=0.5)
		fc2 = self.fclayer(fc1, 272, reluFlag='relu',name = 'fc_2')
		fc2 = tf.nn.dropout(fc2, keep_prob=0.5)
		fc3 = self.fclayer(fc2, 272, reluFlag='relu',name = 'fc_3')
		fc3 = tf.nn.dropout(fc3, keep_prob=0.5)

		# 最后将类目特征的结果和数值特征的结果全连接
		output = tf.concat([fc3, cross2], 1)
		out = self.fclayer(output, 1, reluFlag='exp', name='out')
		return out

	def fit(self):
		cate_values = self.all_data[self.cate_feat]
		cont_values = self.all_data[self.num_feat].values
		X_train_cate = self.train[self.cate_feat]
		X_test_cate = self.test[self.cate_feat]

		# 对类型数据进行数值变化
		lb = LabelEncoder()
		for i in self.cate_feat:
			tmp = lb.fit_transform(list(self.all_data[i]))
			X_train_cate[i] = lb.transform(list(X_train_cate[i]))
			X_test_cate[i] = lb.transform(list(X_test_cate[i]))
			cate_values[i] = lb.transform(list(cate_values[i]))

		features = [
            'instance_id', 'item_brand_id', 'item_city_id', 'item_price_level', 'item_sales_level',
            'item_collected_level', 'item_pv_level', 'user_gender_id', 'user_age_level', 'user_star_level',
            'context_page_id', 'shop_review_num_level', 'shop_review_positive_rate', 'shop_star_level',
            'shop_score_service', 'shop_score_delivery', 'shop_score_description', 'category_0', 'category_1',
            'category_2', 'predict_category_0', 'predict_category_2', 'hour', 'days_1', 'days_2', 'second_1',
            'second_2', 'low_number', 'item_id_sum', 'item_id_rank', 'shop_id_sum', 'shop_id_rank', 'user_item_sum',
            'occup_prices_rate', 'occup_rate', 'page_rate', 'pv_rate', 'sales_level_rate', 'star_price_rate',
            'age_page_rate', 'occup_page_rate', 'occip_gender_page_rate', 'occup_cate_rate', 'occup_brand_rate',
            'occup_city_rate', 'occup_hour_rate', 'age_cate_rate', 'age_brand_rate', 'age_hour_rate', 'city_cate1_rate',
            'is_last', 'dayclick_times', 'user_day_cate_1', 'user_shop_day', 'diff_user_item', 'user_cate1_day',
            'user_cate0_day', 'user_day_cate_12', 'user_day_page', 'mean_item_prices', 'shop_mean_price',
            'mean_user_age', 'mean_brand_prices', 'mean_price_age', 'mean_brand_age'
        ]


		# 对数值变化采用最大最小变化，也可以采用均值最大最小化
		X_train_cont = self.train[self.num_feat]
		X_test_cont = self.test[features]
		print(X_train_cont.shape)
		print(X_test_cont.shape)
		min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
		X_train_cont = min_max_scaler.fit_transform(X_train_cont)
		X_test_cont = min_max_scaler.fit_transform(X_test_cont)

		cate_values = cate_values.values
		num_cate = []
		for i in range(cate_values.shape[1]):   
			nunique = np.unique(cate_values[:, i]).shape[0]
			num_cate.append(nunique)  

		X_cate = tf.placeholder(tf.int32, [None, cate_values.shape[1]])
		X_cont = tf.placeholder("float", [None, cont_values.shape[1]])
		Y = tf.placeholder("float", [None, 1])
		Y_temp = tf.placeholder("float", [None, 1])

		output = self.build_model(X_cate,X_cont,num_cate)
		# y_list = Y.eval(session=tf.Session())
		# log_los = log_loss(list(Y), list(output))
		cross_entropy = -1 * tf.where(tf.greater(Y, Y_temp),
			tf.log(tf.clip_by_value(output,1e-10,1.0)), tf.log(tf.clip_by_value(1 - output,1e-10,1.0)))
		# entropy=tf.nn.softmax_cross_entropy_with_logits(logits=output,labels=Y)
		cost = tf.reduce_mean(cross_entropy)
		train_op = tf.train.AdamOptimizer(0.001).minimize(cost)
		predict_op = tf.arg_max(output, 1)
		
		y_train = train[['is_trade']]
		y_train_temp = y_train.copy()
		y_train_temp['is_trade'] = 0
		
		# y_test = test[['is_trade']]
		# y_test_temp = y_test.copy()
		# y_test_temp['is_trade'] = 0

		# 这个地方你需要好好写一下，我这里写的比较乱，模型跑完记得要保存，这些你都可以自己去百度写一下
		# 迭代次数和 batch要怎么循环你可以百度一下，看别人怎么写
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			for i in range(3):
				lost = []
				for start, end in zip(range(0, len(X_train_cate), 256), range(256, len(X_train_cate)+1, 256)):
					print(start)
					tt, output1, Y1 = sess.run([train_op, output, cross_entropy], feed_dict={X_cate: X_train_cate[start:end], X_cont: X_train_cont[start:end],Y:y_train[start:end], Y_temp:y_train_temp[start:end]})
					#lost.append(sess.run(cost,feed_dict={X_cate: X_train_cate[start:end], X_cont: X_train_cont[start:end],Y:y_train[start:end],Y_temp:y_train_temp[start:end]}))
			for start, end in zip(range(0, len(X_test_cate), 256), range(256, len(X_test_cate)+1, 256)):
				lost.extend(sess.run(output,feed_dict={X_cate: X_test_cate[start:end], X_cont: X_test_cont[start:end]}))
			# print(np.mean(lost))
			# test['predicted_score'] = lost
			# test[['instance_id', 'predicted_score']].to_csv('baseline.csv', index=False,sep=' ')
			# test['rate'] = tt
			# test.to_csv('test.csv',index = None)

def getFeature(data):
	not_used = ['age_brand_sum', 'city_cate1_sum', 'context_timestamp', 'age_hour_sum', 'age_cate_count',
				'occip_gender_page_sum', 'is_low2', 'age_page_sum', 'age_hour_count', 'occup_city_sum', 'max_time',
				'star_price_sum', 'max_occup', 'mean3', 'star_price_count', 'user_item_rank', 'age_brand_count',
				'city_cate1_count', 'occup_cate_sum', 'day_y', 'cate0_item_rank', 'is_low1', 'pv_sum', 'user_occupation_id',
				'occup_prices_sum', 'occup_brand_count', 'is_trade.1', 'page_count', 'pv_count', 'context_id', 'good_review',
				'occup_page_sum', 'mean2', 'predict_category_property', 'occup_brand_sum', 'occup_page_count', 'day_x',
				'occup_hour_count', 'user_item_times', 'occip_gender_page_count', 'occup_hour_sum', 'user_day_cate_0',
				'occup_prices_count', 'sales_level_count', 'time.1', 'age_cate_sum', 'sales_level_sum', 'is_low3',
				'occup_cate_count', 'occup_city_count', 'cate0_item_sum', 'occup_sum', 'age_page_count', 'page_sum', 'mean1',
				'time_rank', 'occup_count'
    ]
	not_used_feature = ['category_0','category_2','user_id','index', 'day',
                        'is_trade','context_timestamp','max_time','min_time','all_max_time',
                        'date','predict_category_0','time',
                        'predict_category_1','predict_category_2','is_trade.1', 'good_review', 'max_occup',
                        'category_1', 'predict_category_property', 'property_0', 'property_1', 'property_2']
	id_feat = ['user_id', 'item_id' ,'predict_category_1', 'shop_id', 'property_0', 'property_1', 'property_2']
	temp = [f for f in data.columns if f not in (not_used)]
	num_feat = [f for f in temp if f not in (id_feat)]
	print(num_feat)
	return num_feat, id_feat

if __name__ == "__main__":
	train = pd.read_csv('train_data1.csv',index_col=0)
	test = pd.read_csv('test_data1.csv',index_col=0)

	num_feat, id_feat = getFeature(train)
	# print (num_feat)
	# train = train.loc[:len(train)/100,:]
	# print (num_feat)
	# num_feat = ['shop_age_cha', 'item_price_cha', 'item_sales_cha', 'item_collected_cha', 'item_pv_cha', 'item_age_abs'
	# , 'brand_age_abs', 'category_1_age_abs', 'shop_age_abs', 'item_price_abs', 'item_sales_abs', 'item_collected_abs',
	#  'item_pv_abs', 'user_count', 'user_sum', 'user_rate', 'user_hour_count', 'user_hour_sum', 'user_hour_rate',
	# 'gender_count', 'gender_sum', 'gender_rate', 'gender_hour_count', 'gender_hour_sum', 'gender_hour_rate',
	#  'age_count', 'age_sum', 'age_rate', 'age_hour_count', 'age_hour_sum', 'age_hour_rate', 'occupation_count',
	# 'occupation_sum', 'occupation_rate', 'occupation_hour_count', 'occupation_hour_sum', 'occupation_hour_rate',
	# 'star_count', 'star_sum', 'star_rate', 'star_hour_count', 'star_hour_sum', 'star_hour_rate', 'item_count',
	# 'item_sum', 'item_rate', 'item_hour_count', 'item_hour_sum', 'item_hour_rate', 'brand_count', 'brand_sum',
	# 'brand_rate', 'brand_hour_count', 'brand_hour_sum', 'brand_hour_rate', 'brand_price_count', 'brand_price_sum',
	# 'brand_price_rate', 'brand_city_count', 'brand_city_sum', 'brand_city_rate', 'price_count', 'price_sum',
	# 'price_rate', 'item_city_count', 'item_city_sum', 'item_city_rate', 'item_city_hour_count', 'item_city_hour_sum',
	# 'item_city_hour_rate', 'collected_count', 'collected_sum', 'collected_rate', 'collected_brand_count',
	# 'collected_brand_sum', 'collected_brand_rate', 'sales_count', 'sales_sum', 'sales_rate', 'page_count', 'page_sum',
	#  'page_rate', 'review_count', 'review_sum', 'review_rate', 'user_shop_count', 'user_shop_sum', 'user_shop_rate',
	# 'user_star_count', 'user_star_sum', 'user_star_rate', 'occup_cate_count', 'occup_cate_sum', 'occup_cate_rate',
	# 'occup_brand_count', 'occup_brand_sum', 'occup_brand_rate', 'occup_city_count', 'occup_city_sum',
	# 'occup_city_rate', 'occup_price_count', 'occup_price_sum', 'occup_price_rate', 'occup_collected_count',
	# 'occup_collected_sum', 'occup_collected_rate', 'age_cate_count', 'age_cate_sum', 'age_cate_rate',
	# 'age_brand_count', 'age_brand_sum', 'age_brand_rate', 'age_price_count', 'age_price_sum', 'age_price_rate',
	# 'age_city_count', 'age_city_sum', 'age_city_rate', 'user_collected_count', 'user_collected_sum',
	# 'user_collected_rate', 'user_category_count', 'user_category_sum', 'user_category_rate']
	model = DeepAndCross(train, test, id_feat, num_feat)
	model.fit()
