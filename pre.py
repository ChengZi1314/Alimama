import pandas as pd

#
# data = pd.read_csv(r'E:\first_commition1.txt', sep=' ')
# data2 = pd.read_csv(r'E:\123.txt',sep=',')
#
# result = pd.concat([data, data2], axis=1)
# print(result)
# # result = result.drop([1],inplace = True)
# del result['instance_id1']
# del result['predicted1_score']
# result['predicted_score'] = round(result['predicted_score'],9)
# print(result)
# pd.DataFrame(result).to_csv('commition.txt',sep = ' ',index=None)
data = pd.read_csv('D:\Alimama\cha.txt', sep=' ')
data['predicted_score'] = round(abs(data['predicted_score']), 9)
cache = data.sort_values(['predicted_score'], ascending=True)
print(cache)
print(cache.shape)
gate = cache['predicted_score'].values[int(1209768/20)]
print(gate)
for i in range(1209768):
    if data['predicted_score'].values[i] < gate:
        data['predicted_score'].values[i] = 0.0000001
        # print(data[])
print('done!')

pd.DataFrame(data).to_csv('hey.txt', index=None, sep=' ')
