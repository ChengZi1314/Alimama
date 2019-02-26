fw = open(r'E:\123.txt', 'w')
with open(r'E:\round2_ijcai_18_test_a_20180425.txt') as pred_file:
    fw.write('{} {}\n'.format('instance_id', 'predicted_score'))
    for line in pred_file.readlines()[1:]:
        splits = line.strip().split(' ')
        fw.write('{} {}\n'.format(splits[0], 0.0))