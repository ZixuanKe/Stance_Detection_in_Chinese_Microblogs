import numpy as np

import pandas as pd


orgin_data = pd.read_csv(
    'TaskA_all_testdata_14966.csv',
    sep='\t',
    encoding='utf8',
    header=0
)

compare_data = pd.read_csv(
    '1.csv',
    sep=',',
    encoding='utf8',
    header=0
)

orgin_data['PREDICT'] = orgin_data['PREDICT'].fillna(0)
# print test_data['PREDICT']
orgin_data = orgin_data[ orgin_data['PREDICT']!= 0]
print orgin_data[u'ID']


# compare_data = compare_data[compare_data['ID'] == orgin_data['ID']]

print compare_data