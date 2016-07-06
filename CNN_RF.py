#encoding=utf8
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

__author__ = 'jdwang'
__date__ = 'create date: 2016-05-29'
import numpy as np
import logging
import timeit
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import  RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
import pandas as pd


config = yaml.load(file('./config.yaml'))	#读取yaml配置文件
config = config['main']						#以字典的方式读取2
logging.basicConfig(filename=''.join(config['log_file_path']), filemode='w',
                    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
start_time = timeit.default_timer()

#可保存为日志文件进行管理


import jieba
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

count = 0
list_cover = []
replace = []
list_target = []
def cover(x):
    global count
    #此时的x仅是unicode
    if x == 'other' or x == 'INCONSISTENT' or x == 'EQUAL':
         print "before: ",x
         x = list_cover[count]
         print "after: ",x
         count += 1
         replace.append(x)
         return x
    else:
         count += 1
         replace.append(x)
         return x

def TargetToNumber(target):
    if target == u'IphoneSE':
        list_target.append(2)
    elif target == u'春节放鞭炮':
        list_target.append(1)
    elif target == u'俄罗斯在叙利亚的反恐行动':
        list_target.append(0)
    elif target == u'开放二胎':
        list_target.append(-1)
    elif target == u'深圳禁摩限电':
        list_target.append(-2)
    return target

def appendData(data1,data2):
    for i in range(len(data1)):
            data1[i] = list(data1[i])   #每一个类型都要转换 才能成功加入
            print "Before:",len(data1[i])
            for m in data2[i]:
                data1[i].append(m)
            print "after:",len(data1[i])
    return data1

train_data = pd.read_csv(
    config['train_data_file_path'],
    sep='\t',
    encoding='utf8',
    header=0
)

test_data = pd.read_csv(
    config['test_data_file_path'],
    sep='\t',
    encoding='utf8',
    header=0
)

step2_data = pd.read_csv(
    config['step2_file_path'],
    sep='\t',
    encoding='utf8',
    header=0
)

vect = CountVectorizer()
train_data_bow_fea = list(vect.fit_transform(train_data['WORDS']).toarray())
test_data_bow_fea = list(vect.transform(test_data['WORDS']).toarray())

print type(train_data_bow_fea)

# cnn_feature_train = np.loadtxt('train_CNN_feature.npy',delimiter=',')
# cnn_feature_test = np.loadtxt('test_CNN_feature.npy',delimiter=',')

cnn_feature_train = np.loadtxt('train_cnn_feature_300d.npy',delimiter=',')
cnn_feature_test = np.loadtxt('test_cnn_feature_300d.npy',delimiter=',')



train_data_bow_fea = appendData(train_data_bow_fea,cnn_feature_train)
test_data_bow_fea = appendData(test_data_bow_fea,cnn_feature_test)

#Target变数字
# test_data['TARGET'].apply(TargetToNumber)
# test_data['TARGET'] = list_target
# # test_data_bow_fea = appendData(test_data_bow_fea,list_target)
#
# list_target = []
# train_data['TARGET'].apply(TargetToNumber)
# train_data['TARGET'] = list_target
# train_data_bow_fea = appendData(train_data_bow_fea,list_target) #拼接list可以更优雅



#两阶段法
#1、最好的RF
#2、+4个标准


test_data['WORDS'] = test_data_bow_fea


print '计算最大熵模型'

print "Training MaxEnt"
clf  = RandomForestClassifier(n_estimators=1000) #随机森林
# clf = MultinomialNB()       #朴素贝叶斯
# clf = LogisticRegression(multi_class="multinomial",solver="newton-cg") #最大熵
# clf = LogisticRegression()  #逻辑斯蒂回归

clf.fit(train_data_bow_fea,train_data['STANCE'])
list_cover = clf.predict(test_data_bow_fea)


test_data['PREDICT'] = step2_data['PREDICT']    #优先cue-phrase
test_data['PREDICT'].apply(cover)   #对每一个进行处理，不能处理的上全全
test_data['PREDICT'] = replace      #此为原始替换方法，更加pandas/python的方法待考证

zero = []
for i in range(len(test_data_bow_fea)):
    zero.append(0)
#函数式编程，一切应用函数进行解决
# test_data.drop(['WORDS'],axis=1)        #axis=1 drop 全部行
test_data['WORDS'] = zero

test_data.to_csv("1.csv")
print  clf.predict(test_data_bow_fea)
print  sum(test_data['STANCE'] == test_data['PREDICT'])/(1.0 * len(test_data['STANCE']))

