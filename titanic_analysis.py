import warnings

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn import preprocessing

warnings.filterwarnings('ignore')
#读取数据
train_data = pd.read_csv('D:/file/train.csv')
test_data = pd.read_csv('D:/file/test.csv')
test_labels = pd.read_csv('D:/file/gender_submission.csv')
#数据探索
# print(train_data.info())
# print(train_data.isnull().sum())
# print(test_data.isnull().sum())

#使用平均年龄来填充年龄中的 nan 值
train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)
test_data['Age'].fillna(test_data['Age'].mean(),inplace=True)

#查看票价的取值情况
# print(test_data['Fare'].value_counts())
#使用票价的均值填充票价中的 nan 值
test_data['Fare'].fillna(test_data['Fare'].mean(),inplace=True)

#查看Embarked的取值情况
# print(train_data['Embarked'].value_counts())
#使用登录最多的港口来填充登录港口的 nan 值
train_data['Embarked'].fillna('S', inplace=True)

#选取特征，拆分数据与标签
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
#选取非数值属性的特征
feature1 = ['Sex','Embarked']

train_features = train_data[features]
train_labels = train_data['Survived']
test_features = test_data[features]
test_labels = test_labels['Survived']

#处理非字符属性
le=preprocessing.LabelEncoder()
#处理训练集
def replace1(s):
    le.fit(train_features[s])
    train_features[s] = le.transform(train_features[s])

#处理测试集
def replace2(s):
    le.fit(test_features[s])
    test_features[s] = le.transform(test_features[s])

for i in feature1:
    replace1(i)
    replace2(i)

print(train_features.info())
