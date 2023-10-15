import pandas as pd
import warnings
from sklearn import preprocessing, linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score

warnings.filterwarnings('ignore')



#读取数据
train_data = pd.read_csv('D:/file/train.csv')
test_data = pd.read_csv('D:/file/test.csv')
test_labels = pd.read_csv('D:/file/gender_submission.csv')
#数据探索
print(train_data.info())
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

#分类器评价函数
def evaluate_classifier( real_label_list,predict_label_list):
    """
       return Precision, Recall and ConfusionMatrix
       Input : predict_label_list,real_label_list
    """
    msg=''
    Confusion_matrix = confusion_matrix( real_label_list,predict_label_list)
    msg += '\n Confusion Matrix\n ' + str(Confusion_matrix)
    accuracy=accuracy_score(real_label_list,predict_label_list)
    precision = precision_score(real_label_list,predict_label_list, average=None)
    recall = recall_score(real_label_list,predict_label_list, average=None)
    msg += '\n Accuracy =%s' %str(accuracy)
    msg += '\n Precision of tag 0 and 1 =%s' %str(precision)
    msg += '\n Recall of tag 0 and 1 =%s' %str(recall)
    msg += '\n f1_score of tag 0 =%s' % str(2*(precision[0]*recall[0])/(precision[0]+recall[0]))
    msg += '\n f1_score of tag 1 =%s' % str(2 * (precision[1] * recall[1]) / (precision[1] + recall[1]))
    return msg

#DecisionTree
clf_dec=DecisionTreeClassifier(criterion='gini',splitter='best',max_depth=5)       #生成模型
clf_dec.fit(train_features,train_labels)                        #训练模型
pred_dec=clf_dec.predict(test_features)                  #预测结果
eval_dec=evaluate_classifier(test_labels,pred_dec)    #模型评价
print('DecisionTree:'+eval_dec)
print('')

#RandomForest
clf_ran=RandomForestClassifier(n_estimators=10,criterion='gini',max_depth=None,n_jobs=2)
clf_ran.fit(train_features,train_labels)
pred_ran=clf_ran.predict(test_features)
eval_ran=evaluate_classifier(test_labels,pred_ran)
print('RandomForest:'+eval_ran)
print('')

#KNN
clf_knn=KNeighborsClassifier(n_neighbors=5,weights='uniform',algorithm='auto',n_jobs=2)
clf_knn.fit(train_features,train_labels)
pred_knn=clf_knn.predict(test_features)
eval_knn=evaluate_classifier(test_labels,pred_knn)
print('KNN:'+eval_knn)
print('')

#LogisticRegression
clf_log=linear_model.LogisticRegression(C=1.0)
clf_log.fit(train_features,train_labels)
pred_log=clf_log.predict(test_features)
eval_log=evaluate_classifier(test_labels,pred_log)
print('LogisticRegression:'+eval_log)
print('')

#VotingClassifier
clf_vot=VotingClassifier(estimators=[('dec',clf_dec),('ada',clf_log),
                                    ('ran',clf_ran),('knn',clf_knn),
                                     ],
                        voting='soft',
                         weights=[5,1,5,1])
clf_vot.fit(train_features,train_labels)
pred_vot=clf_vot.predict(test_features)
eval_vot=evaluate_classifier(test_labels,pred_vot)
print('VotingClassifier:'+eval_vot)
print('')

