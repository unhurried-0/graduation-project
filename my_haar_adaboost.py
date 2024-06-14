# %% 库函数
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
from time import time
import joblib

from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report#导入混淆矩阵对应的库
from sklearn.metrics import confusion_matrix
# sklearn 主要是机器学习库

from skimage.transform import integral_image
from skimage.feature import haar_like_feature


# %% 提取图像内的HAAR特征

def extract_feature_image(img, feature_type, feature_coord=None):
    """Extract the haar feature for the current image"""
    ii = integral_image(img)
    return haar_like_feature(ii, 0, 0, ii.shape[0], ii.shape[1],
                             feature_type=feature_type,
                             feature_coord=feature_coord)

# %% 定义关键参数
images = np.empty((0,64,64))
feature_types = ['type-3-y','type-3-x','type-2-x','type-2-y','type-4'] # 一次提取特征只会用到feature_types中的feature_types[feature_type_idx]，最后再整合
train_num = 6850 # 必须偶数且×0.8是整数
train_num_str = str(train_num/2)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# %% 读取数据集
images = np.load('img_data_all_'+train_num_str+'_.npy', allow_pickle=True)

# %%
t_start = time()

feature_coord_selected = np.load('feature_coord_selected_'+'900.0'+'_all_5_features'+'.npy', allow_pickle=True)
feature_type_selected = np.load('feature_type_selected_'+'900.0'+'_all_5_features'+'.npy', allow_pickle=True)

X = [
    extract_feature_image(img, feature_type_selected, feature_coord_selected)
    for img in images
]
X = np.stack(X)
time_subs_feature_comp = time() - t_start
np.save('haar_feaure_vector_'+train_num_str+'_.npy',X)

y = np.array([1] * int(train_num/2) + [0] * int(train_num/2))
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5,
                                                    random_state=0,
                                                    stratify=y)


'''
estimator_num = range(2,100,2)
scores = []
for i in estimator_num:
        
    t_start = time()
    clf3 = AdaBoostClassifier(estimator=None,algorithm="SAMME",n_estimators=i, learning_rate=1)
    # clf3 = RandomForestClassifier(n_estimators=200, max_depth=None, max_features=None, n_jobs=-1, random_state=0)
    clf3.fit(X_train, y_train)
    time_subs_train = time() - t_start

    Score= clf3.predict(X_test)
    test_score = accuracy_score(y_test,Score)
    scores.append(test_score)
    # auc_subs_features = roc_auc_score(y_test, clf3.predict_proba(X_test)[:, 1])
    # print(classification_report(y_test,Score))
    # print(confusion_matrix(y_test,Score))
    # summary = (f'Computing the full feature set took '
    #             f'{time_subs_feature_comp:.3f}s, plus {time_subs_train:.3f}s '
    #             f'training, for an AUC of {auc_subs_features:.2f}.')

    # print(summary)
    
    # fpr, tpr, thresholds = roc_curve(y_test, clf3.decision_function(X_test))
    # # 使用RocCurveDisplay绘制ROC曲线
    # display = RocCurveDisplay(fpr=fpr,tpr=tpr)
    # display.plot()
    # # 显示图形
    # plt.show()
    print(i)

plt.title("弱学习器个数对AdaBoost的分类准确率的影响", pad=20)
plt.xlabel("弱学习器个数")
plt.ylabel("AdaBoost的分类准确率")
plt.plot(estimator_num, scores)
plt.show()
'''

# %% 测roc 迭代50次
t_start = time()
clf3 = AdaBoostClassifier(estimator=None,algorithm="SAMME",n_estimators=50, learning_rate=1)
clf3.fit(X_train, y_train)
time_subs_train = time() - t_start
joblib.dump(clf3,'haar_adaboost.pkl',compress=0) 

Score= clf3.predict(X_test)
auc_subs_features = roc_auc_score(y_test, clf3.predict_proba(X_test)[:, 1])
print(classification_report(y_test,Score))
print(confusion_matrix(y_test,Score))
summary = (f'Computing the full feature set took '
            f'{time_subs_feature_comp:.3f}s, plus {time_subs_train:.3f}s '
            f'training, for an AUC of {auc_subs_features:.5f}.')

print(summary)

fpr, tpr, thresholds = roc_curve(y_test, clf3.decision_function(X_test))
# 使用RocCurveDisplay绘制ROC曲线
plt.plot(fpr, tpr,color='r')
plt.axis("square")
plt.xlabel("假阳性率")
plt.ylabel("真阳性率")
plt.title("Haar特征-分类器的ROC曲线")
plt.show()

