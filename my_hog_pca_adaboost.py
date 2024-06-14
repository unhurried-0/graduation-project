# %% 库函数
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
from time import time

from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay  
from sklearn.metrics import classification_report#导入混淆矩阵对应的库
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import joblib

# sklearn 主要是机器学习库

from skimage.transform import integral_image
from skimage.feature import hog


# %% 提取图像内的hog特征

def extract_feature_image(img):
    """Extract the hog feature for the current image"""
    ii = integral_image(img)
    fig_vector, hog_image = hog(
    img,
    orientations=9,
    pixels_per_cell=(8 , 8),
    cells_per_block=(1 , 1),
    visualize=True,
    channel_axis=None,)
    return fig_vector

 

# %% 定义关键参数
images = np.empty((0,64,64))
train_num = 6850  # 必须偶数且×0.8是整数
train_num_str = str(train_num/2)
pca_dimension = 203
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# %% 读取数据集
images = np.load('img_data_all_'+train_num_str+'_.npy', allow_pickle=True)


# %% 提取特征并PCA，分割训练集测试集
t_start = time()
X_long = [
    extract_feature_image(img)
    for img in images
]
# 上面的x只是一个train_num个特征向量构成的列表，不是numpy数组。之前有程序下面一行是numpy.stack,没啥用
X_long = np.array(X_long)
pca = PCA(n_components=pca_dimension)
pca.fit(X_long)
X = pca.transform(X_long)
time_subs_feature_comp = time() - t_start
np.save('hog_feaure_vector_after_pca_'+train_num_str+'_.npy',X)
joblib.dump(pca,'hog_pca.pkl',compress=0) 

y = np.array([1] * int(train_num/2) + [0] * int(train_num/2))
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
                                                    random_state=0,
                                                    stratify=y)

'''
#  由于adaboost准确率过于离谱，换一个测试数据，换成middle close
images_new = np.empty((0,64,64))
i = 0
file_pathname = './data/OwnCollection/non-vehicles/MiddleClose'
for filename in os.listdir(file_pathname): # listdir返回指定的文件夹包含的文件，或包含的文件夹的名字的列表
    img = imread(file_pathname+'/'+filename)
    # 下面第一行是将RGB转成单通道灰度图，第二步是将单通道灰度图转成3通道灰度图
    img = rgb2gray(img)
    x = img[np.newaxis, :]
    images_new = np.vstack((images_new, x))
    i = i + 1
    if i >= train_num/2:
        print("all non-vehicle middleclose image loaded")
        break

X_newtest_long = [
    extract_feature_image(img)
    for img in images_new
]
X_newtest_long = np.array(X_newtest_long)
X_newtest = pca.transform(X_newtest_long)
y_newtest =  np.array([-1] * int(train_num/2))


问题已解决，AUC太高就是训练数据太单调，用训练得到的adaboost直接预测middleclose的数据，几乎不可拟合
'''

# %%

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
    print(i)

plt.title("弱学习器个数对AdaBoost的分类准确率的影响", pad=20)
plt.xlabel("弱学习器个数")
plt.ylabel("AdaBoost的分类准确率")
plt.plot(estimator_num, scores)
plt.show()
'''

# %% 测roc 迭代40次
t_start = time()
clf3 = AdaBoostClassifier(estimator=None,algorithm="SAMME",n_estimators=40, learning_rate=1)
clf3.fit(X_train, y_train)
time_subs_train = time() - t_start
joblib.dump(clf3,'hog_pca_adaboost.pkl',compress=0) 

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
plt.title("HOG特征-分类器的ROC曲线")
plt.show()