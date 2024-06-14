# %% 库函数
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
from time import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, RocCurveDisplay
# sklearn 主要是机器学习库

from skimage.transform import integral_image
from skimage.feature import haar_like_feature
from skimage.feature import draw_haar_like_feature

# %% 提取图像内的HAAR特征
counts = 0
en_count = 0

def extract_feature_image(img, feature_type, feature_coord=None):
    """Extract the haar feature for the current image"""
    global counts
    ii = integral_image(img)
    counts = counts + 1
    if en_count == 1:
        print("%d "%counts)
    return haar_like_feature(ii, 0, 0, ii.shape[0], ii.shape[1],
                             feature_type=feature_type,
                             feature_coord=feature_coord)

# %% 定义关键参数
images = np.empty((0,64,64))
feature_types = ['type-3-y','type-3-x','type-2-x','type-2-y','type-4'] # 一次提取特征只会用到feature_types中的feature_types[feature_type_idx]，最后再整合
train_num = 1800 # 必须偶数且×0.8是整数
train_num_str = str(train_num/2)

# %% 读取数据集
file_pathname = './data/OwnCollection/vehicles/Far'
i = 0
for filename in os.listdir(file_pathname): # listdir返回指定的文件夹包含的文件，或包含的文件夹的名字的列表
    img = imread(file_pathname+'/'+filename)
    # 下面第一行是将RGB转成单通道灰度图，第二步是将单通道灰度图转成3通道灰度图
    img = rgb2gray(img)
    x = img[np.newaxis, :]
    images = np.vstack((images, x))
    i = i + 1
    if i == train_num/2:
        print("all vehicle image loaded")
        break

i = 0
file_pathname = './data/OwnCollection/non-vehicles/Far'
for filename in os.listdir(file_pathname): # listdir返回指定的文件夹包含的文件，或包含的文件夹的名字的列表
    img = imread(file_pathname+'/'+filename)
    # 下面第一行是将RGB转成单通道灰度图，第二步是将单通道灰度图转成3通道灰度图
    img = rgb2gray(img)
    x = img[np.newaxis, :]
    images = np.vstack((images, x))
    i = i + 1
    if i == train_num/2:
        print("all non-vehicle image loaded")
        break


# %% 提取特征
train_num_str_last='100.0' #上一次训练随机森林模型只用了200数据嘛

feature_type_sel = np.load('feature_type_selected_'+train_num_str_last+'_'+feature_types[0]+'.npy', allow_pickle=True)
feature_coord_sel = np.load('feature_coord_selected_'+train_num_str_last+'_'+feature_types[0]+'.npy', allow_pickle=True)

i = 1
while i <=4 :
    feature_type_sel = np.concatenate((feature_type_sel,np.load('feature_type_selected_'+train_num_str_last+'_'+feature_types[i]+'.npy', allow_pickle=True)))
    feature_coord_sel = np.concatenate((feature_coord_sel,np.load('feature_coord_selected_'+train_num_str_last+'_'+feature_types[i]+'.npy', allow_pickle=True)))
    i = i + 1


t_start = time()
X = [
    extract_feature_image(img, feature_type_sel, feature_coord_sel)
    for img in images
]
X = np.stack(X)
print('all feature got!')
time_subs_feature_comp = time() - t_start

y = np.array([1] *  int(train_num/2) + [0] * int(train_num/2))
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75,
                                                    random_state=0,
                                                    stratify=y)

# %% 训练新的随机森林

clf2 = RandomForestClassifier(n_estimators=200, max_depth=None, max_features=None, n_jobs=-1, random_state=0)
#clf2 = AdaBoostClassifier(estimator=None,algorithm="SAMME",n_estimators=200, learning_rate=0.8)
t_start = time()
clf2.fit(X_train, y_train)
time_subs_train = time() - t_start

auc_subs_features = roc_auc_score(y_test, clf2.predict_proba(X_test)[:, 1])

summary = (f'Computing the full feature set took '
            f'Computing the restricted feature set took '
            f'{time_subs_feature_comp:.3f}s, plus {time_subs_train:.3f}s '
            f'training, for an AUC of {auc_subs_features:.2f}.')

print(summary)

# %% 重新选择好的特征
idx_sorted = np.argsort(clf2.feature_importances_)[::-1]
# idx_sorted应该是特征重要性从大到小排，从而得到的特征序号feature_importances_[idx_sorted[0]]就是最重要的那个特征的权重,[,,-1]是倒序选取的意思，argsort得到的是从小到大
feature_importances = clf2.feature_importances_
feature_importances_sorted = feature_importances[idx_sorted]
fig, axes = plt.subplots(4,5)
for idx, ax in enumerate(axes.ravel()):
    image = images[0]
    image = draw_haar_like_feature(image, 0, 0,
                                   images.shape[2],
                                   images.shape[1],
                                   [feature_coord_sel[idx_sorted[idx]]])
# feature_coord和feature_importances_，二者的特征是一一对应的，feature_coord第一个是第一个特征的坐标，feature_importances_是第一个特征的权值(重要性)
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()

cdf_feature_importances = np.cumsum(clf2.feature_importances_[idx_sorted])
cdf_feature_importances /= cdf_feature_importances[-1]  # 除以最大值
sig_feature_count = np.count_nonzero(cdf_feature_importances < 0.75)
sig_feature_percent = round(sig_feature_count /
                            len(cdf_feature_importances) * 100, 4)
print(f'{sig_feature_count} of {feature_types} features, or {sig_feature_percent}%, '
       f'account for 75% of branch points in the random forest.')

feature_coord_selected = feature_coord_sel[idx_sorted[:sig_feature_count]]
feature_type_selected = feature_type_sel[idx_sorted[:sig_feature_count]]

np.save('feature_coord_selected_'+train_num_str+'_all_5_features'+'.npy',feature_coord_selected)
np.save('feature_type_selected_'+train_num_str+'_all_5_features'+'.npy',feature_type_selected)



# %% 第二次重新提取特征，重新训练
t_start = time()
X = [
    extract_feature_image(img, feature_type_selected, feature_coord_selected)
    for img in images
]
X = np.stack(X)
time_subs_feature_comp = time() - t_start

y = np.array([1] * int(train_num/2) + [0] * int(train_num/2))
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5,
                                                    random_state=0,
                                                    stratify=y)

t_start = time()
clf3 = AdaBoostClassifier(estimator=None,algorithm="SAMME",n_estimators=200, learning_rate=0.8)
# clf3 = RandomForestClassifier(n_estimators=200, max_depth=None, max_features=None, n_jobs=-1, random_state=0)
clf3.fit(X_train, y_train)
time_subs_train = time() - t_start

auc_subs_features = roc_auc_score(y_test, clf3.predict_proba(X_test)[:, 1])

summary = (f'Computing the full feature set took '
            f'{time_subs_feature_comp:.3f}s, plus {time_subs_train:.3f}s '
            f'training, for an AUC of {auc_subs_features:.2f}.')

print(summary)

fpr, tpr, thresholds = roc_curve(y_test, clf3.decision_function(X_test))
# 使用RocCurveDisplay绘制ROC曲线
display = RocCurveDisplay(fpr=fpr,tpr=tpr)
display.plot()
# 显示图形
plt.show()
