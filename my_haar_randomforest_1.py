# %% 库函数
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
from time import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
# sklearn 主要是机器学习库

from skimage.transform import integral_image
from skimage.feature import haar_like_feature
from skimage.feature import haar_like_feature_coord
from skimage.feature import draw_haar_like_feature

# %% 提取图像内的HAAR特征
counts = 0

def extract_feature_image(img, feature_type, feature_coord=None):
    """Extract the haar feature for the current image"""
    global counts
    ii = integral_image(img)
    counts = counts + 1
    print("%d "%counts)
    return haar_like_feature(ii, 0, 0, ii.shape[0], ii.shape[1],
                             feature_type=feature_type,
                             feature_coord=feature_coord)
# 输入一个积分图int_image,r、c分别是检测窗左上角那个点的横纵坐标， width、height是检测窗的，feature_type就是Haar的类型,
# 可见https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.haar_like_feature ，
# 有type-2-x,type-2-y之类的，不写就是默认所有特征全都有
# feature_coord是特征坐标
# 返回值是Haar_features型对象，有俩属性: (n_features，) narray of int或float。产生类似哈尔的特征。
# 每个值等于正负矩形之和的减法。数据类型取决于int_image的数据类型:当int_image的数据类型为int时为int;当int_image的数据类型为float时为int + float。


# %% 定义一些重要的参数和初始化变量
# 我们使用vehicle database数据集的一个子集,该子集由100张车辆图像和100张非人脸图像组成。每个图像都被调整为64 * 64像素的ROI。
# 我们从每组中选择75张图像来训练分类器,并确定最显著的特征。每个类别的其余25张图像用于评估分类器的性能

images = np.empty((0,64,64))
feature_types = ['type-2-x','type-2-y','type-3-y','type-3-x','type-4'] # 一次提取特征只会用到feature_types中的feature_types[feature_type_idx]，最后再整合
feature_type_idx =  4# 被提取的特征
train_signal = 0 # 0是用随机森林提取特征，跳过提取全部Haar特征。1是只提取全部的Haar特征，而不用接下来的随机森林
train_num = 200 # 必须偶数且×0.8是整数
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

# plt.imshow(images[0],cmap = 'gray')
# plt.axis('off')
# plt.show()


# %%  提取特征，分割数据集

if train_signal == 1:
    t_start = time()
    X = [extract_feature_image(img, feature_types[feature_type_idx]) for img in images]
    # np.save('haar_feature_before_stack_'+train_num_str+'_'+feature_types[0]+'.npy',X)
    X = np.stack(X)
    np.save('haar_feature_'+train_num_str+'_'+feature_types[feature_type_idx]+'.npy',X)
    time_full_feature_comp = time() - t_start
    print('Feature extracted')
    sys.exit() # 提取特征结束就结束，不做多余的事情
else :
    X = np.load('haar_feature_'+train_num_str+'_'+feature_types[feature_type_idx]+'.npy')


y = np.array([1] * int((train_num/2)) + [0] * int((train_num/2)))
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0, stratify=y)
# X_train, X_test, y_train, y_test 分别是训练集数据，测试集数据，训练集标签，测试集标签，random_state只要不是none，每次分割都是一样的数据集和测试集而不是每次都随机分割

# 下面的反斜杠除了与某些字符构成转义字符外，还可以与回车键连用表示跨行
feature_coord, feature_type = \
    haar_like_feature_coord(width=images.shape[2], height=images.shape[1], feature_type=feature_types[feature_type_idx])
# 输入检测窗的宽和高、Haar特征类型。输出feature_coord是Haar特征模板所在的坐标，feature_type是对应特征模板对应的模板类型 
# 但是这个函数里给出的小例子输出的坐标有两个，左上和右下角（？

# %% 建立随机森林分类器，提取重要的特征
# 随机森林分类器可以通过训练来选择最显著的特征,这个方法是确定哪些特征最常被整个决策树使用。通过在后续步骤中只使用最显著的特征，我们可以在保持准确性的同时大大加快计算速度。

clf = RandomForestClassifier(n_estimators=100, max_depth=None, max_features=100, n_jobs=-1, random_state=0)
# max_feature随机森林中，每个决策树建立于随机选取的m个特征，而不是全部特征，默认是根号N
# max_depth相当于预剪枝，防止过拟合
# 主要调参就是n_estimator,max_feature，不是炼丹，可以控制变量，从小到大调整这俩值，看效果，是会逐步拟合的
# n_jobs就是并行计算，因为随机森林的决策树不同于boost，他是独立建立的，所以应该并行计算提高速度
# 控制3个随机性来源：1. 构建树木时使用的示例的引导程序(如果bootstrap=True)2. 在每个节点上寻找最佳分割时要考虑的特征采样（如果max_features < n_features)，3. 绘制每个max_features的分割
# random_state的值可能是：1. None 无（默认） 使用numpy.random中的全局随机状态实例。多次调用该函数将重用同一实例，并产生不同的结果。
# 2. 一个整数 使用以给定整数作为种子的新随机数生成器。使用int将在不同的调用中产生相同的结果。但是，值得检查一下在多个不同的随机种子中的结果是否稳定。流行的整数随机种子为0和42。
t_start = time()
clf.fit(X_train, y_train)
time_full_train = time() - t_start
auc_full_features = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])

# Sort features in order of importance and plot the six most significant
idx_sorted = np.argsort(clf.feature_importances_)[::-1]
# idx_sorted应该是特征重要性从大到小排，从而得到的特征序号feature_importances_[idx_sorted[0]]就是最重要的那个特征的权重,[,,-1]是倒序选取的意思，argsort得到的是从小到大
feature_importances = clf.feature_importances_
feature_importances_sorted = feature_importances[idx_sorted]
fig, axes = plt.subplots(2,4)
for idx, ax in enumerate(axes.ravel()):
    image = images[0]
    image = draw_haar_like_feature(image, 0, 0,
                                   images.shape[2],
                                   images.shape[1],
                                   [feature_coord[idx_sorted[idx]]])
# feature_coord和feature_importances_，二者的特征是一一对应的，feature_coord第一个是第一个特征的坐标，feature_importances_是第一个特征的权值(重要性)
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])

_ = fig.suptitle('The most important features')
plt.show()
print(auc_full_features)

# %% 选取70%权重的特征，合并5类特征

cdf_feature_importances = np.cumsum(clf.feature_importances_[idx_sorted])
cdf_feature_importances /= cdf_feature_importances[-1]  # 除以最大值
sig_feature_count = np.count_nonzero(cdf_feature_importances < 0.7)
sig_feature_percent = round(sig_feature_count /
                            len(cdf_feature_importances) * 100, 4)
print(f'{sig_feature_count} of {feature_types[feature_type_idx]} features, or {sig_feature_percent}%, '
       f'account for 70% of branch points in the random forest.')

feature_coord_selected = feature_coord[idx_sorted[:sig_feature_count]]
feature_type_selected = feature_type[idx_sorted[:sig_feature_count]]

np.save('feature_coord_selected_'+train_num_str+'_'+feature_types[feature_type_idx]+'.npy',feature_coord_selected)
np.save('feature_type_selected_'+train_num_str+'_'+feature_types[feature_type_idx]+'.npy',feature_type_selected)