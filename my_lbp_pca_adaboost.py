import skimage.feature
import matplotlib.pyplot as plt
from skimage.io import imread
import cv2
import numpy as np
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from time import time
import joblib

from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay  
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report#导入混淆矩阵对应的库
from sklearn.metrics import confusion_matrix

def getLBPH(src,P_num, Pattern, grid_x, grid_y, normed):
    '''
    src为输入的图片

    P_num为采样点个数P

    Pattern为LBP值的种类，决定了LBPH的维度。举例说明LBPH的维度:采样点为8个，特征图分成8行8列64块区域
    如果用的是原始的LBP或Extended LBP特征，其LBP特征值的模式为numPatterns=2^8=256种，则一幅图像的LBP特征向量维度为:64*256=16384维
    如果使用的Uniform Pattern LBP特征，其LBP值的模式为numPatterns=8+2=10种，其特征向量维度为:64*10=640维

    grid_x,grid_y为分割出的网格横、纵向数量，_src.shape[1] // grid_x = width 就是src_cell的宽

    normed=True代表归一化处理

    返回一个一维数组，大小为grid_x*grid_y*numPatterns
    '''
    _src = skimage.feature.local_binary_pattern(
        image=src,
        P=P_num,
        R=1.0,
        #method='var'
        method=Pattern
    )

    width = _src.shape[1] // grid_x     # 计算src_cell宽度，shape[1]是_src数组列数，也就是宽度
    height = _src.shape[0] // grid_y   
    if Pattern=='uniform':
        numPatterns = P_num + 2
    if Pattern=='default':
        numPatterns = 2**P_num
    result = np.zeros((grid_x * grid_y, numPatterns), dtype=np.float32)     # grid_x*grid_y就是总的src_cell数量，每个src_cell产生一个有numPattern维的特征向量

    resultRowIndex = 0
    for i in range(grid_x):
        for j in range(grid_y):
            src_cell = _src[i * height : (i + 1) * height, j * width : (j + 1) * width]     # python的切片不会引发索引越界的错误,如果一个src_cell尺寸不足width*height,仍会统计出它的局部lbph
            hist_cell, _ = np.histogram(src_cell.ravel(), numPatterns, range=(0,numPatterns-1), density=normed)     # 如果numPattern=256,那lbp值最大为11111111=255,所以range是(0,255)
            hist_cell = hist_cell.astype(np.float32)
            result[resultRowIndex, :] = hist_cell
            resultRowIndex += 1

    return result.ravel()




# %% 定义关键参数
images = np.empty((0,64,64),np.int8)
train_num = 6850  # 必须偶数且×0.8是整数
train_num_str = str(train_num/2)
pca_dimension=122
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# %% 读取数据集
images = np.load('img_data_all_'+train_num_str+'_.npy', allow_pickle=True)

# %%
t_start = time()
X_long = [
    getLBPH(src=img,P_num=8,Pattern='uniform',grid_x=8,grid_y=8,normed=True)
    for img in images
]
# 上面的x只是一个train_num个特征向量构成的列表，不是numpy数组。之前有程序下面一行是numpy.stack,没啥用
X_long = np.array(X_long)
pca = PCA(n_components=pca_dimension)
pca.fit(X_long)
X = pca.transform(X_long)
time_subs_feature_comp = time() - t_start
np.save('lbp_feaure_vector_after_pca_'+train_num_str+'_.npy',X)
joblib.dump(pca,'lbp_pca.pkl',compress=0) 

y = np.array([1] * int(train_num/2) + [0] * int(train_num/2))
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5,
                                                    random_state=0,
                                                    stratify=y)

# %% 测试弱学习器个数
'''
estimator_num = range(2,150,2)
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

# %% 测roc 迭代70次

t_start = time()
clf3 = AdaBoostClassifier(estimator=None,algorithm="SAMME",n_estimators=70, learning_rate=1)
clf3.fit(X_train, y_train)
time_subs_train = time() - t_start
joblib.dump(clf3,'lbp_pca_adaboost.pkl',compress=0) 

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
plt.title("LBP特征-分类器的ROC曲线")
plt.show()
