import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay  
from time import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report#导入混淆矩阵对应的库
from sklearn.metrics import confusion_matrix
import joblib

train_num = 6850 # 必须偶数且×0.8是整数
train_num_str = str(train_num/2)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

lbp_feature = np.load('lbp_feaure_vector_after_pca_'+train_num_str+'_.npy' , allow_pickle=True)
hog_feature = np.load('hog_feaure_vector_after_pca_'+train_num_str+'_.npy' , allow_pickle=True)
haar_feature = np.load('haar_feaure_vector_'+train_num_str+'_.npy',allow_pickle=True)
feature1=hog_feature
feature2=lbp_feature
feature3=haar_feature



# haar特征和hog特征分别是维，维，都有6850个
cca1 = CCA(n_components=50)
cca1.fit(feature1, feature2)
joblib.dump(cca1,'cca1.pkl',compress=0) 

feature1_cca, feature2_cca = cca1.transform(feature1,feature2)
# 降维后，haar和hog特征都是10维，1800个，但是这1800个haar特征向量的第一维（也就是一个1800*1的列向量），
# 与hog特征向量的第一维（也是一个1800*1的列向量）有较高的相关性，但是与其他维度几乎不相关。其他维度亦是如此。
correlation_matrix = np.corrcoef(feature1_cca.T, feature2_cca.T) # 因为是计算列向量的相关性，要转置一下
# Plot the correlation matrix as a heatmap
plt.figure()
sns.heatmap(correlation_matrix[50:55,0:5], annot=True, annot_kws={"fontsize":12},cmap='Set2', 
            xticklabels=['haar'],
            yticklabels=['lbp'])
plt.title('Canonical Variables Correlation Matrix')
plt.show()

feature12 = np.hstack((feature1_cca,feature2_cca))

cca2 = CCA(n_components=50)
cca2.fit(feature12, feature3)
joblib.dump(cca2,'cca2.pkl',compress=0) 
feature12_caa, feature3_cca = cca2.transform(feature12, feature3)
correlation_matrix = np.corrcoef(feature12_caa.T, feature3_cca.T) # 因为是计算列向量的相关性，要转置一下
# Plot the correlation matrix as a heatmap
plt.figure()
sns.heatmap(correlation_matrix[50:55,0:5], annot=True, annot_kws={"fontsize":12},cmap='Set2', 
            xticklabels=['hog'],
            yticklabels=['haar_lbp'])
plt.title('Canonical Variables Correlation Matrix')
plt.show()
X = np.hstack((feature12_caa,feature3_cca))

y = np.array([1] * int(train_num/2) + [0] * int(train_num/2))
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5,
                                                    random_state=0,
                                                    stratify=y)

t_start = time()
clf = AdaBoostClassifier(estimator=None,algorithm="SAMME",n_estimators=100, learning_rate=1)
clf.fit(X_train, y_train)
time_subs_train = time() - t_start
joblib.dump(clf,'cca_feature_adaboost.pkl',compress=0) 

auc_subs_features = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
Score= clf.predict(X_test)
auc_subs_features = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
print(classification_report(y_test,Score))
print(confusion_matrix(y_test,Score))

summary = (f'{time_subs_train:.3f}s '
            f'training, for an AUC of {auc_subs_features:.5f}.')

print(summary)
 
fpr, tpr, thresholds = roc_curve(y_test, clf.decision_function(X_test))
# 使用RocCurveDisplay绘制ROC曲线
plt.plot(fpr, tpr,color='r')
plt.axis("square")
plt.xlabel("假阳性率")
plt.ylabel("真阳性率")
plt.title("融合特征-分类器的ROC曲线")
plt.show()



'''
特征融合是将来自不同特征提取方法的特征向量进行合并或组合，以生成更具代表性和信息丰富度的特征表示。以下是一些常见的特征融合方法：

1. **加权平均（Weighted Average）**：给每个特征向量分配一个权重，然后将它们加权求和得到融合后的特征向量。权重可以基于特征的重要性或根据数据的经验分配。

2. **特征串联（Feature Concatenation）**：将不同特征向量简单地连接在一起形成一个更长的特征向量。这种方法在维度较小的情况下通常比较常用。

3. **特征堆叠（Feature Stacking）**：将不同特征向量沿着新的维度堆叠在一起，形成一个更高维度的特征向量。

4. **投影（Projection）**：使用线性或非线性投影方法将不同特征向量映射到一个更低维度的子空间中，并将投影后的特征向量进行组合。

5. **多任务学习（Multi-Task Learning）**：将不同特征用于解决相关的多个任务，通过共享和交叉训练来融合特征表示。

6. **自动特征学习（Autoencoder-based Feature Learning）**：使用自动编码器等神经网络模型自动学习最佳的特征表示，并将学习到的特征进行融合。

7. **子空间融合（Subspace Fusion）**：将不同特征向量映射到不同的子空间中，然后将这些子空间进行融合或组合。

8. **核方法（Kernel Methods）**：使用核技巧将不同特征向量映射到更高维度的特征空间中，然后在该空间中进行线性或非线性融合。

9. **决策级融合（Decision-level Fusion）**：将基于不同特征向量训练的多个分类器的输出进行融合，例如投票、加权投票等方法。

这些方法可以根据具体任务的需求和特征向量的性质进行选择和组合。在实践中，常常需要通过实验评估不同的特征融合方法，以确定最适合任务的方法。
'''