import cv2
import numpy as np
from skimage.feature import hog
import joblib
from skimage.transform import pyramid_gaussian
from imutils.object_detection import non_max_suppression
from skimage.transform import integral_image
from skimage.feature import haar_like_feature
import matplotlib.pyplot as plt
import skimage.feature

# hog
def extract_feature_image_hog(img):
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

# lbp
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

# haar
def extract_feature_image_haar(img, feature_type, feature_coord=None):
    """Extract the haar feature for the current image"""
    ii = integral_image(img)
    return haar_like_feature(ii, 0, 0, ii.shape[0], ii.shape[1],
                             feature_type=feature_type,
                             feature_coord=feature_coord)



def sliding_window(image, stepSize, windowSize):
    # 滑动窗口遍历图像
    for y in range(0, image.shape[0] - windowSize[1] + 1, stepSize):
        for x in range(0, image.shape[1] - windowSize[0] + 1, stepSize):
            # 返回当前窗口的坐标和图像
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

# 加载特征提取算法
pca_hog = joblib.load('hog_pca.pkl')
pca_lbp = joblib.load('lbp_pca.pkl')
cca1 = joblib.load('cca1.pkl')
cca2 = joblib.load('cca2.pkl')
feature_coord_selected = np.load('feature_coord_selected_'+'900.0'+'_all_5_features'+'.npy', allow_pickle=True)
feature_type_selected = np.load('feature_type_selected_'+'900.0'+'_all_5_features'+'.npy', allow_pickle=True)


# 加载预训练的AdaBoost分类器
classifier = joblib.load('cca_feature_adaboost.pkl')

# 定义滑动窗口大小和步长
(winW, winH) = (64, 64)
stepSize = 20

# 定义图像金字塔的缩放因子
scale_factor = 1.5
levels = 2

# 定义非最大抑制的阈值
overlap_threshold = 0.01

# 加载道路图像
image = plt.imread(r"C:\Users\11051\Desktop\grad_proj\code\data\test4.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 初始化列表以存储检测到的边界框
boxes = []

# 循环遍历图像金字塔的不同尺度
for level, resized in enumerate(pyramid_gaussian(gray, downscale=scale_factor,max_layer=levels)): 
    # 循环遍历滑动窗口位置
    for (x, y, window) in sliding_window(resized, stepSize, windowSize=(winW, winH)):
        # 确保窗口大小与分类器期望的大小相匹配
        if window.shape[0] != winH or window.shape[1] != winW:
            continue

        # 从窗口中提取HOG特征
        feature1 = extract_feature_image_hog(window)
        feature1 = pca_hog.transform(feature1.reshape(1,-1))
        feature2 = getLBPH(src=window,P_num=8,Pattern='uniform',grid_x=8,grid_y=8,normed=True)
        feature2 = pca_lbp.transform(feature2.reshape(1,-1))
        feature1_cca,feature2_cca = cca1.transform(feature1,feature2)

        feature12 = np.hstack((feature1_cca,feature2_cca))
        feature3 = extract_feature_image_haar(window, feature_type_selected, feature_coord_selected)
        feature3 = feature3.reshape(1,-1)
        feature12_cca,feature3_cca = cca2.transform(feature12,feature3)

        features = np.hstack((feature12_cca,feature3_cca))

        # 使用分类器进行预测
        prediction = classifier.predict(features.reshape(1,-1))

        # 如果检测到车辆，则将边界框添加到列表中
        if prediction == 1:
            # 调整检测到的车辆框的坐标和大小，以适应原始图像尺寸
            startX = int(x * (scale_factor ** level))
            startY = int(y * (scale_factor ** level))
            endX = int((x + winW) * (scale_factor ** level))
            endY = int((y + winH) * (scale_factor ** level))
            boxes.append((startX, startY, endX, endY))



# 对重叠的边界框应用非最大抑制以去除重叠部分
boxes = np.array(boxes)
pick = non_max_suppression(boxes, probs=None, overlapThresh=overlap_threshold)


# 初始化 Matplotlib 图形
fig, ax = plt.subplots()
ax.axis('off')
# 绘制图像
ax.imshow(image)

# 循环绘制边界框
for (startX, startY, endX, endY) in pick:
    rect = plt.Rectangle((startX, startY), endX - startX, endY - startY,
                         fill=False, edgecolor=(0, 1, 0), linewidth=2)
    ax.add_patch(rect)

# 显示图形
plt.show()
