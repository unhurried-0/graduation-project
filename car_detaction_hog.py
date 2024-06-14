import cv2
import numpy as np
from skimage.feature import hog
import joblib
from skimage.transform import pyramid_gaussian
from imutils.object_detection import non_max_suppression
from skimage.transform import integral_image
from skimage.feature import haar_like_feature
import matplotlib.pyplot as plt

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

def sliding_window(image, stepSize, windowSize):
    # 滑动窗口遍历图像
    for y in range(0, image.shape[0] - windowSize[1] + 1, stepSize):
        for x in range(0, image.shape[1] - windowSize[0] + 1, stepSize):
            # 返回当前窗口的坐标和图像
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

# 加载特征提取算法
pca = joblib.load('hog_pca.pkl')

# 加载预训练的AdaBoost分类器
classifier = joblib.load('hog_pca_adaboost.pkl')

# 定义滑动窗口大小和步长
(winW, winH) = (64, 64)
stepSize = 20

# 定义图像金字塔的缩放因子
scale_factor = 1.5
levels = 10

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
        features = extract_feature_image(window)

        # 使用分类器进行预测
        features = pca.transform(features.reshape(1,-1))
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
