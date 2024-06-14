import cv2
import math
import matplotlib.pyplot as plt 
from skimage.feature import hog 
from skimage import exposure  
from skimage.io import imread
import numpy as np

def gamma_trans(img, gamma):  # gamma函数处理
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]  # 建立映射表
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)  # 颜色值为整数
    return cv2.LUT(img, gamma_table)  


img=imread(r"C:\Users\11051\Desktop\grad_proj\code\data\test2.jpg")
#img=imread(r"C:\Users\11051\Desktop\grad_proj\code\data\OwnCollection\vehicles\Far\image0224.png")
#img_gray=rgb2gray(img)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
mean=np.mean(img_gray)
#gamma_val=math.log10(0.5) / math.log10(mean / 255) 
gamma_val=0.5
img_gray_gamma=gamma_trans(img_gray,gamma_val)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

fig_vector, hog_image = hog(
    img_gray_gamma,
    orientations=9,
    pixels_per_cell=(8 , 8),
    cells_per_block=(1 , 1),
    visualize=True,
    channel_axis=None,
   # transform_sqrt=True,
   # 彩色图这个一用效果就会变差，很奇怪。灰度化也会变差。
   # 已解决。是第18行rgb2gray的问题！这个函数返回的灰度图不是255阶数的，是浮点数
)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True)



ax1.axis('off')
ax1.imshow(img_gray, cmap=plt.cm.gray)
ax1.set_title('输入图像')

ax2.axis('off')
ax2.imshow(img_gray_gamma, cmap=plt.cm.gray)
ax2.set_title('Gamma校正后的图像')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

ax3.axis('off')
ax3.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax3.set_title('HOG特征图谱')
plt.show()
cv2.waitKey()