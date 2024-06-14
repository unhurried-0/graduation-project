import cv2
from skimage.transform import integral_image
from skimage.feature import haar_like_feature

feature_types = ['type-2-x', 'type-2-y']
img = cv2.imread("C:/Users/HP/Desktop/grad_proj/code/data/OwnCollection/vehicles/Far/image0224.png")
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ii = integral_image(img)
haar = haar_like_feature(ii, 0, 0, ii.shape[0], ii.shape[1],
                             feature_type=feature_types)
print(haar)

'''
skimage 的 haar 实现似乎没有标准化。标准化一般是除以标准差，变成“接近标准正态”，因为没有减均值。
'''
