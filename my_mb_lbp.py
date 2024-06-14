import numpy as np
import cv2
from skimage.io import imread
import matplotlib.pyplot as plt

def get_multi_scale_block_lbp_feature(src, scale):
    '''
    src为原图，应为灰度图。
    scale为block像素尺寸。如block为9*9像素，scale=9。
    注:scale应该满足:“scale≥3 且 scale是3的倍数”，否则无意义，如scale=4，和scale=3效果相同
    '''
    cell_size = scale // 3
    y_cnt = (src.shape[0] + cell_size - 1) // cell_size     # 不足一个cell看作一个cell
    x_cnt = (src.shape[1] + cell_size - 1) // cell_size
    cell_image = np.zeros((y_cnt, x_cnt), dtype=np.uint8)

    for i in range(y_cnt):
        for j in range(x_cnt):
            temp = 0
            cnt = 0
            for m in range(i * cell_size, min((i + 1) * cell_size, src.shape[0])):
                for n in range(j * cell_size, min((j + 1) * cell_size, src.shape[1])):
                    temp += src[m, n]
                    cnt += 1
            # 计算均值
            temp //= cnt
            cell_image[i, j] = temp

    src = np.array(src)
    cell_image = np.array(cell_image)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.axis('off')
    ax1.imshow(src, cmap=plt.cm.gray)
    ax1.set_title('Input image'+'('+str(src.shape[0])+','+str(src.shape[1])+')')

    ax2.axis('off')
    ax2.imshow(cell_image, cmap=plt.cm.gray)
    ax2.set_title('Multi scale block image'+'('+str(cell_image.shape[0])+','+str(cell_image.shape[1])+')')
    plt.show()

    return get_origin_lbp_feature(cell_image)


def get_origin_lbp_feature(src):
    '''
    注：网页抄的代码此法与skimage里的default模式输出结果不一样，颜色相反，
    把>center改成>=center,对haar模板这种就几乎一样了，但是车辆图片黑白交界相反。
    但对车辆图还是相反。以为是二进制数排序问题，倒序一下，还是没用。后来看论文：
    "Performance Evaluation of Texture Measures with Classification Based on Kullback Discrimination of Distributions",
    论文不是按照顺时针顺序排的，而是：
    0  2  4
    8  m  16
    32 64 128
    改一下就和skimage一样了!
    '''
    dst = np.zeros((src.shape[0] - 2, src.shape[1] - 2), dtype=np.uint8)  # 其实边缘一行不包含边缘、角点信息，不算它的lbp值，也没关系
    for i in range(1, src.shape[0] - 1):
        for j in range(1, src.shape[1] - 1):
            center = src[i, j]
            lbp_code = 0   # 从左上角开始顺时针和中间比大小

            '''
            （网上例子）
            lbp_code |= (src[i - 1, j - 1] > center) << 7
            lbp_code |= (src[i - 1, j] > center) << 6
            lbp_code |= (src[i - 1, j + 1] > center) << 5
            lbp_code |= (src[i, j + 1] > center) << 4
            lbp_code |= (src[i + 1, j + 1] > center) << 3
            lbp_code |= (src[i + 1, j] > center) << 2
            lbp_code |= (src[i + 1, j - 1] > center) << 1
            lbp_code |= (src[i, j - 1] > center) << 0
            '''
            '''
            fuzz = 10 # 最开始尝试加入模糊值去噪,自己的小灵感，一个trick,还没实验效果如何
            lbp_code |= (src[i - 1, j - 1] > center + fuzz) << 7
            lbp_code |= (src[i - 1, j] > center + fuzz) << 6
            lbp_code |= (src[i - 1, j + 1] > center + fuzz) << 5
            lbp_code |= (src[i, j + 1] > center + fuzz) << 4
            lbp_code |= (src[i + 1, j + 1] > center + fuzz) << 3
            lbp_code |= (src[i + 1, j] > center + fuzz) << 2
            lbp_code |= (src[i + 1, j - 1] > center + fuzz) << 1
            lbp_code |= (src[i, j - 1] > center + fuzz) << 0
            '''
            
            # 自己对着论文改的
            lbp_code |= (src[i + 1, j + 1] >= center) << 7
            lbp_code |= (src[i + 1, j] >= center) << 6
            lbp_code |= (src[i + 1, j - 1] >= center) << 5
            lbp_code |= (src[i, j + 1] >= center) << 4
            lbp_code |= (src[i, j - 1] >= center) << 3
            lbp_code |= (src[i - 1, j+1] >= center) << 2
            lbp_code |= (src[i - 1, j] >= center) << 1
            lbp_code |= (src[i - 1, j - 1] >= center) << 0

            ''' 
            # 以为是二进制读取顺序问题，倒序了一下
            str = format(lbp_code,'b')  
            lbp_code = int(str[::-1], 2)
            '''

            dst[i - 1, j - 1] = lbp_code
    return dst



def SEMB_LBPFeature(src, scale):
    '''
    原LBP有2^8=256种特征，对应264维，而SEMB_LBP只有64种，对应64维，这样我们就不用降维了
    
    争议(我的思考):
    但是SEMB_LBP仍然是256种值，每张图产生的LBP向量的统计分布可能有很大不同，那么对多个特
    征向量，在同一个维度的含义也可能不同。例如，一张图是左边一半白，右边一半黑(注意白色灰度值大),
    那么LBP值最多的就是00000000(全黑or全白or左边缘)、10000011(右边缘)，该情况对应的SEMB_LBP向量
    就是:(0x0的个数,0x83的个数,0,0,0,……)。而对于左黑右白，LBP值最多的就是00000000(全黑全白右边缘)、
    00111000(左边缘)，SEMB_LBP向量是:(0x0的个数,0x38的个数,0,0,0,……)。同一维度，物理意义完全不同。
    所以本人觉得，此降维方法还不如之前的那个等阶模式，甚至不如提取256维LBP之后PCA有说服力。
    '''
    # 得到MB_LBP特征图
    MB_LBPImage = get_multi_scale_block_lbp_feature(src, scale)
    
    # 计算LBP特征值0-255的直方图
    histMat = cv2.calcHist([MB_LBPImage], [0], None, [256], [0, 255])
    histMat = histMat.reshape(1, -1)
    histVector = histMat.flatten()

    # 对histVector进行排序，降序排列
    sorted_indices = np.argsort(histVector)[::-1]

    # 构建编码表
    table = np.zeros(256, dtype=np.uint8)
    for i in range(63):
        table[sorted_indices[i]] = i

    # 根据编码表得到SEMB-LBP
    dst = np.copy(MB_LBPImage)
    for i in range(dst.shape[0]):
        for j in range(dst.shape[1]):
            dst[i, j] = table[dst[i, j]]

    return dst


def getLBPH(src, numPatterns, grid_x, grid_y, normed):
    src = src.getMat()
    width = src.cols // grid_x
    height = src.rows // grid_y
    result = np.zeros((grid_x * grid_y, numPatterns), dtype=np.float32)

    if src.empty():
        return result.reshape(1, 1)

    resultRowIndex = 0
    for i in range(grid_x):
        for j in range(grid_y):
            src_cell = src[i * height : (i + 1) * height, j * width : (j + 1) * width]
            hist_cell = getLocalRegionLBPH(src_cell, 0, numPatterns - 1, True)
            rowResult = result[resultRowIndex, :]
            hist_cell.reshape(1, 1).convertTo(rowResult, cv2.CV_32FC1)
            resultRowIndex += 1

    return result.reshape(1, 1)

def getLocalRegionLBPH(src, minValue, maxValue, normed):
    histSize = maxValue - minValue + 1
    range = [minValue, maxValue + 1]
    ranges = [range]

    result = cv2.calcHist([src], [0], None, [histSize], ranges, True, False)

    if normed:
        result /= src.total()

    return result.reshape(1, 1)


img = imread(r"C:\Users\11051\Desktop\grad_proj\code\data\OwnCollection\vehicles\Far\image0163.png")
#img = imread(r"C:\Users\11051\Desktop\grad_proj\code\data\haar-like.png")

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
mb_lbp_img = get_multi_scale_block_lbp_feature(img_gray,3)
#mb_lbp_img = SEMB_LBPFeature(img_gray,6)
img_gray = np.array(img_gray)
mb_lbp_img = np.array(mb_lbp_img)

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.axis('off')
ax1.imshow(img, cmap=plt.cm.gray)
ax1.set_title('Input image')

ax2.axis('off')
ax2.imshow(mb_lbp_img, cmap=plt.cm.gray)
ax2.set_title('Local Binary Pattern image')
plt.show()
print(img_gray.shape)
print(mb_lbp_img.shape)




'''
get_multi_scale_block_lbp_feature就是把一张图片先按cell归一化,然后算传统lbp特征。
这个函数接受两个参数:src 是输入的图像数据，scale 是block的尺度。
1.首先，根据给定的尺度，计算出单元格大小cell_size和将生成的归一化图的大小y_cnt*x_cnt。如果原图src边长不是cell_size的整数倍，
  把剩下的像素归一化。比如src是65*65，cell_size为3，那么63*63是可以正常归一化的，剩下的264个像素每4个归一化为一个cell。
  “y_cnt = (src.shape[0] + cell_size - 1) // cell_size”的作用就是:
  (1)src.shape[0]能被cell_size整除，那y_cnt=src.shape[0]/cell_size;
  (2)src.shape[0]不能被cell_size整除，那y_cnt=src.shape[0]//cell_size+1
2.创建一个和输入图像尺寸相应的空白图像 cell_image 用于存储计算得到的多尺度块 LBP 特征。
3.然后，使用嵌套的循环遍历源图像中的像素，计算每个像素周围像素的平均值，并将结果存储在 cell_image 中。
4.最后，调用 get_origin_lbp_feature 函数计算原始 LBP 特征。
'''

'''
分别实验MB_LBP和SEMB_LBP，他们得到的特征图同意有很多噪点，但是SEMB_LBP损失了更多信息，感觉不可信任，还不如用之前那个最原始的lbp，自己搓个LBPH。
'''

'''
放弃SEMB_LBP!使用旋转不变、等阶模式的圆形lbp("Face Detection Based on Multi-Block LBP Representation"文章)。
直接用skimage的，没空加模糊值去噪了!
而skimage.feature.multiblock_lbp()很含糊，返回一个8bit值，应该就是统计图片中一个区域内的SEMB_LBP直方图，得到64位的特征向量。
而且MB_LBP没有旋转不变性，降维的SEMB_LBP物理意义不明，不敢用。skimage.feature.multiblock_lbp()也没法可视化，没法验证特征合理性。
故放弃skimage.feature.multiblock_lbp()!
'''
