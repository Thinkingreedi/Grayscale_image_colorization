import cv2  # 导入opencv
import numpy as np  # 导入numpy(可用来存储和处理大型矩阵)并起别名

from Guided_iltering import guideFilter  # 函数导入

if __name__ == '__main__':
    eps = 0.01  # 正则化参数
    winSize = (2, 2)  # 均值平滑窗口半径,类似卷积核（数字越大，柔和效果越好）
    image = cv2.imread("Shaded_image2.jpg", cv2.IMREAD_ANYCOLOR)
    image = cv2.resize(image, None, fx=0.8, fy=0.8,
                       interpolation=cv2.INTER_CUBIC)
    I = image/255.0  # 导向图像，将图像归一化
    p = I  # 利用导向滤波进行图像平滑处理时，通常令p = I
    s = 3  # 步长(缩放比例)
    guideFilter_img = guideFilter(I, p, winSize, eps, s)

    # 保存导向滤波结果
    guideFilter_img = guideFilter_img * 255  # (0,1)->(0,255)
    guideFilter_img[guideFilter_img > 255] = 255  # 防止像素溢出
    guideFilter_img = np.round(guideFilter_img)
    # 导向滤波返回的是灰度值范围在[0,1]之间的图像矩阵，像保存8位图要先乘255，再转换数据类型
    guideFilter_img = guideFilter_img.astype(np.uint8)
    cv2.imshow("Shaded_image", image)
    cv2.imshow("Natural_treatment", guideFilter_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("Natural_treatment2.jpg", guideFilter_img)
