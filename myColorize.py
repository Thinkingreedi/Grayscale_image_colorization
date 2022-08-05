import cv2      # 导入opencv
import bisect   # 导入bisect,用于有序序列的插入和查找
import numpy as np  # 导入numpy(可用来存储和处理大型矩阵)并起别名

from Colorize import WeightPixel  # 函数导入
from Colorize import Pixel
from Colorize import ConsultImage


if __name__ == '__main__':

    # 创建参考图像的分析类；
    print("---Initializes the reference image---\n")
    consult_image = ConsultImage("Reference_image2.jpg")
    print("---Initialization complete---\n")

    # 读取灰度图像；opencv默认读取的是3通道的，不需要扩展通道；
    gray_image = cv2.imread("Gray_image2.jpg")
    cv2.imshow("Grayscale_image", gray_image)  # 展示灰度图像
    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2LAB)  # 实现了RGB和Lab之间的转换
    height, width, channel = gray_image.shape  # 读取矩阵

    # 获取灰度图像的亮度信息；
    gray_image_l_origin, gray_image_a, gray_image_b = cv2.split(gray_image)
    gray_image_l = np.zeros(gray_image_l_origin.shape, np.uint8)

    max_pixel, min_pixel = np.max(
        gray_image_l_origin), np.min(gray_image_l_origin)
    consult_max_pixel, consult_min_pixel = consult_image.max_min_pixel()
    pixel_ratio = (consult_max_pixel - consult_min_pixel) / \
        (max_pixel - min_pixel)

    # 把灰度图像的亮度值映射到参考图像范围内；
    print("---Mapping---\n")
    for i in range(height):
        for j in range(width):
            pixel_light = consult_min_pixel + \
                (gray_image_l_origin[i, j] - min_pixel) * pixel_ratio

            gray_image_l[i, j] = pixel_light

    # 获取领域的窗口大小；
    window_size = consult_image.get_window_size()
    ratio = consult_image.get_ratio()

    print("---Start colorization---\n")
    for row in range(height):
        for col in range(width):
            pixel = Pixel(row, col)
            pixel_light = gray_image_l[pixel.x, pixel.y]

            # 求窗口内像素方差；
            window_left = max(pixel.x - window_size, 0)
            window_right = min(pixel.x + window_size + 1, height)
            window_top = max(pixel.y - window_size, 0)
            window_bottom = min(pixel.y + window_size + 1, width)

            window_slice = gray_image_l[window_left: window_right,
                                        window_top: window_bottom]

            pixel_std = np.std(window_slice)

            weight_pixel = WeightPixel(
                ratio * pixel_light + (1 - ratio) * pixel_std, 0, 0)

            search_pixel = bisect.bisect(
                consult_image.get_weight_list(), weight_pixel)
            search_pixel = 1 if search_pixel == 0 else search_pixel
            search_pixel = len(consult_image.get_weight_list()) - 1 \
                if search_pixel == len(consult_image.get_weight_list()) else search_pixel

            left_pixel = consult_image.get_weight_list()[search_pixel - 1]
            right_pixel = consult_image.get_weight_list()[search_pixel]

            nearest_pixel = left_pixel if left_pixel.weight + \
                right_pixel.weight > 2 * weight_pixel.weight else right_pixel

            gray_image_a[row, col] = nearest_pixel.a
            gray_image_b[row, col] = nearest_pixel.b

    merge_image = cv2.merge([gray_image_l, gray_image_a, gray_image_b])

    rgb_image = cv2.cvtColor(merge_image, cv2.COLOR_LAB2BGR)
    cv2.imshow("Shaded_image", rgb_image)  # 展示着色后的图像
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("Shaded_image2.jpg", rgb_image)  # 写入
