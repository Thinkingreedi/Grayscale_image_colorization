import cv2
import numpy as np
import collections
import os

Pixel = collections.namedtuple("Pixel", ["x", "y"])


class WeightPixel(object):
    """保存权重点的信息"""

    def __init__(self, weight, a, b):
        self.weight = weight
        self.a = a
        self.b = b

    def __lt__(self, other):
        return self.weight < other.weight


class ConsultImage(object):
    """参考图片类,参考图片为一幅RGB图像;"""

    def __init__(self, image_path=None, segment=10000, window_size=5, ratio=0.5):
        self.image_path = image_path
        self.segment = segment
        self.window_size = window_size
        self.ratio = ratio

        assert os.path.exists(self.image_path)

        self.image = cv2.imread(self.image_path)
        self.lab_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2Lab)
        self.l_image, self.a_image, self.b_image = cv2.split(self.lab_image)

        self.height, self.width, self.channel = self.lab_image.shape

        self.height_array = np.arange(self.height, dtype=np.int32)
        self.width_array = np.arange(self.width, dtype=np.int32)

        self.weight_list = self.get_weight_pixel()

    def get_weight_pixel(self):
        weight_list = []
        # 对参考图片创建一个模型，选取一定量的参考点，对每个参考点，记录其权值W
        for _ in range(self.segment):
            height_index = np.random.choice(self.height_array)
            width_index = np.random.choice(self.width_array)

            pixel = Pixel(height_index, width_index)
            pixel_light = self.l_image[pixel.x, pixel.y]
            pixel_a = self.a_image[pixel.x, pixel.y]
            pixel_b = self.b_image[pixel.x, pixel.y]

            pixel_std = self.get_domain_std(pixel)

            weight_list.append(WeightPixel(
                pixel_light * self.ratio + pixel_std * (1 - self.ratio), pixel_a, pixel_b))
        weight_list.sort()
        return weight_list

    def get_window_size(self):
        return self.window_size

    def max_min_pixel(self):
        return np.max(self.l_image), np.min(self.l_image)

    def get_ratio(self):
        return self.ratio

    def get_weight_list(self):
        return self.weight_list

    def get_domain_std(self, pixel):
        window_left = max(pixel.x - self.window_size, 0)
        window_right = min(pixel.x + self.window_size + 1, self.height)
        window_top = max(pixel.y - self.window_size, 0)
        window_bottom = min(pixel.y + self.window_size + 1, self.width)

        window_slice = self.l_image[window_left: window_right,
                                    window_top: window_bottom]

        return np.std(window_slice)
