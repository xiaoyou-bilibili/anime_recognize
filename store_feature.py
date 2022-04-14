import json
import re
from cv2 import cv2
import os
import tqdm
import random
from core.recognize import get_feature
from core.es import store_feature

# 全局的标签信息
anime_map = {}


# 读取所有的标签信息
def get_all_info():
    with open("anime_info.txt") as f:
        for item in f.readlines():
            data = json.loads(item)
            anime_map[re.findall(r"rectrain_([0-9]+)", data["id"])[0]] = data["name"]


# 获取所有图片的特征信息
def get_all_image_feature():
    # 首先我们读取所有的特征
    data_path = 'data/recognize/images'
    # 遍历所有图片路径，这里使用进度条显示当前进度
    for img_path in tqdm.tqdm(os.listdir(data_path)):
        # 获取对应的名称
        name = anime_map[img_path]
        # 读取分类下所有的图片
        images = os.listdir("%s/%s" % (data_path, img_path))
        # 随机从数组里面取10张图片
        images = random.sample(images, 10)
        for filename in images:
            img_data = cv2.imread("%s/%s/%s" % (data_path, img_path, filename))
            # 计算特征信息
            feature = get_feature(img_data)
            if len(feature) > 0:
                # 存储特征数据到es
                store_feature(img_path, name, feature, filename)


if __name__ == '__main__':
    # 读取所有的标签信息
    get_all_info()
    # 获取特征信息
    get_all_image_feature()
