import json
import re
from cv2 import cv2
import os
import torch.nn as nn
import torch
from core.recognize.fmobilenet import FaceMobileNet
import torchvision.transforms as T
from PIL import Image
from core.recognize.config import config as conf

# 全局的标签信息
anime_map = {}

# 加载我们的模型
model = FaceMobileNet(512)
model = nn.DataParallel(model)
model.load_state_dict(torch.load("model/23.pth", map_location=conf.device))
model.eval()


def get_feature(img):
    im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    data = conf.test_transform(im)
    # print(data.shape)
    # res = [im]
    # data = torch.cat(res, dim=0)  # shape: (batch, 128, 128)
    data = data[:, None, :, :]  # shape: (batch, 1, 128, 128)
    # print(data.shape)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = data.to(device)
    net = model.to(device)
    with torch.no_grad():
        features = net(data)
        features = features.cpu().numpy()
        if len(features):
            return features[0]
        else:
            return [0] * 512


# 获取所有图片的特征信息
def get_all_image_feature():
    # 首先我们读取所有的特征
    data_path = '../../data/recognize/images'
    for img_path in os.listdir(data_path):
        # 获取对应的名称
        name = anime_map[img_path]
        # 读取分类下所有的图片
        for filename in os.listdir("%s/%s" % (data_path, img_path)):
            img_data = cv2.imread("%s/%s/%s" % (data_path, img_path, filename))
            # 计算特征信息
            feature = get_feature(img_data).cpu().numpy()
            if len(feature) > 0:
                feature = feature[0]
                # 存储特征数据到es
                es_client.index(index="anime_face", body={
                    "id": img_path,
                    "name": name,
                    "embedding": feature
                })
                print(filename)


# 读取所有的标签信息
def get_all_info():
    with open("info.txt") as f:
        for item in f.readlines():
            data = json.loads(item)
            anime_map[re.findall(r"rectrain_([0-9]+)", data["id"])[0]] = data["name"]
