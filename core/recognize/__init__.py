from cv2 import cv2
import torch.nn as nn
import torch
from core.recognize.fmobilenet import FaceMobileNet
from PIL import Image
from core.recognize.config import config as conf

# 全局的标签信息
anime_map = {}

# 加载我们的模型
model = FaceMobileNet(512)
model = nn.DataParallel(model)
model.load_state_dict(torch.load(conf.model_path, map_location=conf.device))
model.eval()


def get_feature(img):
    im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    data = conf.test_transform(im)
    data = data[:, None, :, :]  # shape: (batch, 1, 128, 128)
    # print(data.shape)
    data = data.to(conf.device)
    net = model.to(conf.device)
    with torch.no_grad():
        features = net(data)
        features = features.cpu().numpy()
        if len(features):
            return features[0]
        else:
            return []

