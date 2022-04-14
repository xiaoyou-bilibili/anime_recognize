import torch

# 加载本地的模型
model = torch.hub.load('yolov5', 'custom', source='local', path="model/best.pt")


# 图片检测，这里我们直接传入cv格式的图片就可以了
def detect_img(img):
    # 调用模型进行检测获取结果
    results = model(img)
    # 获取我们的坐标信息并返回
    points = results.xyxy[0].cpu().numpy()
    return points


# 检测图片并保存
def detect_img_with_save(img, filename):
    # 调用模型进行检测获取结果
    results = model(img)
    results.save(filename)
    # 获取我们的坐标信息并返回
    points = results.xyxy[0].cpu().numpy()
    return points
