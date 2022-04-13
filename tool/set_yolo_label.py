# 计算每张图片的标签
from cv2 import cv2

# 所有的检测图片路径
detection_path = "../data/detection"


# 开始计算标签
def calculate_label(data_type):
    # 首先我们读取标签
    with open("%s/%s.txt" % (detection_path, data_type)) as f:
        for item in f.readlines():
            data = str(item).split(",")
            if len(data) >= 5:
                name = data[0]
                # 计算对应的标签名字
                label_name = "%s/%s/labels/%s.txt" % (detection_path, data_type, name[:name.find(".")])
                # 先读取图片获取图片的坐标
                filename = "%s/%s/images/%s" % (detection_path, data_type, name)
                print(filename)
                shape = cv2.imread(filename).shape
                height = shape[0]
                width = shape[1]
                # 计算中心点
                point = [
                    "0",
                    str((float(data[3]) + float(data[1])) / 2 / float(width)),
                    str((float(data[4]) + float(data[2])) / 2 / float(height)),
                    str((float(data[3]) - float(data[1])) / float(width)),
                    str((float(data[4]) - float(data[2])) / float(height))
                ]
                # 写入文件
                with open(label_name, "a") as f2:
                    f2.write(" ".join(point) + "\n")
        print("%s遍历完成" % data_type)


if __name__ == '__main__':
    # 开始计算各项指标
    calculate_label("train")
    calculate_label("test")
    calculate_label("val")
