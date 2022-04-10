# 计算每张图片的标签
from cv2 import cv2
import os


# 这一块是显示yolo的标签
def show_yolo_img(filename, point):
    point = str(point).split(" ")
    # 对应的文件路径
    img = cv2.imread(filename)
    # 把框画出来
    height = img.shape[0]
    width = img.shape[1]
    print(height, width)
    # 计算左上和右下角的坐标
    x1 = int((float(point[1]) - float(point[3]) / 2) * width)
    y1 = int((float(point[2]) - float(point[4]) / 2) * height)
    x2 = int((float(point[1]) + float(point[3]) / 2) * width)
    y2 = int((float(point[2]) + float(point[4]) / 2) * height)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow("a", img)
    cv2.waitKey(-1)


# 显示标注图片的链接
def show_label():
    path = "../data/detection/val/labels/"
    # 显示文件夹下所有图片
    files = os.listdir(path)
    # 读取文件
    for file in files:
        with open(path + file) as f:
            filename = path + "../images/" + file[:-4] + ".jpg"
            for point in f.readlines():
                show_yolo_img(filename, point)
    # print(files)


# 所有的检测图片路径
detection_path = "../data/detection"


# 这一块是开始计算标签
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
        print("遍历完成")


if __name__ == '__main__':
    # 开始计算各项指标
    calculate_label("train")
    calculate_label("test")
    calculate_label("val")
    # show_label()
