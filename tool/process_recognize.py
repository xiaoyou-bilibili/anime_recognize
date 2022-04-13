from cv2 import cv2
import re
import os


# 下面这个计算标签信息
def get_recognize_train():
    # 根路径和训练图片的路径
    root_dir = "../data/recognize"
    train_path = "%s/images" % root_dir
    img_count = 0
    # 读取所有文件夹
    with open("%s/train.txt" % root_dir) as f:
        # with open("%s/label.txt" % train_path, "w") as f2:
        for img in f.readlines():
            img = img.replace("\n", "")
            data = img.split("\t")
            # 读取图片
            filename = "%s/train/%s" % (root_dir, data[0])
            # 对图片进行裁剪
            img = cv2.imread(filename)
            x1 = int(data[1])
            y1 = int(data[2])
            x2 = int(data[3])
            y2 = int(data[4])
            img = img[y1:y2, x1:x2]
            # 写入新的图片
            new_img_name = "%07d.jpg" % img_count
            # 获取图片的类别
            img_class = re.findall(r"rectrain_([0-9]+)/personai", filename)
            if len(img_class) > 0:
                # 如果文件路径不存在，那么就新建一个路径
                file_dir = "%s/%s" % (train_path, img_class[0])
                if not os.path.exists(file_dir):
                    os.mkdir(file_dir)
                print(file_dir)
                # 缩放图片
                img = cv2.resize(img, (250, 250))
                # 写入图片
                cv2.imwrite("%s/%s" % (file_dir, new_img_name), img)
                # 显示进度
                print("%s:%s" % (new_img_name, img_class[0]))
            # 图片序号+1
            img_count += 1


if __name__ == '__main__':
    get_recognize_train()
