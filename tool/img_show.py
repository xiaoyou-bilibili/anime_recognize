from cv2 import cv2

# 检测框路径
box_path = "../data/detection"

# 读取图片
def read_img(data, data_type):
    data = data.replace("\n","")
    # 切割数据集
    data = data.split(",")
    img_path = "%s/%s/images/%s" % (box_path, data_type, data[0])
    img = cv2.imread(img_path)
    print(img_path)
    x1 = int(data[1])
    y1 = int(data[2])
    x2 = int(data[3])
    y2 = int(data[4])
    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
    cv2.imshow("a", img)
    cv2.waitKey(-1)

    print(data)

# 获取所有的框
def get_all_label(data_type):
    with open("%s/%s.txt" % (box_path,data_type)) as f:
        for i in f.readlines():
            read_img(i, data_type)

# 显示框数据
if __name__ == '__main__':
    get_all_label("train")