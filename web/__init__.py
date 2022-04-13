import cv2
from flask import Flask, request, Response, render_template
from flask_cors import CORS
import json
import numpy as np
from core.detection import detect_img
from core.recognize import get_feature
from core.es import feature_search
import numpy
from PIL import Image, ImageDraw, ImageFont

# 初始化flaskAPP
app = Flask(__name__)
# r'/*' 是通配符，让本服务器所有的URL 都允许跨域请求
# 允许跨域请求
CORS(app, resources=r'/*')


# 返回JSON字符串
def return_json(data):
    return Response(json.dumps(data, ensure_ascii=False), mimetype='application/json')


# 显示图片
def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if isinstance(img, numpy.ndarray):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    # 加载字体
    fontText = ImageFont.truetype(
        "font/font.ttf", textSize, encoding="utf-8")
    draw.text((left, top), text, textColor, font=fontText)
    return cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)


# 基于图片的目标检测
@app.route('/yolov5/detect_img', methods=['POST'])
def detect_image():
    # 获取请求的图片
    file = request.files['file']
    # 使用opencv读取图像
    img = cv2.imdecode(np.frombuffer(file.stream.read(), np.uint8), cv2.IMREAD_COLOR)
    # 这里顺便保存一下图片用于前端展示
    cv2.imwrite("./web/static/detect.jpg", img)
    # 运行检测模型
    points = detect_img(img)
    if len(points) > 0:
        # 遍历坐标信息
        for face in points:
            # 获取坐标信息
            x1, y1, x2, y2 = int(face[0]), int(face[1]), int(face[2]), int(face[3])
            # 截取脸部图片
            tmp = img[y1:y2, x1:x2]
            # 获取图片的特征信息
            feature = get_feature(tmp)
            # 对特征进行搜索
            response = feature_search(feature)
            # 显示中文信息
            img = cv2ImgAddText(img, response[0]["name"], x1, y1, (255, 0, 0), 20)
            # 显示框信息
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # print(points[0])
    cv2.imwrite("./web/static/detect_res.jpg", img)
    # 返回json类型字符串
    return return_json({
        "detect": "/static/detect_res.jpg",
        "row": "/static/detect.jpg"
    })


# 主页显示HTML
@app.route('/', methods=['GET'])
def index():
    return render_template('content.html')
