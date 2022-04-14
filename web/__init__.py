import cv2
from flask import Flask, request, Response, render_template
from flask_cors import CORS
import json
import numpy as np
from core.detection import detect_img, detect_img_with_save
from core.recognize import get_feature
from core.es import feature_search, store_feature
from core.tool import cv2ImgAddText, cv_to_base64, base64_to_cv

# 初始化flaskAPP
app = Flask(__name__)
# r'/*' 是通配符，让本服务器所有的URL 都允许跨域请求
# 允许跨域请求
CORS(app, resources=r'/*')


# 返回JSON字符串
def return_json(data):
    return Response(json.dumps(data, ensure_ascii=False), mimetype='application/json')


# 基于图片的目标检测
@app.route('/anime_face/recognize_img', methods=['POST'])
def recognize_image():
    # 获取请求的图片
    file = request.files['file']
    # 使用opencv读取图像
    img = cv2.imdecode(np.frombuffer(file.stream.read(), np.uint8), cv2.IMREAD_COLOR)
    # 运行检测模型
    points = detect_img(img)
    # 人脸信息
    face_info = []
    if len(points) > 0:
        # 遍历坐标信息
        for face in points:
            # 获取坐标信息
            x1, y1, x2, y2 = int(face[0]), int(face[1]), int(face[2]), int(face[3])
            # 截取脸部图片
            tmp = img[y1:y2, x1:x2]
            # 对脸部图片进行缩放
            tmp_scale = cv2.resize(tmp, (250, 250))
            # 获取图片的特征信息
            feature = get_feature(tmp_scale)
            # 对特征进行搜索
            response = feature_search(feature)
            # 存储结果
            face_info.append({
                "img": cv_to_base64(tmp),
                "point": json.dumps([x1, y1, x2, y2]),
                "name": response[0]["name"],
                "score": response[0]["score"]
            })
        for face in face_info:
            point = json.loads(face["point"])
            # 获取坐标信息
            x1, y1, x2, y2 = int(point[0]), int(point[1]), int(point[2]), int(point[3])
            # 显示中文信息
            img = cv2ImgAddText(img, face["name"], x1, y1, (245, 108, 108), 40)
            # 显示框信息
            cv2.rectangle(img, (x1, y1), (x2, y2), (64, 158, 255), 4)
    # 保存我们绘制的结果
    cv2.imwrite("./web/static/recognize_res.jpg", img)
    # 返回json类型字符串
    return return_json({
        "detect": "/static/recognize_res.jpg",
        "face_info": face_info
    })


# 基于图片的目标检测
@app.route('/anime_face/detect_img', methods=['POST'])
def detect_image():
    # 获取请求的图片
    file = request.files['file']
    # 使用opencv读取图像
    img = cv2.imdecode(np.frombuffer(file.stream.read(), np.uint8), cv2.IMREAD_COLOR)
    # 运行检测模型
    points = detect_img(img)
    # 人脸信息
    face_info = []
    if len(points) > 0:
        # 遍历坐标信息
        for face in points:
            # 获取坐标信息
            x1, y1, x2, y2 = int(face[0]), int(face[1]), int(face[2]), int(face[3])
            # 截取脸部图片
            tmp = img[y1:y2, x1:x2]
            # 存储结果
            face_info.append({
                "img": cv_to_base64(tmp),
                "point": json.dumps([x1, y1, x2, y2]),
            })
        for face in face_info:
            point = json.loads(face["point"])
            # 获取坐标信息
            x1, y1, x2, y2 = int(point[0]), int(point[1]), int(point[2]), int(point[3])
            # 显示框信息
            cv2.rectangle(img, (x1, y1), (x2, y2), (64, 158, 255), 4)
    # 保存我们绘制的结果
    cv2.imwrite("./web/static/detect_res.jpg", img)
    # 返回json类型字符串
    return return_json({
        "detect": "/static/detect_res.jpg",
        "face_info": face_info
    })


# 添加人脸信息
@app.route('/anime_face/add_face', methods=['POST'])
def add_face():
    # 获取所有的参数
    data = request.form
    name = data["name"]
    img = base64_to_cv(data["img"])
    img = cv2.resize(img, (250, 250))
    # 获取图片的特征信息
    feature = get_feature(img)
    # 保存图片的特征
    store_feature("0", name, feature, "")
    # 返回生成的图片和种子
    return return_json({})


# 主页显示HTML
@app.route('/', methods=['GET'])
def index():
    return render_template('content.html')
