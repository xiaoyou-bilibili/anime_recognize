import cv2
import numpy
from PIL import Image, ImageDraw, ImageFont
import base64


# 显示图片
def cv2ImgAddText(img, text, left, top, text_color=(0, 255, 0), text_size=20):
    if isinstance(img, numpy.ndarray):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    # 加载字体
    font_text = ImageFont.truetype(
        "font/font.ttf", text_size, encoding="utf-8")
    draw.text((left, top), text, text_color, font=font_text)
    return cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)


# opencv转base64
def cv_to_base64(image) -> str:
    base64_str = cv2.imencode('.jpg', image)[1].tostring()
    return base64.b64encode(base64_str).decode("utf-8")


def base64_to_cv(b64: str) -> cv2:
    img_b64decode = base64.b64decode(b64)  # base64解码
    img_array = numpy.frombuffer(img_b64decode, numpy.uint8)  # 转换np序列
    return cv2.imdecode(img_array, cv2.COLOR_BGR2RGB)  # 转换Opencv格式
