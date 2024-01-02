import base64
import io
import logging
import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)


def calcGrayHist(I):
    # 计算灰度直方图
    h, w = I.shape[:2]
    grayHist = np.zeros([256], np.uint64)
    for i in range(h):
        for j in range(w):
            grayHist[I[i][j]] += 1
    return grayHist

def equalHist(img):
    import math
    # 灰度图像矩阵的高、宽
    h, w = img.shape
    # 第一步：计算灰度直方图
    grayHist = calcGrayHist(img)
    # 第二步：计算累加灰度直方图
    zeroCumuMoment = np.zeros([256], np.uint32)
    for p in range(256):
        if p == 0:
            zeroCumuMoment[p] = grayHist[0]
        else:
            zeroCumuMoment[p] = zeroCumuMoment[p - 1] + grayHist[p]
    # 第三步：根据累加灰度直方图得到输入灰度级和输出灰度级之间的映射关系
    outPut_q = np.zeros([256], np.uint8)
    cofficient = 256.0 / (h * w)
    for p in range(256):
        q = cofficient * float(zeroCumuMoment[p]) - 1
        if q >= 0:
            outPut_q[p] = math.floor(q)
        else:
            outPut_q[p] = 0
    # 第四步：得到直方图均衡化后的图像
    equalHistImage = np.zeros(img.shape, np.uint8)
    for i in range(h):
        for j in range(w):
            equalHistImage[i][j] = outPut_q[img[i][j]]
    return equalHistImage


def linear(source):
    img = cv2.imread(source, 0)
    # 使用自己写的函数实现
    equa = equalHist(img)
    return equa


# homepage
@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


# router for HTML graph enhancement
@app.route('/enhance', methods=['GET', 'POST'])
def enhance_image():
    # POST for requiring the graph
    try:
        image_file = request.files.get('files')
        if image_file is None:
            return jsonify({'error': 'Missing image file in the request.'}), 400

        image_data = image_file.read()
        image_data = Image.open(io.BytesIO(image_data)).convert("RGB")
        image_data = linear(image_data)

        # print("输出Image_data的数据为：", image_data)

        enhanced_image_array = image_data.astype(np.float32)

        print("增强后的矩阵为：\n", enhanced_image_array)
        # print(enhanced_image_array.dtype)
        # print(enhanced_image_array.shape)

        print("enhanced_image_array.dtype:", enhanced_image_array)

        enhanced_image = Image.fromarray(np.uint8(enhanced_image_array))
        # print("enhanced_image:", enhanced_image)

        # 将增强后的图像转换为Bytes数据
        enhanced_image_bytes = io.BytesIO()
        # print("数据bytes",enhanced_image_bytes)
        enhanced_image.save(enhanced_image_bytes, format='JPEG')
        enhanced_image_bytes = enhanced_image_bytes.getvalue()
        enhanced_image_base64 = base64.b64encode(enhanced_image_bytes).decode('utf-8')
        print("增强成功")

        return jsonify({
            'enhanced_image': enhanced_image_base64,
        })

    except Exception as e:
        logging.error("An error occurred during image enhancement: %s", str(e))
        return jsonify({'error': 'An error occurred during image enhancement.'}), 500


@app.route("/enhance_image", methods=["GET"])
def enhance_image1():
    # Perform image enhancement using your model
    # Replace this with your actual image enhancement code
    enhanced_image = Image.open("path_to_enhanced_image.jpg")

    # Convert enhanced image to base64
    enhanced_image_bytes = io.BytesIO()
    enhanced_image.save(enhanced_image_bytes, format="JPEG")
    enhanced_image_base64 = base64.b64encode(enhanced_image_bytes.getvalue()).decode("utf-8")

    return jsonify({"enhanced_image": enhanced_image_base64})


if __name__ == '__main__':
    app.run(debug=True)
