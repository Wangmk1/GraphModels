import base64
import io
import logging

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)


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
        image_data = Image.open(io.BytesIO(image_data)).convert('RGB')
        print("1:",image_data)
        image_data = np.array(image_data).astype('float32')
        # print("2:",image_data)
        imgYUV = cv2.cvtColor(image_data, cv2.COLOR_BGR2YCrCb)
        print("3",imgYUV)
        channelsYUV = cv2.split(imgYUV)
        t = channelsYUV[0]
        t = np.array(t,dtype='uint8')
        # 限制对比度的自适应阈值均衡化
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        print("4:",clahe)
        p = clahe.apply(t)
        p1 = channelsYUV[1]
        channelsYUV1 = np.array(p1 ,dtype='uint8')

        p2 = channelsYUV[2]
        channelsYUV2 = np.array(p2, dtype='uint8')

        print("p:\n", p)
        channels = cv2.merge([p, channelsYUV1, channelsYUV2])
        print("channels:\n",channels)
        result = cv2.cvtColor(channels, cv2.COLOR_YCrCb2BGR)
        print("result:",result)

        enhanced_data = result.astype(np.float32)

        # print(enhanced_image_array.dtype)
        # print(enhanced_image_array.shape)
        print("enhanced_data:", enhanced_data)

        enhanced_image = Image.fromarray(np.uint8(enhanced_data))
        # print("enhanced_image:", enhanced_image)

        # 将增强后的图像转换为Bytes数据
        enhanced_image_bytes = io.BytesIO()
        # print("数据bytes",enhanced_image_bytes)
        enhanced_image.save(enhanced_image_bytes, format='JPEG')
        enhanced_image_bytes = enhanced_image_bytes.getvalue()
        enhanced_image_base64 = base64.b64encode(enhanced_image_bytes).decode('utf-8')
        print("增强成功")

        return jsonify({
            'original_image': base64.b64encode(image_data.tobytes()).decode('utf-8'),
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
