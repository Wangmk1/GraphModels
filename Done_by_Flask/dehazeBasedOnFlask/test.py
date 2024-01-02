import base64
import io
import logging
import os
import argparse
import numpy as np
from PIL import Image
import onnxruntime as ort
import torch
from flask import Flask, request, jsonify, render_template
from torch.autograd import Variable


from utils import make_dataset, edge_compute

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


# router for HTML graph enhancement
@app.route('/enhance', methods=['GET', 'POST'])
def enhance_image():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--network', default='GCANet')
        parser.add_argument('--task', default='dehaze', help='dehaze | derain')
        parser.add_argument('--gpu_id', type=int, default=-1)
        opt = parser.parse_args()
        assert opt.task in ['dehaze', 'derain']
        opt.only_residual = opt.task == 'dehaze'
        opt.model = 'models/wacv_gcanet_%s.pth' % opt.task
        opt.use_cuda = opt.gpu_id >= 0

        if opt.network == 'GCANet':
            from GCANet import GCANet
            net = GCANet(in_c=4, out_c=3, only_residual=opt.only_residual)
            # print(net)
        else:
            print('network structure %s not supported' % opt.network)
            raise ValueError

        if opt.use_cuda:
            torch.cuda.set_device(opt.gpu_id)
            net.cuda()
        else:
            net.float()

        net.load_state_dict(torch.load(opt.model, map_location='cpu'))
        net.eval()

        image_file = request.files.get('files')
        # print(image_file)
        if image_file is None:
            return jsonify({'error': 'Missing image file in the request.'}), 400

        image_data = image_file.read()
        # print(image_data)
        img = Image.open(io.BytesIO(image_data)).convert('RGB')
        # print(img)
        im_w, im_h = img.size
        if im_w % 4 != 0 or im_h % 4 != 0:
            img = img.resize((int(im_w // 4 * 4), int(im_h // 4 * 4)))
        img = np.array(img).astype('float')
        # print(img)
        img_data = torch.from_numpy(img.transpose((2, 0, 1))).float()
        edge_data = edge_compute(img_data)
        in_data = torch.cat((img_data, edge_data), dim=0).unsqueeze(0) - 128
        in_data = in_data.cuda() if opt.use_cuda else in_data.float()
        with torch.no_grad():
            pred = net(Variable(in_data))
        if opt.only_residual:
            out_img_data = (pred.data[0].float() + img_data).round().clamp(0, 255)
        else:
            out_img_data = pred.data[0].float().round().clamp(0, 255)
        out_img = Image.fromarray(out_img_data.numpy().astype(np.uint8).transpose(1, 2, 0))
        # 将增强后的图像转换为Bytes数据
        enhanced_image_bytes = io.BytesIO()
        # print("数据bytes",enhanced_image_bytes)
        out_img.save(enhanced_image_bytes, format='JPEG')
        enhanced_image_bytes = enhanced_image_bytes.getvalue()
        enhanced_image_base64 = base64.b64encode(enhanced_image_bytes).decode('utf-8')
        # print(enhanced_image_base64)
        return jsonify({
            'enhanced_image': enhanced_image_base64,
        })
        print("增强成功")
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
    app.run(port=5001,debug=True)
