from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import mindspore as ms
from mindspore import Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import json
import base64
import random

from PIL import Image
import numpy as np
import io

from mindspore.nn import Softmax
import mobilenet_ms as mn # see mobilenet_ms.py 
import resnet_ms as rn

app = FastAPI()

# Load the Mindspore model
param_dict = load_checkpoint("ckpt/MobileNetV2.ckpt") # Check if file is existing
param_dict_rn = load_checkpoint("ckpt/Resnet50.ckpt")

# Create instances of the backbone and head with 5 output classes to match checkpoint
backbone = mn.MobileNetV2Backbone()
head = mn.MobileNetV2Head(input_channel=backbone.out_channels, num_classes=5)  # 5 classes (match checkpoint)

num_class = 5  # Set this to 5 to match the checkpoint
net = mn.mobilenet_v2(num_class)
net_rn = rn.resnet50(num_class)

# Load model parameters
ms.load_param_into_net(net, param_dict)
ms.load_param_into_net(net_rn, param_dict_rn)

# Ensure models are in evaluation mode
net.set_train(False)
net_rn.set_train(False)

model = ms.Model(net)
model_rn = ms.Model(net_rn)

# Preprocessing function
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = img.transpose(2, 0, 1)
    img = img[np.newaxis, ...]
    return Tensor(img, ms.float32)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    input_data = preprocess_image(image_bytes)
    output = net(input_data)
    output_rn = net_rn(input_data)
    predicted_class = np.argmax(output.asnumpy())
    predicted_class = int(predicted_class)
    predicted_class_rn = np.argmax(output_rn.asnumpy())
    predicted_class_rn = int(predicted_class_rn)
    return {"class": predicted_class, "output": output.asnumpy(), "class_rn": predicted_class_rn, "output_rn": output_rn.asnumpy()}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_text() 
            class_name = {0: 'Brown Spot',1: 'Healthy',2: 'Bacterial Leaf Blight',3: 'Leaf Blast'}

            # --- Prediction logic ---
            image_data = base64.b64decode(json.loads(data)["data"])  # Decode the image data
            input_data = preprocess_image(image_data)
            output = net(input_data)
            output_rn = net_rn(input_data)

            predicted_class = np.argmax(output.asnumpy())
            predicted_class_int = int(predicted_class)
            predicted_class_str = class_name[predicted_class_int]
            rn_predicted_class = np.argmax(output_rn.asnumpy())
            rn_predicted_class_int = int(rn_predicted_class) 
            rn_predicted_class_str = class_name[rn_predicted_class_int]

            # Confidence calculation
            softmax = Softmax()
            confidence = softmax(output).asnumpy()
            confidence = np.max(confidence) * 100
            confidence_rn = softmax(output_rn).asnumpy()
            confidence_rn = np.max(confidence_rn) *100

            # Send the prediction back to the client
            await websocket.send_text(json.dumps({"type": "prediction", "class": predicted_class_str, "class_rn": rn_predicted_class_str, "output": str(confidence), "output_rn": str(confidence_rn)}))
    except WebSocketDisconnect:
        pass  # Handle disconnections gracefully
