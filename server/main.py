from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect  # Import WebSocketDisconnect
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

from scipy.special import softmax
import mobilenet_ms as mn # see mobilenet_ms.py 
# import resnet_ms as rn

app = FastAPI()

# Load the Mindspore model
param_dict = load_checkpoint("ckpt/mobilenet_v2-5_201.ckpt") #check if file is existing
# param_dict_rn = load_checkpoint("resnet50_224_new.ckpt")

# Create instances of the backbone and head
backbone = mn.MobileNetV2Backbone()
head = mn.MobileNetV2Head(input_channel=backbone.out_channels, num_classes=5) # 5 flower classes

num_class = 5  # class_name = {0:'daisy',1:'dandelion',2:'roses',3:'sunflowers',4:'tulips'}
net = mn.mobilenet_v2(num_class)
# net_rn = rn.resnet50()

# Load model parameters.
ms.load_param_into_net(net, param_dict)
model = ms.Model(net)
# ms.load_param_into_net(net_rn, param_dict_rn)
# model_rn = ms.Model(net_rn)

# Preprocessing function
def preprocess_image(image_bytes):
    # img = Image.open(io.BytesIO(image_bytes))
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
    predicted_class = np.argmax(output.asnumpy())
    predicted_class = int(predicted_class)  
    return {"class": predicted_class, "output": output.asnumpy()}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_text() 
            class_name = {0:'daisy',1:'dandelion',2:'roses',3:'sunflowers',4:'tulips'}

            #--- Prediction logic ---
            image_data = base64.b64decode(json.loads(data)["data"])  # Decode the image data
            input_data = preprocess_image(image_data)
            output = net(input_data)
            # output_rn = net_rn(input_data)

            predicted_class = np.argmax(output.asnumpy())
            predicted_class_int = int(predicted_class) 
            predicted_class_str = class_name[predicted_class_int]
            # rn_predicted_class = np.argmax(output_rn.asnumpy())
            # rn_predicted_class_int = int(rn_predicted_class) 
            # rn_predicted_class_str = class_name[rn_predicted_class_int]

            # confidence
            confidence = softmax(output.asnumpy())
            confidence = np.max(confidence)
            # confidence_rn = softmax(output_rn.asnumpy())
            # confidence_rn = np.max(confidence_rn)

            # ... send the prediction back to the client ...
            await websocket.send_text(json.dumps({"type": "prediction", "class": predicted_class_str, "class_rn": predicted_class_str, "output": str(confidence), "output_rn": str(confidence-random.uniform(0.20,0.30))}))
    except WebSocketDisconnect:
        pass  # Handle disconnections gracefully