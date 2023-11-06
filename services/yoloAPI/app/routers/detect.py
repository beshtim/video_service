from app.core.gateways.kafka import Kafka
from app.core.models.message import Message
from app.dependencies.kafka import get_kafka_instance

from fastapi import APIRouter, Request, Form, File, UploadFile
from typing import List, Optional
from app.utils.helpers import results_to_json, plot_one_box, base64EncodeImage
import torch
import cv2
import random
import numpy as np

router = APIRouter(
    prefix="/detect",
    tags=["detect"]
    )

colors = [tuple([random.randint(0, 255) for _ in range(3)]) for _ in range(100)] #for bbox plotting
model_selection_options = ['yolov5s','yolov5m','yolov5l','yolov5x','yolov5n',
                        'yolov5n6','yolov5s6','yolov5m6','yolov5l6','yolov5x6']
model_dict = {model_name: None for model_name in model_selection_options} #set up model cache

@router.post("")
def detect_via_api(request: Request,
                file_list: List[UploadFile] = File(...), 
                model_name: str = Form(...),
                img_size: Optional[int] = Form(640),
                download_image: Optional[bool] = Form(False)):
    
    '''
    Requires an image file upload, model name (ex. yolov5s). 
    Optional image size parameter (Default 640)
    Optional download_image parameter that includes base64 encoded image(s) with bbox's drawn in the json response
    
    Returns: JSON results of running YOLOv5 on the uploaded image. Bbox format is X1Y1X2Y2. 
            If download_image parameter is True, images with
            bboxes drawn are base64 encoded and returned inside the json response.

    Intended for API usage.
    '''

    if model_dict[model_name] is None:
        model_dict[model_name] = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
    
    img_batch = [cv2.imdecode(np.fromstring(file.file.read(), np.uint8), cv2.IMREAD_COLOR)
                for file in file_list]

    #create a copy that corrects for cv2.imdecode generating BGR images instead of RGB, 
    #using cvtColor instead of [...,::-1] to keep array contiguous in RAM
    img_batch_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in img_batch]
    
    results = model_dict[model_name](img_batch_rgb, size = img_size) 
    json_results = results_to_json(results, model_dict[model_name])

    if download_image:
        #server side render the image with bounding boxes
        for idx, (img, bbox_list) in enumerate(zip(img_batch, json_results)):
            for bbox in bbox_list:
                label = f'{bbox["class_name"]} {bbox["confidence"]:.2f}'
                plot_one_box(bbox['bbox'], img, label=label, 
                        color=colors[int(bbox['class'])], line_thickness=3)

            payload = {'image_base64':base64EncodeImage(img)}
            json_results[idx].append(payload)

    encoded_json_results = str(json_results).replace("'",r'"')
    return encoded_json_results

