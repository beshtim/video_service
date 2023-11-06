
import json

from app.core.gateways.kafka import Kafka
from app.core.models.message import Message
from app.dependencies.kafka import get_kafka_instance

from fastapi import APIRouter, Depends, Request, Form, File, UploadFile
from fastapi.templating import Jinja2Templates

from typing import List, Optional
from app.utils.helpers import results_to_json, plot_one_box, base64EncodeImage
import random
import torch
import cv2
import numpy as np

router = APIRouter(
    prefix="",
    tags=["index"]
    )

model_selection_options = ['yolov5s','yolov5m','yolov5l','yolov5x','yolov5n',
                        'yolov5n6','yolov5s6','yolov5m6','yolov5l6','yolov5x6']
model_dict = {model_name: None for model_name in model_selection_options} #set up model cache
colors = [tuple([random.randint(0, 255) for _ in range(3)]) for _ in range(100)] #for bbox plotting


templates = Jinja2Templates(directory = 'app/templates')

@router.get("/")
def home(request: Request):
    ''' Returns html jinja2 template render for home page form
    '''

    return templates.TemplateResponse('home.html', {
            "request": request,
            "model_selection_options": model_selection_options,
        })

@router.post("/")
def detect_with_server_side_rendering(request: Request,
                        file_list: List[UploadFile] = File(...), 
                        model_name: str = Form(...),
                        img_size: int = Form(640)):

    if model_dict[model_name] is None:
        model_dict[model_name] = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)

    img_batch = [cv2.imdecode(np.fromstring(file.file.read(), np.uint8), cv2.IMREAD_COLOR)
                    for file in file_list]

    #create a copy that corrects for cv2.imdecode generating BGR images instead of RGB
    #using cvtColor instead of [...,::-1] to keep array contiguous in RAM
    img_batch_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in img_batch]

    results = model_dict[model_name](img_batch_rgb, size = img_size)

    json_results = results_to_json(results,model_dict[model_name])

    img_str_list = []
    #plot bboxes on the image
    for img, bbox_list in zip(img_batch, json_results):
        for bbox in bbox_list:
            label = f'{bbox["class_name"]} {bbox["confidence"]:.2f}'
            plot_one_box(bbox['bbox'], img, label=label, 
                    color=colors[int(bbox['class'])], line_thickness=3)

        img_str_list.append(base64EncodeImage(img))

    #escape the apostrophes in the json string representation
    encoded_json_results = str(json_results).replace("'",r"\'").replace('"',r'\"')

    return templates.TemplateResponse('show_results.html', {
            'request': request,
            'bbox_image_data_zipped': zip(img_str_list,json_results), #unzipped in jinja2 template
            'bbox_data_str': encoded_json_results,
        })