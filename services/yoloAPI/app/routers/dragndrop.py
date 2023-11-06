import json

from app.core.gateways.kafka import Kafka
from app.core.models.message import Message
from app.dependencies.kafka import get_kafka_instance

from fastapi import APIRouter, Depends, Request
from fastapi.templating import Jinja2Templates

router = APIRouter(
    prefix="/dragndrop",
    tags=["dragndrop"]
    )

templates = Jinja2Templates(directory = 'app/templates')
model_selection_options = ['yolov5s','yolov5m','yolov5l','yolov5x','yolov5n',
                           'yolov5n6','yolov5s6','yolov5m6','yolov5l6','yolov5x6']


@router.get("")
def dragndrop(request: Request):
    ''' drag_and_drop_detect detect page. Uses a drag and drop
    file interface to upload files to the server, then renders 
    the image + bboxes + labels on HTML canvas.
    '''

    return templates.TemplateResponse('dragndrop.html', 
            {"request": request,
            "model_selection_options": model_selection_options,
        })