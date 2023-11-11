import time

from app.core.gateways.kafka import Kafka
from app.core.gateways.minio import MinioServer
from app.core.yolov5.detect_video import YoloBase

from app.core.models.message import Message
from app.dependencies.kafka import get_kafka_instance
from app.enum import EnvironmentVariables
from app.routers import publisher, detect, images, minio, video, pipeline


from dotenv import load_dotenv

from fastapi import Depends, FastAPI, Request, Form, File, UploadFile
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional

import cv2
import numpy as np

import torch
import base64
import random

load_dotenv()

app = FastAPI(title='Kafka API')

templates = Jinja2Templates(directory = 'app/templates')

kafka_server = Kafka(
    topic=EnvironmentVariables.KAFKA_TOPIC_NAME.get_env(),
    port=EnvironmentVariables.KAFKA_PORT.get_env(),
    servers=EnvironmentVariables.KAFKA_SERVER.get_env(),
)

minio_server = MinioServer(
            server=EnvironmentVariables.MINIO_HOST.get_env(),
            port=EnvironmentVariables.MINIO_PORT.get_env(),
            user=EnvironmentVariables.MINIO_USER.get_env(),
            psw=EnvironmentVariables.MINIO_PASSWORD.get_env(),
)

yolo_model = YoloBase(
    weights='/weights/medium_1024.pt',
    imgsz=(1024,1024),
    conf_thres=0.6,
    iou_thres=0.5,
    )


# <============ ASYNC KAFKA ============>
# @app.on_event("startup")
# def startup_event():
#     kafka_server.producer.start()


# @app.on_event("shutdown")
# def shutdown_event():
#     kafka_server.producer.stop()


# @app.middleware("http")
# def add_process_time_header(request: Request, call_next):
#     start_time = time.time()
#     response = call_next(request)
#     process_time = time.time() - start_time
#     response.headers["X-Process-Time"] = str(process_time)
#     return response
# <============ =========== ============>

app.include_router(images.router)
app.include_router(publisher.router)
app.include_router(detect.router)
app.include_router(minio.router)
app.include_router(video.router)
app.include_router(pipeline.router)