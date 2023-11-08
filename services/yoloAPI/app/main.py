import time

from app.core.gateways.kafka import Kafka
from app.core.gateways.minio import MinioServer
from app.core.yolov5.detect_video import YoloBase

from app.core.models.message import Message
from app.dependencies.kafka import get_kafka_instance
from app.enum import EnvironmentVariables
from app.routers import publisher, detect, dragndrop, index, minio,video


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

model_selection_options = ['yolov5s','yolov5m','yolov5l','yolov5x','yolov5n',
                        'yolov5n6','yolov5s6','yolov5m6','yolov5l6','yolov5x6']

model_dict = {model_name: None for model_name in model_selection_options} #set up model cache

colors = [tuple([random.randint(0, 255) for _ in range(3)]) for _ in range(100)] #for bbox plotting

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

yolo_model = YoloBase(weights='yolov5s.pt')

@app.on_event("startup")
async def startup_event():
    await kafka_server.aioproducer.start()


@app.on_event("shutdown")
async def shutdown_event():
    await kafka_server.aioproducer.stop()


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


app.include_router(index.router)
app.include_router(publisher.router)
app.include_router(dragndrop.router)
app.include_router(detect.router)
app.include_router(minio.router)
app.include_router(video.router)