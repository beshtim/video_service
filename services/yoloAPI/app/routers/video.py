from app.core.gateways.kafka import Kafka
from app.dependencies.kafka import get_kafka_instance

from app.core.gateways.minio import MinioServer
from app.dependencies.minio import get_minio_instance

from app.core.yolov5.detect_video import YoloBase
from app.dependencies.yolo import get_yolo_instance

from app.utils.helpers import draw_poly, plot_one_box, minio_post, kafka_post

from fastapi import APIRouter, Depends, UploadFile, File, Request, Form
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
from tempfile import NamedTemporaryFile

import random
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import os


router = APIRouter(
    prefix="",
    tags=["video"],
    dependencies=[Depends(get_kafka_instance), Depends(get_minio_instance), Depends(get_yolo_instance)]
    )

colors = [tuple([random.randint(0, 255) for _ in range(3)]) for _ in range(100)]

class Predictor:
    def __init__(self):
        self.stream = None

    def create_stream(self, model, video, minio, kafka):
        self.stream = self.get_stream(model, video, minio, kafka)
        print("stream created")

    def get_stream(self, yolo, video, minio, kafka): 
        cap = cv2.VideoCapture(video)
        if cap is None or not cap.isOpened():
            return 'no_camera'

        def iter_func():
            while cap.isOpened():
                ret, frame = cap.read()  # read the camera frame
                # frame_num = cap.get(cv2.CAP_PROP_POS_FRAMES)
                if not ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break

                img_batch = np.expand_dims(frame, axis=0)

                out = yolo.yolo(img_batch)
                print(out)

                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                    
        return iter_func()  # returns generator

predictor = Predictor()

templates = Jinja2Templates(directory = 'app/templates')

@router.get("/video_endpoint")
def video_endpoint():
    return StreamingResponse(predictor.stream, media_type='multipart/x-mixed-replace; boundary=frame')


@router.post("/")
def video_inference_api(request: Request,
    file: UploadFile = File(...),
    action: str = Form(...),
    minio: MinioServer = Depends(get_minio_instance),
    kafka: Kafka = Depends(get_kafka_instance),
    yolo: YoloBase = Depends(get_yolo_instance)
    ):

    temp = NamedTemporaryFile(delete=False)
    if action == 'OPEN':
        try:
            contents = file.file.read()
            with temp as f:
                f.write(contents);
        except Exception:
            return {"message": "There was an error uploading the file"}
        finally:
            file.file.close()
    
        iterf = predictor.create_stream(yolo, temp.name, minio, kafka)  # Pass temp.name to VideoCapture()
        if iterf == 'no_camera':
            return templates.TemplateResponse('stream.html', {
                    "request": request,
                    "res": False,
                    "no_camera": True
                })
        else:
            return templates.TemplateResponse('stream.html', {
                    "request": request,
                    "res": True,
                    "no_camera": False
                })
        
    if action == 'STOP':
        try:
            os.remove(temp.name)
        except Exception as e:
            print(e)
        return templates.TemplateResponse('stream.html', {
                "request": request,
                "res": False,
                "no_camera": False
            })

@router.get("/")
def video(request: Request):
    return templates.TemplateResponse('stream.html', {
            "request": request,
            "res": False,
            "no_camera": False
        })

