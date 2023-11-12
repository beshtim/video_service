from app.core.gateways.kafka import Kafka
from app.dependencies.kafka import get_kafka_instance

from app.core.gateways.minio import MinioServer
from app.dependencies.minio import get_minio_instance

from app.core.yolov5.detect_video import YoloBase
from app.dependencies.yolo import get_yolo_instance

from fastapi import APIRouter, Depends, UploadFile, File, Request
from tempfile import NamedTemporaryFile

from app.utils.helpers import plot_one_box, draw_poly, kafka_post, minio_post

from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import random

router = APIRouter(
    prefix="/pipeline",
    tags=["pipeline"],
    dependencies=[Depends(get_kafka_instance), Depends(get_minio_instance), Depends(get_yolo_instance)]
    )

colors = [tuple([random.randint(0, 255) for _ in range(3)]) for _ in range(100)] #for bbox plotting

@router.post("/")
def send(request: Request,
        # camera: Camera,
        file: UploadFile = File(...),
        minio: MinioServer = Depends(get_minio_instance),
        kafka: Kafka = Depends(get_kafka_instance),
        yolo: YoloBase = Depends(get_yolo_instance)
        ):
    
    def process_video(video, model):
        cap = cv2.VideoCapture(video)

        while cap.isOpened():
            ret, frame = cap.read()
            # frame_num = cap.get(cv2.CAP_PROP_POS_FRAMES)

            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            img_batch = np.expand_dims(frame, axis=0)
            out = model.yolo(img_batch) 
            print(out)

        cap.release()

    temp = NamedTemporaryFile(delete=False)
    try:
        contents = file.file.read()
        with temp as f:
            f.write(contents);
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()

    res = process_video(temp.name, yolo)  # Pass temp.name to VideoCapture()
        
    return res
