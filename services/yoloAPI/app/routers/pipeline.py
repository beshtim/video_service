import json

from app.core.gateways.kafka import Kafka
from app.core.models.message import Message
from app.dependencies.kafka import get_kafka_instance

from app.core.gateways.minio import MinioServer
from app.dependencies.minio import get_minio_instance

from app.core.yolov5.detect_video import YoloBase
from app.dependencies.yolo import get_yolo_instance


from fastapi import APIRouter, Depends, UploadFile, File, Query, Request, Form
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
from tempfile import NamedTemporaryFile
from typing import List

from datetime import datetime
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import os

router = APIRouter(
    prefix="/pipeline",
    tags=["pipeline"],
    dependencies=[Depends(get_kafka_instance), Depends(get_minio_instance), Depends(get_yolo_instance)]
    )

@router.post("/")
def send(request: Request,
    file: UploadFile = File(...),
    minio: MinioServer = Depends(get_minio_instance),
    kafka: Kafka = Depends(get_kafka_instance),
    yolo: YoloBase = Depends(get_yolo_instance)
    ):
    
    def process_video(video, model, kafka, minio):

        def minio_post(image_bytes, minio):
            try:
                found = minio.client.bucket_exists("test")
                if not found:
                    minio.client.make_bucket("test")
                else:
                    print("Bucket already exists")

                current_datetime = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                im_name = f'{current_datetime}.jpg'

                minio.client.put_object("test", im_name, image_bytes, content_type='image/jpeg', length=-1, part_size=10*1024*1024)
                # server.fput_object("test", "test.jpg", LOCAL_FILE_PATH)
                print("It is successfully uploaded to bucket")
            except Exception as e:
                raise e
            return im_name

        def kafka_post(dict_, server):
            try:
                topic_name = server._topic
                server.producer.send(topic_name, dict_)
            except Exception as e:
                raise e
            return 'Message sent successfully'

        cap = cv2.VideoCapture(video)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            batched_frame = np.expand_dims(frame, axis=0)

            out = model.yolo(batched_frame)
            print(out)

            if len(out[0]) > 0: # TODO
                print("ADD LOGIC HERE")
                    # img = Image.fromarray(frame).convert('RGB')
                    # out_img = BytesIO()
                    # img.save(out_img, format='png')
                    # out_img.seek(0)

                    # im_name = minio_post(out_img, minio)
                    # out.append(im_name)
                    # message = kafka_post(out, kafka)
                    # print(message)

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

    res = process_video(temp.name, yolo, kafka, minio)  # Pass temp.name to VideoCapture()
        
    return res
