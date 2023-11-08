import argparse
import csv
import os
import platform
import sys
from pathlib import Path
from typing import Any

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, smart_inference_mode
from utils.augmentations import letterbox
import numpy as np

class YoloPredictor:
    def __init__(self,
            weights=ROOT / 'yolov5s.pt',  # model path or triton URL
            data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
            imgsz=(640, 640),  # inference size (height, width)
            conf_thres=0.25,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            view_img=False,  # show results
            nosave=False,  # do not save images/videos
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            update=False,  # update all models
            exist_ok=False,  # existing project/name ok, do not increment
            line_thickness=3,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            half=False,  # use FP16 half-precision inference
            dnn=False,  # use OpenCV DNN for ONNX inference
            vid_stride=1,  # video frame-rate stride
        ) -> None:

        # Load model
        device = select_device(device)
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        self.__dict__.update(locals())

    def get_dataloader(self, input_data):
        source = str(input_data)
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
        screenshot = source.lower().startswith('screen')
        if is_url and is_file:
            source = check_file(source)  # download
        
        self.webcam = webcam
        # Dataloader
        self.bs = 1  # batch_size
        if webcam:
            view_img = check_imshow(warn=True)
            dataset = LoadStreams(source, img_size=self.imgsz, stride=self.stride, auto=self.pt, vid_stride=self.vid_stride)
            self.bs = len(dataset)
        elif screenshot:
            dataset = LoadScreenshots(source, img_size=self.imgsz, stride=self.stride, auto=self.pt)
        else:
            dataset = LoadImages(source, img_size=self.imgsz, stride=self.stride, auto=self.pt, vid_stride=self.vid_stride)
        vid_path, vid_writer = [None] * self.bs, [None] * self.bs

        return dataset
    
    def base_transform(self, batch):
            for_stack = []
            for im0 in batch:
                im = letterbox(im0, self.imgsz, stride=self.stride, auto=self.pt)[0]  # padded resize
                im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
                im = np.ascontiguousarray(im)  # contiguous
                
                im = torch.from_numpy(im).to(self.device)
                im = im.half() if self.half else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                for_stack.append(im)
            return torch.stack(for_stack)


    @smart_inference_mode()
    def __call__(self, cv2_image_batch1):

        h, w, c = cv2_image_batch1[0].shape
        batch = self.base_transform(cv2_image_batch1)
        im0 = cv2_image_batch1[0]

        pred = self.model(batch, augment=False, visualize=False)
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
        # print(pred)
        out = []

        # Process predictions
        for i, det in enumerate(pred):  # per image
            annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
            if len(det): 
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(batch.shape[2:], det[:, :4], (h, w, c)).round()
                print(det)

                # ==========================================
                # # Print results
                # for c in det[:, 5].unique():
                #     n = (det[:, 5] == c).sum()  # detections per class
                #     s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # # Write results
                for *xyxy, conf, cls in reversed(det): # all detections
                    c = int(cls)  # integer class
                    label = self.names[c] if self.hide_conf else f'{self.names[c]}'
                    confidence = float(conf)
                    confidence_str = f'{confidence:.2f}'

                    # TODO plot res
                    c = int(cls)  # integer class
                    label = None if self.hide_labels else (self.names[c] if self.hide_conf else f'{self.names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))
                # ==========================================

                det_objcts = reversed(det).cpu().numpy()

                out.append({
                    'pred_boxes':   det_objcts[:,:4],
                    'scores':       det_objcts[:,4],
                    'pred_classes': det_objcts[:,5],
                })

        return out



    @smart_inference_mode()
    def standart_infer(self, input_data):

        dataset = self.get_dataloader(input_data=input_data)

        self.model.warmup(imgsz=(1 if self.pt or self.model.triton else self.bs, 3, *self.imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        for path, im, im0s, vid_cap, s in dataset:
            with dt[0]:
                im = torch.from_numpy(im).to(self.model.device)
                im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with dt[1]:
                # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                pred = self.model(im, augment=False, visualize=False)

            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                if self.webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path

                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
                if len(det): 
                    print(len(det))
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det): # all detections
                        c = int(cls)  # integer class
                        label = self.names[c] if self.hide_conf else f'{self.names[c]}'
                        confidence = float(conf)
                        confidence_str = f'{confidence:.2f}'

                        # TODO plot res
                        c = int(cls)  # integer class
                        label = None if self.hide_labels else (self.names[c] if self.hide_conf else f'{self.names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))

            LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

        # Print results
        t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *self.imgsz)}' % t)

class YoloBase:
    instance = None
    def __init__(
            self,
            weights='yolov5s.pt',
            ) -> None:
        
        self._weights = weights
        self.yolo = self.create_model()
        YoloBase.instance = self

    def create_model(self):
        return YoloPredictor(
                weights = self._weights
            )

if __name__ == '__main__':
    yp = YoloPredictor()
    yp("/raid/nanosemantics/CV/digitalRoads/gates_doors/yolo_del/del/data/images/zidane.jpg")