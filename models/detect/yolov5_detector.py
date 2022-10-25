# @Time : 2022/10/25 16:49 
# @Author : CaoXiang
# @Description:
from config.conf_loader import YamlConfigLoader
from models.detect.yolov5 import non_max_suppression_face, scale_coords, check_img_size
from plugin.detection import BaseDetector, BaseDetResults, FaceDetResults, Face, Target
import torch
import cv2
import numpy as np
import copy
import typing
import torch.nn as nn
from model.common import Conv
from utils.datasets import letterbox


class Yolo5Detector(BaseDetector):
    """
        基于Yolo5Face的检测器
        配置信息从两部分获取，优先是从opt参数列表里面去检查属性和加载，如果没有找到，
        则会从配置文件的参数列表opt_dict里面去尝试加载，如果都没有，则生成报错。
    """
    def __init__(self, config_loader:YamlConfigLoader):
        self.model = None
        self.config_loader = config_loader
        self._weight_path = self.config_loader.attempt_load_param("weight_path")
        self._device = self.config_loader.attempt_load_param("device")
        self._img_size = self.config_loader.attempt_load_param("img_size")
        self._device = torch.device(f'{self._device}' if torch.cuda.is_available() else 'cpu')
        self._conf_thres = self.config_loader.attempt_load_param("conf_thres")
        self._iou_thres = self.config_loader.attempt_load_param("iou_thres")
        self._kept_conf = self.config_loader.attempt_load_param("kept_conf")
        self.load_model()

    def load_model(self):
        model = torch.load(self._weight_path, map_location=self._device)['model'].float().fuse().eval()
        for m in model.modules():
            if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
                # pytorch 1.7.0 compatibility
                m.inplace = True
            elif type(m) is Conv:
                #pytorch 1.6.0 compatibility
                m._non_persistent_buffers_set = set()
        self.model = model

    def detect_batch_images(self, images: typing.Sequence[np.ndarray]) -> typing.Sequence[BaseDetResults]:
        results = []
        inputs = []
        for image in images:
            img_cp = copy.deepcopy(image)
            h0, w0 = img_cp.shape[:2]
            r = self._img_size / max(h0, w0)
            if r != 1:
                interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
                img_cp = cv2.resize(img_cp, (int(w0 * r), int(h0 * r)), interpolation=interp)

            imgsz = check_img_size(self._img_size, s=self.model.stride.max())
            # letter box 外接矩形加灰条，auto=False是长边到imgsz， 然后再加灰条
            img_lb = letterbox(img_cp, new_shape=imgsz, auto=False)[0]
            # convert BGR -> RGB C,H,W
            #cv2.imwrite("xx.jpg", img_lb)
            img_lb = img_lb[:, :, ::-1].transpose(2, 0, 1).copy()
            inputs.append(img_lb)
        inputs = np.array(inputs)
        inputs = torch.from_numpy(inputs).to(self._device)
        inputs = inputs.float()
        inputs /= 255.0
        if inputs.ndimension() == 3:
            inputs = inputs.unsqueeze(0)
        preds = self.model(inputs)[0]
        preds = non_max_suppression_face(preds, self._conf_thres, self._iou_thres)
        for i, det in enumerate(preds):
            targets = []
            if det.shape[0] > 0:
                #print(inputs[i].shape, det.shape)
                det[:, :4] = scale_coords(inputs[i].shape[1:], det[:, :4], images[i].shape).round()
                #det[:, 5:15] = scale_coords_landmarks(img.shape[2:], det[:, 5:15], image.shape).round()
                det = det[:, 0:5]
                det = det.cpu().numpy()
                #det = det[np.argsort(det[:, 0])]
                bboxs = det[:, :4].tolist()
                confs = det[:, 4].tolist()
                for j, (box, conf) in enumerate(zip(bboxs, confs)):
                    if conf > self._kept_conf:
                        tar = Target(box, conf)
                        targets.append(tar)
            targets = targets if targets else []
            results.append(BaseDetResults(targets))
        return results