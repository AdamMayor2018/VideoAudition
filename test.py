# @Time : 2022/10/26 16:26 
# @Author : CaoXiang
# @Description:
import os

from config.conf_loader import YamlConfigLoader
from ensembles.detect.yolov5_detector import Yolo5Detector
yaml_path = "config/config.yaml"
loader = YamlConfigLoader(yaml_path)
detector = Yolo5Detector(loader)
image_path = "image"
for name in os.listdir(image_path):
    print(name)