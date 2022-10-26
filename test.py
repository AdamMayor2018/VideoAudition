# @Time : 2022/10/26 16:26 
# @Author : CaoXiang
# @Description:
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from config.conf_loader import YamlConfigLoader
from ensembles.detect.yolov5_detector import Yolo5Detector
from tools.draw import draw_box
yaml_path = "config/config.yaml"
loader = YamlConfigLoader(yaml_path)
detector = Yolo5Detector(loader)
image_path = "image"
images = []
names = []

for name in os.listdir(image_path):
    ipath = os.path.join(image_path, name)
    img = cv2.imread(ipath)
    images.append(img)
    names.append(name)

result = detector.detect_batch_images(images)
print(result)
for _, (img, res, name) in enumerate(zip(images, result, names)):
    for target in res.targets:
        img = img.astype(np.uint8)
        img = draw_box(img, target.bbox, (0, 0, 255), 2)
    img = img[:, :, ::-1]
    plt.imsave(f"image/img_{name}", img)

