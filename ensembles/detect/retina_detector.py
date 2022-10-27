# @Time : 2022/10/27 9:56 
# @Author : CaoXiang
# @Description: insightface的目标检测接口
import os

import numpy as np
import insightface
import cv2
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from PIL import Image
app = FaceAnalysis(allowed_modules=['detection'], providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
#img = ins_get_image('ldh')  # 不用带后缀，图片放到./insightface/python-package/insightface/data/images
img_dir = "/data/cx/VideoAudition/image/"
for name in os.listdir(img_dir):
    ipath = os.path.join(img_dir, name)
    img = np.array(Image.open(ipath))[:, :, ::-1]
    faces = app.get(img)
    print("faces::::", faces, len(faces))
    rimg = app.draw_on(img, faces)
    cv2.imwrite(img_dir + "insight_" + name, rimg)

