# @Time : 2022/10/20 21:32 
# @Author : CaoXiang
# @Description: 目标检测子模块
import typing
from abc import ABCMeta, abstractmethod
import copy
import numpy as np


class Target(metaclass=ABCMeta):
    def __init__(self, box: typing.Sequence[int], conf: float):
        self._bbox = box
        self._conf = conf

    def check_box(self):
        """
            检查传入box的坐标是否遵循【xmin,ymin,xmax,ymax的格式】
        :return:
        """
        assert self._bbox[0] < self._bbox[2] and self._bbox[1] < self._bbox[3],\
            "xmin should smaller than xmax, check whether input boxs is [xmin, ymin, xmax, ymax]"

    @property
    def height(self) -> float:
        return self._bbox[3] - self._bbox[1]

    @property
    def width(self) -> float:
        return self._bbox[2] - self._bbox[0]

    @property
    def bbox(self) -> typing.Sequence[int]:
        return self._bbox

    @bbox.setter
    def bbox(self, new_bbox):
        self._bbox = new_bbox

    @property
    def conf(self) -> float:
        return self._conf

    @property
    def center(self) -> typing.Tuple[float, float]:
        """
            返回检测目标的中心坐标
        :return:
        """
        return (self._bbox[0] + self._bbox[2]) / 2, (self._bbox[1] + self._bbox[3]) / 2

    def get_shape(self) -> typing.Tuple[float, float]:
        """
        :return: width and height of box
        """
        return self.width, self.height

    @property
    def area(self) -> float:
        return self.width * self.height


class Face(Target):
    bbox: typing.Sequence[int]
    conf: float

    def __init__(self, bbox: typing.Sequence[int], conf: float):
        """
        :param bbox: face box coordinate
        :param conf: confidence
        """
        super().__init__(bbox, conf)

    def __str__(self):
        return f"x0: {self._bbox[0]}, x1: {self._bbox[2]}, y0: {self._bbox[1]}, y1: {self._bbox[3]}" \
               f", confidence: {self._conf}"


class BaseDetResults(metaclass=ABCMeta):
    def __init__(self, targets: typing.Union[typing.Sequence[Target], None]):
        self._targets = list(sorted(targets, key=lambda target: target.bbox[0])) if targets is not None else None
        self.has_target = True if self._targets else False
        self.start_num = 0

    @property
    def targets(self) -> typing.Union[typing.Sequence[Target], None]:
        return self._targets

    def __len__(self):
        return len(self._targets) if self._targets is not None else 0

    def __bool__(self):
        return bool(len(self._targets) if self._targets is not None else 0)

    def __iter__(self):
        return self

    def __next__(self):
        if self.start_num > len(self):
            raise StopIteration()
        target = self._targets[self.start_num]
        self.start_num += 1
        return target

    def __getitem__(self, index):
        return self._targets[index]


class FaceDetResults(BaseDetResults):
    pass


class BaseDetector(metaclass=ABCMeta):
    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def detect_batch_images(self, images:typing.Sequence[np.ndarray]):
        pass