# @Time : 2022/10/20 21:53 
# @Author : CaoXiang
# @Description: 目标特征抽取模块
from abc import abstractmethod, ABCMeta


class BaseEmbedder(metaclass=ABCMeta):
    @abstractmethod
    def load_model(self):
        pass

    def extract_batch_images(self, images):
        pass

