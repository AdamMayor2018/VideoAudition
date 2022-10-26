# @Time : 2022/10/26 14:22 
# @Author : CaoXiang
# @Description: 画图相关功能 比如标记目标框、标记类别、人物名称等等
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import typing
import matplotlib.pyplot as plt

def draw_box(arr: np.ndarray, cords: typing.List[int], color: typing.Tuple[int, int, int],
             thickness: int) -> np.ndarray:
    """
        在原图上绘制出矩形框
    :param arr: 传入的原图ndarray
    :param cords: 框的坐标，按照【xmin,ymin,xmax,ymax】的方式进行组织
    :param color: 框的颜色
    :param thickness: 框线的宽度
    :return: 绘制好框后的图像仍然按照ndarray的数据格式s
    """
    assert len(cords) == 4, "cords must have 4 elements as xmin ymin xmax ymax."
    assert isinstance(arr, np.ndarray), "input must be type of numpy ndarray."
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    draw.rectangle(xy=cords, outline=color, width=thickness)
    img = np.array(img)
    return img


def draw_text(arr: np.ndarray, cords: typing.List[int], text:str, color:typing.Tuple[int, int, int],
              thickness: int) -> np.ndarray:
    """
        在原图上绘制文字类的信息
    :param arr: 传入的原图ndarray
    :param cords: 框的坐标，按照【xmin,ymin,xmax,ymax】的方式进行组织
    :param text: 需要打印的文字
    :param color: 框的颜色
    :param thickness: 框线的宽度
    :return: 绘制好框后的图像仍然按照ndarray的数据格式
    """
    assert len(cords) == 2, "cords must have 2 elements as xmin ymin."
    assert isinstance(arr, np.ndarray), "input must be type of numpy ndarray."
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    draw.text(xy=cords, fill=color, width=thickness, text=text)
    img = np.array(img)
    return img


if __name__ == '__main__':
    arr = np.ones((124, 124, 3)).astype(np.uint8) * 255
    arr = draw_box(arr, cords=[10, 10, 30, 30], color=(255, 0, 0), thickness=2)
    arr = draw_text(arr, cords=[50, 50], color=(0, 255, 0), thickness=2, text="Hello!")
    plt.figure()
    plt.imshow(arr)
    plt.show()


