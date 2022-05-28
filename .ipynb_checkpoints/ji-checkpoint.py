#-----------------------------------------------------------------------#
#   predict.py将单张图片预测、摄像头检测、FPS测试和目录遍历检测等功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
#-----------------------------------------------------------------------#
import json
import time
import copy
import cv2
from PIL import Image
import numpy as np
from yolo import YOLO
from utils.utils import get_classes

classes=get_classes("ev_sdk/src/model_data/class.txt")
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

def init():
    yolo = YOLO()
    yolo.origin(classes,
                '/project/train/best.pth')
    return yolo
def process_image(net, input_image, args=None):
    image = Image.fromarray(input_image)
    image,results = net.detect_image(image)
    print("result:",results)
    label=classes
    alert=[]
    count=0
    is_alert=False
#     判断是否报警
    for [name,score,xmin,ymin,xmax,ymax] in results:
        if name in label:
            alert.append([name,score,xmin,ymin,xmax,ymax])
            count=count+1
            is_alert=True

#        [name,score,xmin,ymin,xmax,ymax]

    # 写入数据
    target_info=[]
    for [name,score,xmin,ymin,xmax,ymax] in alert:
        target_info.append(
            {
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "confidence": score,
                "name": name
            }
        )
    fake_result = {}
    fake_result["algorithm_data"] = {
        "is_alert": is_alert,
        "target_count": count,
        "target_info": target_info
    }

    fake_result["modeldata"] = {}
    objects=[]
    for [name, score, xmin, ymin, xmax, ymax] in results:
        objects.append(
            {
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "confidence": score,
                "name": name
            }
        )
    fake_result["modeldata"]["objects"] = objects
    print(fake_result)
    return json.dumps(fake_result, indent=4,cls=MyEncoder)


if __name__ == "__main__":
    net=init()
    input_image=np.array(Image.open("/home/data/1043/ZDSmask20220524_V2_train_kitchen_2_000792.jpg "))
    jj=process_image(net, input_image, args=None)
    print(jj)