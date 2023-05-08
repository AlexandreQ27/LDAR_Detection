import cv2
import os
import sys
from pathlib import Path
import numpy as np 
import torch
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from models.experimental import attempt_load
from utils.general import non_max_suppression
from utils.torch_utils import select_device
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode
line_thickness=10
min_frames=2
# 定义类别标签
class_names = ['pipe11', 'pipe12','pipe13','pipe14','pipe15','pipe16','pipe121','pipe141','crevice11']
# 加载模型
weights = '/Users/daizhicheng/Documents/Projects/YoloProjects/LDAR_Detection/yolov5/best.pt'
device = select_device('')
model = attempt_load(weights)
model.eval()

# 连续帧数记录器 辅助队列的角色 检测往队末append '1' 
array_len = 10
min_frames = 7  #threshold
object_frames = {class_name: [0] * array_len for class_name in class_names}  

# 定义检测函数
def detect(image):
    # 图像预处理
    img0 = image.copy()
    # 转换颜色通道顺序为RGB，通道数为3
    img = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)  # 将通道维度放到第一维
    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # 模型推理
    with torch.no_grad():
        pred = model(img)[0]
        pred = non_max_suppression(pred, 0.5, 0.45)
    pipe_center = None
    distance = None
    if pred[0].shape[0] != 0:
        # 处理检测结果
        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = det[:, :4].clamp(0, img0.shape[1])
                for *xyxy, conf, cls in reversed(det):
                    #print(cls)
                    label = f'{class_names[int(cls)]} {conf:.2f}'
                    gn = torch.tensor(img0.shape)[[1, 0, 1, 0]] 
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    #xywh = torch.tensor(xyxy).view(1, 4)
                    #print(xywh)
                    for itor in range(len(class_names)):
                        if class_names[itor] == class_names[int(cls)]:
                            object_frames[class_names[int(itor)]].append(1)
                            object_frames[class_names[int(itor)]].pop(0)
                            if sum(object_frames[class_names[int(cls)]]) >= 2 * min_frames:
                                # 计算距离
                                pipe_center = (xywh[0], xywh[1])
                                distance = ((pipe_center[0] - 0.5) ** 2 + (pipe_center[1] - 0.5) ** 2) ** 0.5
                                if distance < 0.2:
                                    bbox = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                                    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)  # 绘制方框
                                    cv2.putText(image, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                                (0, 255, 0), 2)  # 绘制类别标签
                        else:
                            object_frames[class_names[int(itor)]].append(0)
                            object_frames[class_names[int(itor)]].pop(0)
    else:
        for itor2 in range(len(class_names)):
            object_frames[class_names[int(itor2)]].append(0)
            object_frames[class_names[int(itor2)]].pop(0)
    return image, distance

# 加载视频并进行检测
cap = cv2.VideoCapture('E:/CollegeEra/Junior/JuniorSpring/LDAR_Detection/N01155150/N01155150.mp4')
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#print(frame_width)
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#print(frame_height)

# 创建输出视频文件
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_main = cv2.VideoWriter('output_main.mp4', fourcc, 25, (800, 480))
#检测出内容的视频合集
out_detect = cv2.VideoWriter('detected.mp4', fourcc, 25, (800, 480))
#创建黑色帧 用于间隔 
black_frame = np.zeros((480, 640, 3), dtype=np.uint8)

# 遍历每一帧
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 进行检测
    frame = cv2.resize(frame, (800, 480))
    result, distance = detect(frame)
    
    # 显示实时检测效果
    cv2.imshow('result', result)

    # 在图像上绘制检测框
    if distance is not None:
        cv2.putText(result, f'Distance: {distance:.2f}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        out_detect.write(frame)
        # for _ in range(100):  # 发现效果不佳 暂时注释了
        #     out_detect.write(black_frame)
    else:
        cv2.putText(result, 'N/A', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    out_main.write(result)

    # 等待用户按下ESC键退出
    if cv2.waitKey(1) == 27:
        break

# 释放资源
cap.release()
out_main.release()
out_detect.release()
cv2.destroyAllWindows()
