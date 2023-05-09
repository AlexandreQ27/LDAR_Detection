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

# import subprocess
# subprocess.call(["C:\Users\lifel\.conda\envs\monotest\python.exe", "C:\Users\lifel\Documents\Projects\YoloProjects\monodepth2\mono_test_video.py"])
# os.system("C:\Users\lifel\.conda\envs\monotest\python.exe C:\Users\lifel\Documents\Projects\YoloProjects\monodepth2\mono_test_video.py")

line_thickness=10
min_frames=3
# 定义类别标签
class_names = ['pipe11','pipe12','pipe13','pipe14','pipe15','pipe16','pipe121','pipe141','crevice11']
# 加载模型
weights = str(ROOT / 'best.pt')
device = select_device('')
model = attempt_load(weights, device)
model.eval()
# 定义检测函数
def detect(image):
    # 图像预处理
    img0 = image.copy()
    # 转换颜色通道顺序为RGB，通道数为3
    img = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    input_image = img
    img = img.transpose(2, 0, 1)  # 将通道维度放到第一维
    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # cv2.imwrite('cache/frame_{}.png'.format(frame_cnt), img0)
    # os.system(r"C:\Users\lifel\.conda\envs\monotest\python.exe C:\Users\lifel\Documents\Projects\YoloProjects\monodepth2\mono_test_image.py --image_path C:\Users\lifel\Documents\Projects\yolov5\cache\frame_{}.png --model_name mono+stereo_no_pt_640x192".format(frame_cnt))

    # 模型推理
    with torch.no_grad():
        pred = model(img)[0]
        pred = non_max_suppression(pred, 0.5, 0.45)
    pipe_center = None
    distance = None
    if pred[0].shape[0] != 0:
        # 保存检测帧的图片数组信息用于产生深度图
        np.save('cache2/frame_{}.npy'.format(frame_cnt), input_image)
        os.system(
            r"C:\Users\lifel\.conda\envs\monotest\python.exe C:\Users\lifel\Documents\Projects\YoloProjects\monodepth2\mono_test_npy.py --image_path C:\Users\lifel\Documents\Projects\yolov5\cache2\frame_{}.npy --model_name mono+stereo_no_pt_640x192".format(frame_cnt))

        # 处理检测结果
        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = det[:, :4].clamp(0, img0.shape[1])
                for *xyxy, conf, cls in reversed(det):
                    # print(cls)
                    label = f'{class_names[int(cls)]} {conf:.2f}'
                    gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    # xywh = torch.tensor(xyxy).view(1, 4)
                    # print(xywh)
                    for itor in range(len(class_names)):
                        if class_names[itor] == class_names[int(cls)]:
                            object_frames[class_names[itor]].append(1)
                            object_frames[class_names[itor]].pop(0)
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
            object_frames[class_names[itor2]].append(0)
            object_frames[class_names[itor2]].pop(0)
    return image, distance


# 加载视频并进行检测
cap = cv2.VideoCapture(str(ROOT / 'N01155150.mp4'))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# print(frame_width)
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# print(frame_height)
# 创建输出视频文件
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 25, (800, 480))
# 连续帧数记录器
object_frames = {class_name: [0] * 10 for class_name in class_names}
# 定义字典用于跟踪每个物体出现的次数
object_count = {class_name: 0 for class_name in class_names}

# 遍历每一帧
frame_cnt = 1
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 进行检测
    frame = cv2.resize(frame, (800, 480))
    result, distance = detect(frame)
    frame_cnt += 1

    # 在图像上绘制结果
    if distance is not None:
        cv2.putText(result, f'Distance: {distance:.2f}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    else:
        cv2.putText(result, 'Distance: N/A', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    out.write(result)

    # 显示结果
    cv2.imshow('result', result)

    # 等待用户按下ESC键退出
    if cv2.waitKey(1) == 27:
        break

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()