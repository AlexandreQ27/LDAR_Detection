import cv2
import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression
from utils.torch_utils import select_device

# 定义类别标签
class_names = ['pipe', 'probe']

# 加载模型
weights = 'yolov5s.pt'
device = select_device('')
model = attempt_load(weights)
model.eval()
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
        print(pred)
        pred = non_max_suppression(pred, 0.25, 0.45)
    pipe_center = None
    probe_center = None
    distance = None
    if pred is not None:
        # 处理检测结果
        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = det[:, :4].clamp(0, img0.shape[1])
                for *xyxy, conf, cls in reversed(det):
                    label = f'{class_names[int(cls)]} {conf:.2f}'
                    xywh = torch.tensor(xyxy).view(1, 4)
                    xywh[:, [0, 2]] /= img0.shape[1]
                    xywh[:, [1, 3]] /= img0.shape[0]
                    c1, c2 = (xywh[:, :2] + xywh[:, 2:]) / 2.0
                    if class_names[int(cls)] == 'pipe':
                        pipe_center = (c1.item(), c2.item())
                    else:
                        probe_center = (c1.item(), c2.item())
        # 计算距离
        if probe_center is not None and pipe_center is not None:
            distance = ((probe_center[0] - pipe_center[0])**2 + (probe_center[1] - pipe_center[1])**2)**0.5
    return image, distance

# 加载视频并进行检测
cap = cv2.VideoCapture('C:/Users/14471/Desktop/N01155150/N01155150.mp4')
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(frame_width)
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(frame_height)
# 创建输出视频文件
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 25, (800, 480))

# 遍历每一帧
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 进行检测
    frame = cv2.resize(frame, (800, 480))
    result, distance = detect(frame)

    # 在图像上绘制结果
    if distance is not None:
        cv2.putText(result, f'Distance: {distance:.2f}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    else:
        cv2.putText(result, 'Distance: N/A', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    out.write(result)

    # 显示结果
    #cv2.imshow('result', result)

    # 等待用户按下ESC键退出
    if cv2.waitKey(1) == 27:
        break

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()
