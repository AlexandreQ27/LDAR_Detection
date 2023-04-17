import cv2
import os

# 设置视频文件路径
video_path = "video_path"

# 设置图像保存路径
output_folder = "output_folder"

# 创建输出文件夹（如果不存在）
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 打开视频文件
cap = cv2.VideoCapture(video_path)

# 设置计数器
count = 0

# 循环遍历每一帧并保存为图像
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # 每10帧保存一次图像
        if count%10==0:
            # 生成输出文件名
            output_filename = os.path.join(output_folder, f"frame{count}.jpg")
            # 保存图像
            cv2.imwrite(output_filename, frame)
        # 更新计数器
        count += 1
    else:
        break

# 释放资源
cap.release()
