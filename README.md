# YOLTV8 — 大尺度图像目标检测框架

针对大尺度图像（如遥感影像、大尺度工业检测图像等），由于设备的限制，无法利用图像直接进行模型训练。将图像裁剪至小尺度进行训练，再将训练结果进行还原拼接是解决该问题的普遍思路。YOLT项目（）是该思路具体实现，其以改进的YOLOV2作为检测框架，通过重叠裁剪预测处理以及对目标检测框拼接还原结果进行NMS过滤实现大尺度遥感影像的小型目标检测。但在具体方案操作时，本项目作者发现该方法存在以下几点问题：

1. 无法较好地同时性地解决拼接结果中不同类别物体重叠检测框的精确过滤，尤其是位于图像边缘的不完整物体的检测框，会牺牲一定的检测精度。

2. 由于裁剪造成图像中的大型物体被分割于数块图像中，存在无法在单张影像中完整捕获物体的缺陷

3. 所使用的YOLOV2检测框架已经较为落后，已无法满足现在任务场景对检测精度的需求。

因此，本项目以最新的YOLOV8为检测框架，增设多尺度，多信息的预处理模块，捕获大尺度图像的多尺度上下文信息，能够有效识别出大尺度图像的大小型识别物体以及密集型检测目标。另外，此次我们还对对原始NMS算法进行改进，以满足不同类型物体以及重叠框（尤其是位于边缘的检测框）的过滤，实现大尺度影像的精确检测。