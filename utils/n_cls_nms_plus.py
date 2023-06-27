import os
import sys
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "."))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "utils"))


def calculate_area(box):
    """
    计算边界框的面积
    box的格式：[xmin, ymin, xmax, ymax]
    """
    x1, y1, x2, y2 = box
    area = (x2 - x1) * (y2 - y1)
    return area


def calculate_iou(box1, box2):
    """
    计算两个边界框的IoU（Intersection over Union）
    box1和box2的格式：[xmin, ymin, xmax, ymax]
    """
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    # 计算交集的坐标
    x_left = max(x1, x3)
    y_top = max(y1, y3)
    x_right = min(x2, x4)
    y_bottom = min(y2, y4)

    if x_right < x_left or y_bottom < y_top:
        # 两个边界框没有交集
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # 计算并集的面积
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)
    union_area = box1_area + box2_area - intersection_area

    iou = intersection_area / union_area
    return iou


def nms_per_class(
    boxes, scores, classes, iou_threshold, confidence_threshold, area_weight
):
    """
    使用NMS对不同类别的边界框进行后处理
    boxes: 边界框列表，每个边界框的格式为 [xmin, ymin, xmax, ymax]
    scores: 每个边界框的置信度得分列表
    classes: 每个边界框的类别列表
    threshold: 重叠度阈值，高于该阈值的边界框将被抑制
    """
    # 过滤置信度低于阈值的边界框
    filtered_indices = np.where(np.array(scores) >= confidence_threshold)[0]
    boxes = [boxes[i] for i in filtered_indices]
    scores = [scores[i] for i in filtered_indices]
    classes = [classes[i] for i in filtered_indices]

    # 将边界框、置信度、类别转换为NumPy数组
    boxes = np.array(boxes)
    scores = np.array(scores)
    classes = np.array(classes)
    areas = np.array([calculate_area(box) for box in boxes])

    # 初始化空列表来存储保留的边界框索引
    keep_indices = []

    # 获取所有唯一的类别标签
    unique_classes = np.unique(classes)

    for cls in unique_classes:
        # 获取属于当前类别的边界框索引
        cls_indices = np.where(classes == cls)[0]

        # 根据当前类别的置信度得分和面积对边界框进行排序
        sorted_indices = np.lexsort(
            (scores[cls_indices], areas[cls_indices] * area_weight)
        )[::-1]
        # sorted_indices = np.argsort(areas[cls_indices])[::-1]
        cls_indices = cls_indices[sorted_indices]
        while len(cls_indices) > 0:
            # 选择当前得分最高的边界框
            current_index = cls_indices[0]
            current_box = boxes[current_index]
            keep_indices.append(filtered_indices[current_index])

            # 计算当前边界框与其他边界框的IoU
            other_indices = cls_indices[1:]
            ious = np.array(
                [calculate_iou(current_box, boxes[i]) for i in other_indices]
            )

            # 找到重叠度低于阈值的边界框索引
            low_iou_indices = np.where(ious < iou_threshold)[0]

            # 更新剩余边界框索引
            cls_indices = cls_indices[1:][low_iou_indices]

    return keep_indices


def apply_nms(
    outputs, iou_threshold, confidence_threshold, area_weight
):
    # 将边界框列表转换为NumPy数组
    outputs = np.array(outputs)

    boxes = []
    scores = []
    class_ids = []
    for out in outputs:
        x = out[1]
        y = out[2]
        w = out[3]
        h = out[4]
        score = out[5]
        class_id = int(out[0])
        orgimg_w = int(out[6])
        orgimg_h = int(out[7])

        # 计算边界
        left = float(x - w / 2)
        top = float(y - h / 2)
        right = float(x + w / 2)
        bottom = float(y + h / 2)

        # Add the class ID, score, and box coordinates to the respective lists
        class_ids.append(class_id)
        scores.append(score)
        boxes.append([left, top, right, bottom])

    # 应用NMS
    # indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold, nms_threshold)
    indices = nms_per_class(
        boxes=boxes,
        scores=scores,
        classes=class_ids,
        iou_threshold=iou_threshold,
        confidence_threshold=confidence_threshold,
        area_weight=area_weight,
    )

    # 选择通过NMS过滤后的边界框
    nms_out_lines = []
    for i in indices:
        # Get the box, score, and class ID corresponding to the index
        box = boxes[i]
        score = scores[i]
        class_id = class_ids[i]
        x = float(box[0] + (box[2] - box[0]) / 2) / orgimg_w
        y = float(box[1] + (box[2] - box[0]) / 2) / orgimg_h
        w = float(box[2] - box[0]) / orgimg_w
        h = float(box[3] - box[1]) / orgimg_h
        nms_out_line = f"{class_id} {x} {y} {w} {h} {score}\n"
        nms_out_lines.append(nms_out_line)
    return nms_out_lines