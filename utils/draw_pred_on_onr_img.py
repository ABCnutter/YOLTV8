import os
import sys
import cv2
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "."))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "utils"))

def draw_predictions_on_image(
    image_path, results_file_path, class_labels, class_names, completed_output_path
):
    # 确保类别标签和类别名称数量一致
    assert len(class_labels) == len(
        class_names
    ), "Number of class labels should match the number of class names."

    # 定义每个类别对应的颜色
    colors = [
        # (255, 0, 0),  # head: 红色
        # (0, 255, 0),  # boxholder: 绿色
        (0, 0, 255),  # greendevice: 蓝色
        (255, 255, 0),  # baseholer: 青色
        # (0, 255, 255),  # circledevice: 黄色
        # (255, 0, 255),  # alldrop: 品红色
    ]

    # 定义类别标签到类别名称的映射
    label_map = dict(zip(class_labels, class_names))

    # 定义每个类别对应的颜色
    color_map = dict(zip(class_labels, colors))

    # 读取原图像
    image = cv2.imread(image_path)

    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 读取存放变换结果的文本文件
    with open(results_file_path, 'r') as file:
        lines = file.readlines()

    # 遍历每行结果
    for line in lines:
        line = line.strip().split(' ')
        class_label, x, y, w, h, conf = map(float, line)

        # 计算边界框的坐标
        image_height, image_width, _ = image.shape
        abs_x = int(x * image_width)
        abs_y = int(y * image_height)
        abs_w = int(w * image_width)
        abs_h = int(h * image_height)
        x_min = abs_x - abs_w // 2
        y_min = abs_y - abs_h // 2
        x_max = abs_x + abs_w // 2
        y_max = abs_y + abs_h // 2

        # 获取类别名称和颜色
        class_name = label_map.get(int(class_label), "Unknown")
        color = color_map[int(class_label)]

        # 绘制边界框和类别标签
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.putText(
            image,
            f"{class_name}: {conf:.2f}",
            (x_min, y_min - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    # 保存绘制结果
    filename = os.path.basename(image_path)
    if not os.path.exists(completed_output_path):
        os.makedirs(completed_output_path)

    output_image_path = os.path.join(completed_output_path, filename)
    if os.path.exists(output_image_path):
        # import shutil
        import logging
        os.remove(output_image_path)
        logging.warning(f"completed predict visual result of image-{filename} have been existed! The original content will be overwritten!")

    cv2.imwrite(output_image_path, image)
    print(f"completed predict visual result of image-{filename} is saved at: {output_image_path}")


# def generate_color_map(num_classes):
#     # 生成一组固定的颜色，确保同一类别使用相同的颜色, 颜色版通用函数
#     random.seed(42)
#     colors = []
#     for _ in range(num_classes):
#         color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
#         colors.append(color)

#     return colors
