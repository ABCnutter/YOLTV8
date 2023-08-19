import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "."))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "utils"))

from utils.n_cls_nms_plus import apply_nms


def convert_coordinates(
    txt_label_path, output_file_dir, iou_threshold, confidence_threshold, area_weight, slice_sep
):
    # txt_file_path: 存放 YOLOv8 小图检测结果的 TXT 文件的上级路径
    # output_file_path: 变换后的结果存放的 TXT 文件路径
    if not os.path.exists(output_file_dir):
        os.makedirs(output_file_dir)
        print(f"Created folder {output_file_dir}")
    output_lines = dict()  # 存储转换后的结果

    # orgimg_w = 0
    # orgimg_h = 0
    # 遍历文件夹中的每个 TXT 文件
    for root, dirs, files in os.walk(txt_label_path):
        for index, filename in enumerate(files):
            if filename.endswith(".txt"):
                filepath = os.path.join(root, filename)

                # 解析文件名中的信息
                slice_info = filename.split(".")[0].split(slice_sep)
                y0 = int(slice_info[-6])
                x0 = int(slice_info[-5])
                sliceHeight = int(slice_info[-4])
                sliceWidth = int(slice_info[-3])
                orgimg_w = int(slice_info[-2])
                orgimg_h = int(slice_info[-1])

                exclude_imgname_char = slice_sep + str(y0) + slice_sep + str(x0) + slice_sep + str(sliceHeight) + \
                slice_sep + str(sliceWidth) + slice_sep + str(orgimg_w) + slice_sep + str(orgimg_h)
                exclude_imgname_index = filename.split(".")[0].index(exclude_imgname_char)
                imgname = filename.split(".")[0][:exclude_imgname_index]

                # 读取小图检测结果的坐标信息
                with open(filepath, "r") as f:
                    lines = f.readlines()

                # 将边界框坐标转换到原图的坐标空间，并将结果存储到列表中
                converted_lines = []
                for line in lines:
                    class_label, x, y, w, h, conf = line.strip().split(" ")
                    x = float(x) * sliceWidth
                    y = float(y) * sliceHeight
                    w = float(w) * sliceWidth
                    h = float(h) * sliceHeight

                    # # 未归一化，因为后文需要
                    # x_in_original = (float(x) + x0) / orgimg_w
                    # y_in_original = (float(y) + y0) / orgimg_h
                    # w_in_original = float(w) / orgimg_w
                    # h_in_original = float(h) / orgimg_h
                    # converted_line = f"{class_label} {x_in_original} {y_in_original} {w_in_original} {h_in_original} {conf}\n"

                    x_in_original = float(x) + x0
                    y_in_original = float(y) + y0
                    w_in_original = float(w)
                    h_in_original = float(h)
                    converted_line = [
                        int(class_label),
                        x_in_original,
                        y_in_original,
                        w_in_original,
                        h_in_original,
                        float(conf),
                        orgimg_w,
                        orgimg_h
                    ]

                    converted_lines.append(converted_line)

                # 将转换后的结果添加到输出列表中
                if imgname not in output_lines.keys():
                    output_lines[imgname] = converted_lines
                else:
                    output_lines[imgname].extend(converted_lines)
                    
    # print(f'orgimg_w-------{orgimg_w}')
    # print(f'orgimg_h-------{orgimg_h}')

    outputs_file_path_list = []
    for key, value in output_lines.items():
        nms_output_lines = apply_nms(
            value,
            iou_threshold,
            confidence_threshold,
            area_weight,
        )

        # 将转换后的结果写入输出文件
        output_file_path = os.path.join(output_file_dir, f"{key}.txt")
        if os.path.exists(output_file_path):
            # import shutil
            import logging
            os.remove(output_file_path)
            logging.warning(f"completed predict txt results of image—{key} have been existed! The original content will be overwritten!")

        with open(output_file_path, "w") as f:
            f.writelines(nms_output_lines)
        print(f"completed predict txt results of image—{key} is saved at: {output_file_path}")
        outputs_file_path_list.append(output_file_path)

    return outputs_file_path_list
