import os
import sys
import argparse
from pprint import pprint
from tqdm import tqdm
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "."))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "utils"))
from utils.slice_images import slice_image
from utils.convert_coordinates import convert_coordinates
from utils.draw_pred_on_onr_img import draw_predictions_on_image


def predict(
    # clip args
    images_dir=os.path.join(PROJECT_ROOT, 'dataset', 'predict', 'init_images'),
    outdir_slice_ims=os.path.join(PROJECT_ROOT, 'dataset', 'predict', 'slice_images'),
    project_name="sersor_detect",
    im_ext=".JPG",
    sliceHeight=1088,
    sliceWidth=1088,
    overlap=0.6,
    slice_sep='_',
    overwrite=False,
    out_ext='.png',
    # infer args
    model=r"E:\CS\GitHubClone\ultralytics\checkpoint\best.pt",
    conf=0.25,  # object confidence threshold for detection
    iou=0.7,  # intersection over union (IoU) threshold for NMS
    half=False,  # use half precision (FP16)
    device=None,  # device to run on, i.e. cuda device=0/1/2/3 or device=cpu
    show=False,  # show results if possible
    save=True,  # save images with results
    save_txt=True,  # save results as .txt file
    save_conf=True,  # save results with confidence scores
    save_crop=False,  # save cropped images with results
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidence scores
    max_det=300,  # maximum number of detections per image
    vid_stride=False,  # video frame-rate stride
    line_width=None,  # The line width of the bounding boxes. If None, it is scaled to the image size.
    visualize=False,  # visualize model features
    augment=False,  # apply image augmentation to prediction sources
    agnostic_nms=False,  # class-agnostic NMS
    retina_masks=False,  # use high-resolution segmentation masks
    classes=None,  # filter results by class, i.e. class=0, or class=[0,2,3]
    boxes=True,  # Show boxes in segmentation predictions
    # regress args
    output_file_dir=os.path.join(PROJECT_ROOT, 'results', 'completed_txt'),
    iou_threshold=0.01,
    confidence_threshold=0.6,
    area_weight=5,
    # draw args
    class_labels=[0, 1, 2, 3, 4, 5],
    class_names=[
            "head",
            "boxholder",
            "greendevice",
            "baseholer",
            "circledevice",
            "alldrop",
        ],
    completed_output_path=os.path.join(PROJECT_ROOT, 'results', 'completed_predict')
):
    im_list = [z for z in os.listdir(images_dir) if z.endswith(im_ext)]

    if not os.path.exists(os.path.join(outdir_slice_ims, project_name)):
        os.makedirs(os.path.join(outdir_slice_ims, project_name))
    else:
        import shutil
        shutil.rmtree(os.path.join(outdir_slice_ims, project_name))
        os.makedirs(os.path.join(outdir_slice_ims, project_name))
        print(f"{os.path.join(outdir_slice_ims, project_name)} is existed! The original content will be overwritten!!")

        # slice images
    for i, im_name in tqdm(enumerate(im_list)):
        im_path = os.path.join(images_dir, im_name)
        print("=========================== ", im_name, "--", i, "/", len(im_list), " =========================== ")

        slice_image(
            im_path,
            project_name,
            outdir_slice_ims,
            sliceHeight=sliceHeight,
            sliceWidth=sliceWidth,
            overlap=overlap,
            slice_sep=slice_sep,
            overwrite=overwrite,
            out_ext=out_ext,
        )
    yolov8_predict_results_path = os.path.join(PROJECT_ROOT, 'results', 'yolov8_detect', project_name)

    if os.path.exists(yolov8_predict_results_path):
        import shutil
        import logging
        shutil.rmtree(yolov8_predict_results_path)
        logging.warning(f"detect predict path: {yolov8_predict_results_path} is existed! The original content will be overwritten!")
        
    predict_shell = (
        'yolo predict model={} source={} project={} name={} conf={} iou={} half={} device={} show={} ' \
        'save={} save_txt={} save_conf={} save_crop={} hide_labels={} hide_conf={} ' \
        'max_det={} vid_stride={} line_width={} visualize={} augment={} agnostic_nms={} ' \
        'retina_masks={} classes={} boxes={}'.format(
            model,
            os.path.join(outdir_slice_ims, project_name),
            f'results/yolov8_detect',
            project_name,
            conf,
            iou,
            half,
            device,
            show,
            save,
            save_txt,
            save_conf,
            save_crop,
            hide_labels,
            hide_conf,
            max_det,
            vid_stride,
            line_width,
            visualize,
            augment,
            agnostic_nms,
            retina_masks,
            classes,
            boxes,
        )
    )
    pprint(f"prefict shell: {predict_shell}")
    os.system(predict_shell)

    txt_label_path = os.path.join(yolov8_predict_results_path, 'labels')

    txt_regress_path_list = convert_coordinates(
        txt_label_path=txt_label_path,
        output_file_dir=os.path.join(output_file_dir, project_name),
        iou_threshold=iou_threshold,
        confidence_threshold=confidence_threshold,
        area_weight=area_weight,
        slice_sep=slice_sep
    )
    for txt_regress_path in txt_regress_path_list:
        image_name = os.path.basename(txt_regress_path).split('.')[0]
        image_path = os.path.join(images_dir, image_name + im_ext)
        draw_predictions_on_image(
        image_path=image_path,
        results_file_path=txt_regress_path,
        class_labels=class_labels,
        class_names=class_names,
        completed_output_path=os.path.join(completed_output_path, project_name),
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=str, default=os.path.join(PROJECT_ROOT, 'dataset', 'predict', 'init_images'))
    parser.add_argument("--outdir_slice_ims", type=str, default=os.path.join(PROJECT_ROOT, 'dataset', 'predict', 'slice_images'))
    parser.add_argument("--project_name", type=str, default="sersor_detect")
    parser.add_argument("--im_ext", type=str, default=".JPG")
    parser.add_argument("--sliceHeight", type=int, default=1088)
    parser.add_argument("--sliceWidth", type=int, default=1088)
    parser.add_argument("--overlap", type=float, default=0.6)
    parser.add_argument("--slice_sep", type=str, default="_")
    parser.add_argument("--overwrite", type=bool, default=False)
    parser.add_argument("--out_ext", type=str, default=".png")
    parser.add_argument("--model", type=str, default=r"E:\CS\GitHubClone\ultralytics\checkpoint\best.pt")
    parser.add_argument("--conf", type=float, default=0.25)  # object confidence threshold for detection
    parser.add_argument("--iou", type=float, default=0.7)  # intersection over union (IoU) threshold for NMS
    parser.add_argument("--half", type=bool, default=False)  # use FP16 half-precision inference
    parser.add_argument("--device", type=str, default=None)  # cuda device, i.e. 0 or 0,1,2,3 or
    parser.add_argument("--show", type=bool, default=False)  # show results
    parser.add_argument("--save", type=bool, default=True)  # save images with results
    parser.add_argument("--save_txt", type=bool, default=True)  # save results"
    parser.add_argument("--save_conf", type=bool, default=True)
    parser.add_argument("--save_crop", type=bool, default=False)  # save cropped prediction boxes
    parser.add_argument("--hide_labels", type=bool, default=False)  # hide labels
    parser.add_argument("--hide_conf", type=bool, default=False)
    parser.add_argument("--max_det", type=int, default=300)  # maximum detections per image
    parser.add_argument("--vid_stride", type=bool, default=False)  # video frame-rate stride
    parser.add_argument("--line_width", type=float, default=None)
    parser.add_argument("--visualize", type=bool, default=False)
    parser.add_argument("--augment", type=bool, default=False)
    parser.add_argument("--agnostic_nms", type=bool, default=False)
    parser.add_argument("--retina_masks", type=bool, default=False)
    parser.add_argument("--classes", type=int, nargs="+", default=None)
    parser.add_argument("--boxes", type=bool, default=True)
    parser.add_argument("--output_file_dir", type=str, default=os.path.join(PROJECT_ROOT, 'results', 'completed_txt'))
    parser.add_argument("--iou_threshold", type=float, default=0.01)
    parser.add_argument("--confidence_threshold", type=float, default=0.6)
    parser.add_argument("--area_weight", type=float, default=5)
    parser.add_argument("--class_labels", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--class_names", type=str, nargs="+", default=[
            "head",
            "boxholder",
            "greendevice",
            "baseholer",
            "circledevice",
            "alldrop",
        ])
    parser.add_argument("--completed_output_path", type=str, default=os.path.join(PROJECT_ROOT, 'results', 'completed_predict'))

    agrs = parser.parse_args()
    predict(**vars(agrs))