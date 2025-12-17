from PIL import Image
import cv2
import numpy as np
import os


def cv2pil(cv_image: np.ndarray) -> Image.Image:
    """
    Converts cv2 BGR image to PIL RGB image
    """
    rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_image)


def cxcywh2xywh(box):
    """
    Converts c_x, c_y, w, h (normalized) bbox to x_min, y_min, w, h (normalized) bbox.
    """
    x_c, y_c, w, h = box
    x_min = max(0, x_c - w / 2)
    y_min = max(0, y_c - h / 2)
    W = min(1, w)
    H = min(1, h)
    return np.array([x_min, y_min, W, H])


def calc_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Computes intersection over union for semantic segmentation

    Args:
        pred_mask: binary mask (H, W), uint8/book
        gt_mask:   binary mask (H, W), uint8/book

    Returns:
        iou: float [0,1]
    """
    # convert to bool
    pred_bool = pred_mask.astype(bool)
    gt_bool = gt_mask.astype(bool)

    # compute intersection and union
    intersection = np.logical_and(pred_bool, gt_bool)
    union = np.logical_or(pred_bool, gt_bool)

    intersection_sum = intersection.sum()
    union_sum = union.sum()

    # if both masks are empty
    if union_sum == 0:
        return 1.0 if intersection_sum == 0 else 0.0

    return float(intersection_sum / union_sum)


def get_bbox_prompts(label_path):
    """
    Extract bboxes from all .txt files in label path

    Params: label_path (str)
    Returns: dict {filename:[box1,box2,...]}
    boxes are in normalized xywh format
    """
    files = sorted(os.listdir(label_path))
    bboxes = {}
    for file in files:
        file_name, ext = os.path.splitext(file)
        # check if text file
        if not os.path.isfile("%s/%s" % (label_path, file)):
            continue
        if ext != ".txt":
            continue
        local_bboxes = []
        with open("%s/%s" % (label_path, file), "r") as f:
            boxes = f.read().split("\n")
            for box in boxes:
                coords = box.split(" ")
                if len(coords) != 5:
                    continue
                # format is class x_c y_c w h
                # skipping class and converting to xywh
                xywh_box = cxcywh2xywh(np.array(coords[1:], dtype=float))
                local_bboxes.append(xywh_box)
        bboxes[file_name] = local_bboxes
    return bboxes
