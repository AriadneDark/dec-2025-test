import numpy as np
import cv2
from utils import *
import distinctipy


def visualize(img, masks, alpha=0.3, colors=[]):
    """
    Visualizes segmentation masks.
    Params: img - cv2 image (h,w,c)
            masks - segmentation masks (np.array (N,c,h,w))
            alpha - blending coefficient
            colors: colors in format [(R, G, B), ...], values in [0, 1] range.
    Returns: semantic mask (cv2 image, bool), colored instance masks (cv2 image), image with mask overlay (cv2 image)
    """
    overlay = img.copy()
    full_mask = np.zeros_like(img)
    N = len(masks)
    # get visibly distinctive colors for masks
    if len(colors) == 0:
        colors = distinctipy.get_colors(N)
    for i in range(N):
        mask = masks[i]
        # make colored
        full_mask[mask] = np.array(colors[i]) * 255
    # get full black and white mask
    bitmap = full_mask.max(axis=2) > 0
    overlay[bitmap] = full_mask[bitmap]
    # overlayed image
    overlayed = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    return bitmap, full_mask, overlayed


def visualize_on_frames(
    image_path,
    outputs,
    prompt,
    prompt_idx,
    n_frames_to_draw=5,
    draw_on_video=False,
    max_objects=200,
):
    """
    Visualize predicted masks on video frames
    Params:
        image_path - path to folder with video frames (in format 0001.jpg, 0002.jpg, etc)
        outputs - Sam3 predictor outputs
        prompt - prompt dict, can contain bounding_boxes in normalized xywh format
        prompt_idx - frame index for which prompt was used
        n_frames_to_draw - bbox will be drawn on n frames before and after prompt frame
        draw_on_video - draw or not prompt bboxes on video
        max_objects - max number of objects.
    Returns:
        prompt_img - cv2 image, prompt with prompt bbox
        visuals - list of cv2 images, frames overlayed with predicted masks

    """
    image_list = sorted(os.listdir(image_path))
    # get n visually distinctive colors
    colors = np.array(distinctipy.get_colors(max_objects))
    visuals = []
    prompt_boxes = []
    if "bounding_boxes" in prompt.keys():
        boxes = prompt["bounding_boxes"]
        for box in boxes:
            prompt_boxes.append(box)
    for i in range(len(outputs)):
        img = cv2.imread("%s/%s" % (image_path, image_list[i]))
        H, W, c = img.shape
        if i == prompt_idx:
            # drawing prompt bbox on prompt frame
            prompt_img = img.copy()
            for box in prompt_boxes:
                x, y, w, h = box
                cv2.rectangle(
                    prompt_img,
                    (int(x * W), int(y * H)),
                    (int((x + w) * W), int((y + h) * H)),
                    color=(0, 255, 0),
                    thickness=10,
                )
        # drawing prompt bbox on frames
        if np.abs(i - prompt_idx) < n_frames_to_draw and draw_on_video:
            for box in prompt_boxes:
                x, y, w, h = box
                cv2.rectangle(
                    img,
                    (int(x * W), int(y * H)),
                    (int((x + w) * W), int((y + h) * H)),
                    color=(0, 255, 0),
                    thickness=10,
                )
        masks = outputs[i]["out_binary_masks"]
        # indices for tracking, to keep the same instance in all overlayed images
        idx = outputs[i]["out_obj_ids"]
        # get overlayed frame
        _, _, vis = visualize(img, masks, 0.3, colors[idx])
        visuals.append(vis)
    return prompt_img, visuals


def put_prompt_on_frame(prompt, frame_):
    """
    Puts text on frames
    Params:
        prompt - dict with text or bbox prompts
        frame_ - cv2 image
    Returns:
        frame with prompt text
    """
    frame = frame_.copy()
    # visualization params
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255, 255, 255)
    font_scale = 3
    thickness = 4
    H, W, c = frame.shape
    # text prompt case
    if "text" in prompt.keys():
        pr_text = "Prompt:"
        text = prompt["text"]
        # max length of string in symbols
        text_maxlen = 18
        n_lines = int(len(text) / text_maxlen) + 1
        # put word "Prompt"
        (pr_text_width, pr_text_height), baseline = cv2.getTextSize(
            pr_text, font, font_scale, thickness
        )
        cv2.putText(
            frame,
            pr_text,
            (int((W - pr_text_width) / 2), H - 100 * (n_lines + 1)),
            font,
            font_scale,
            color,
            thickness,
        )
        words = text.split(" ")
        line = ""
        i_line = 0
        # put prompt strings
        for i, word in enumerate(words):
            if len(line + " " + word) > text_maxlen:
                (text_width, text_height), baseline = cv2.getTextSize(
                    line, font, font_scale, thickness
                )
                cv2.putText(
                    frame,
                    line,
                    (int((W - text_width) / 2), H - 100 * (n_lines - i_line)),
                    font,
                    font_scale,
                    color,
                    thickness,
                )
                i_line += 1
                line = word
            else:
                line += " " + word
            if i == len(words) - 1:
                (text_width, text_height), baseline = cv2.getTextSize(
                    line, font, font_scale, thickness
                )
                cv2.putText(
                    frame,
                    line,
                    (int((W - text_width) / 2), H - 100 * (n_lines - i_line)),
                    font,
                    font_scale,
                    color,
                    thickness,
                )
    else:
        # bbox prompt case
        pr_text = "Prompt: bbox"
        (pr_text_width, pr_text_height), baseline = cv2.getTextSize(
            pr_text, font, font_scale, thickness
        )
        cv2.putText(
            frame,
            pr_text,
            (int((W - pr_text_width) / 2), H - 100),
            font,
            font_scale,
            color,
            thickness,
        )
    return frame
