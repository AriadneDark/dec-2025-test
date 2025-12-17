import json
import re
import os

data_path = "datasets/"
big_path = "%s/BIG/test/" % (data_path)


def build_big_manifest() -> None:
    """Makes json file with BIG data"""
    image_names = os.listdir(big_path)
    files = []
    for image_name in image_names:
        name, ext = os.path.splitext(image_name)
        parts = name.split("_")
        # check if image: ends with im
        # name is like 'nums_nums_o_prompt_im.jpg'
        if len(parts) < 5:
            continue
        if parts[-1] != "im":
            continue
        prompt = parts[-2]
        full_img_name = "%s/%s" % (big_path, image_name)
        full_mask_name = "%s/%s_gt.png" % (big_path, "_".join(parts[:-1]))
        # check mask exists
        if os.path.exists(full_mask_name):
            filedict = {"img": full_img_name, "mask": full_mask_name, "prompt": prompt}
            files.append(filedict)
    with open("%s/BIG.json" % (data_path), "w") as file:
        json.dump(files, file, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    build_big_manifest()
