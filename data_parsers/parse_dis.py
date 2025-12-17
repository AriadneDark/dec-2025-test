import json
import re
import os


def extract_prompt_from_filename_dis(filename: str) -> str:
    """
    Extracts prompt from filename
    Examples:
        "#1#cat#2#BlackCat#abc" → "Black Cat"
        "#5#tech#3#UAV#xyz" → "UAV"
    """
    parts = filename.split("#")
    # corrupted filename case
    if len(parts) < 5:
        return ""
    raw_prompt = parts[3]
    # it can be CamelCase words or word like TV
    words = re.findall("[A-Z][a-z]+", raw_prompt)
    if len(words) == 0:
        # TV, UAV, etc.
        return raw_prompt
    else:
        # adding spaces between words
        return " ".join(words)


data_path = "datasets/"
dis_path = "%s/DIS5K/" % (data_path)
dis_subdirs = ["DIS-TR", "DIS-VD", "DIS-TE1", "DIS-TE2", "DIS-TE3", "DIS-TE4"]


def build_dis5k_manifest() -> None:
    """Makes json file with DIS5K data"""
    files = []
    for subdir in dis_subdirs:
        # image folder
        imdir = "%s/%s/im/" % (dis_path, subdir)
        # semantic masks folder
        maskdir = "%s/%s/gt/" % (dis_path, subdir)
        # get image names
        image_names = os.listdir(imdir)
        for image_name in image_names:
            name, ext = os.path.splitext(image_name)
            prompt = extract_prompt_from_filename_dis(name)
            full_img_name = "%s/%s" % (imdir, image_name)
            full_mask_name = "%s/%s.png" % (maskdir, name)
            # check if mask exists
            if os.path.exists(full_mask_name):
                # check if prompt is ok
                if len(prompt) > 0:
                    filedict = {
                        "img": full_img_name,
                        "mask": full_mask_name,
                        "prompt": prompt,
                    }
                    files.append(filedict)
    with open("%s/DIS5K.json" % (data_path), "w") as file:
        json.dump(files, file, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    build_dis5k_manifest()
