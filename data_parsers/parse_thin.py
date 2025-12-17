import json
import re
import os


def extract_prompt_from_filename_thin(filename: str) -> str:
    """
    Extracts prompt from filename
    Examples:
        "black_cat_PNG1" → "black cat"
    """
    find = re.search(r"(.*)(PNG)", filename)
    # corrupted filename case
    if not find:
        return ""
    # first_group
    raw_prompt = find[1]
    # replacing _ with spaces
    prompt = re.sub("_", " ", raw_prompt).strip()
    return prompt


data_path = "datasets/"
thin_path = "%s/ThinObject5K/" % (data_path)


def build_thin5k_manifest() -> None:
    """Makes json file with ThinObject5K data"""
    # image folder
    imdir = "%s/images/" % (thin_path)
    # semantic masks folder
    maskdir = "%s/masks/" % (thin_path)
    files = []
    image_names = os.listdir(imdir)
    for image_name in image_names:
        name, ext = os.path.splitext(image_name)
        # extracting prompt from name
        prompt = extract_prompt_from_filename_thin(name)
        full_img_name = "%s/%s" % (imdir, image_name)
        full_mask_name = "%s/%s.png" % (maskdir, name)
        # check mask exists
        if os.path.exists(full_mask_name):
            # check prompt is ok
            if len(prompt) > 0:
                filedict = {
                    "img": full_img_name,
                    "mask": full_mask_name,
                    "prompt": prompt,
                }
                files.append(filedict)
    with open("%s/ThinObject5K.json" % (data_path), "w") as file:
        json.dump(files, file, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    build_thin5k_manifest()
