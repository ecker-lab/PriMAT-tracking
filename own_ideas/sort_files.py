import numpy as np
import pandas as pd
import os
import re
#import matplotlib.pyplot as plt
from PIL import Image


from pathlib import Path
import json
import shutil

data_root = Path('~/Data/Monkey/02_VoTT/')

imgs = data_root
lbls = data_root
img_list = imgs.glob('**/*.png')
labels_list = lbls.glob('**/*.json')

i_monkey = 0
for label in labels_list:
    with open(label) as f:
        content = json.load(f)
        
    origin_path = Path(content["asset"]["path"])
    image_path_end = Path(origin_path.parent.name, origin_path.name)
    image_path = data_root / image_path_end
    img = np.asarray(Image.open(image_path))
    img_h, img_w, _ = img.shape
    

    all_boxes = content["regions"]
    # print(all_boxes[0])
    monkey_boxes = [box for box in all_boxes if "monkey" in box["tags"]]
    if monkey_boxes:
        shutil.copy(image_path, (data_root / 'images' /img_path.name))

    label_fpath = (data_root / "labels_with_ids" / image_path.name).with_suffix('.txt')

    for monkey_box in monkey_boxes:
        id = monkey_box["id"]
        h, w, left, top = monkey_box["boundingBox"].values()

        x_center = left + w/2
        y_center = top + h/2

        #  img_id = re.sub("[.]jpg","", img_id)
        #Label-String schreiben.

        label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                i_monkey, x_center / img_w, y_center / img_h, w / img_w, h / img_h)

        i_monkey += 1
        with label_fpath.open(mode='a') as f:
            f.write(label_str)
