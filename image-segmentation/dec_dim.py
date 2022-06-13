import os
import glob
import tqdm
import pdb
import pickle
import numpy as np
import cv2
from PIL import Image


images = glob.glob("./**/*/*.JPG", recursive=True)
images = [x for x in images if "reduced" not in x]
images = [x for x in images if "capitole" in x]

trained_image_width=512
mean_subtraction_value=127.5
for image_path in tqdm.tqdm(images):
    image = np.array(Image.open(image_path))     

    # resize to max dimension of images from training dataset
    w, h, n = image.shape
    image_path_new = image_path.replace('images/','images_reduced/')
    if n > 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

    image_overlay = image.copy()

    ratio = float(trained_image_width) / np.max([w, h])
    resized_image = np.array(Image.fromarray(image.astype('uint8')).resize((int(ratio * h), int(ratio * w))))
    Image.fromarray(resized_image).save(image_path_new)
    print(image_path)
    print(image_path_new)
    print(w)
    print(h)