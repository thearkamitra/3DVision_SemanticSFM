import os
import glob
from pixellib.semantic import semantic_segmentation
import tqdm
import pdb
import pickle
segment_image = semantic_segmentation()

segment_image.load_ade20k_model("deeplabv3_xception65_ade20k.h5")

images = glob.glob("./**/*/*.JPG", recursive=True)
images = [x for x in images if "South"  in x]

masks = [x.replace('images','imgsOverlay') for x in images]
masks = masks[::-1]
images = images[::-1]
for img, mask in zip(tqdm.tqdm(images), masks):
    segment_image.segmentAsAde20k(image_path = img, output_image_name = mask, overlay=True)
    print(mask)
