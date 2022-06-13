import os
import glob
from pixellib.semantic import semantic_segmentation
import tqdm
import pdb
import pickle
import numpy as np
from PIL import Image
segment_image = semantic_segmentation()

segment_image.load_ade20k_model("deeplabv3_xception65_ade20k.h5")
model = segment_image.model2
images = glob.glob("./**/*/*.JPG", recursive=True)
images = [x for x in images if "South-Building"  in x]
images = [x for x in images if "reduced" not in x]

masks = [x.replace('images','masks_high_dim').rstrip('.JPG')+'.pkl' for x in images]
# masks = masks[::-1]
# images = images[::-1]

def get_mask(model, image_path, mask_path):
    if os.path.exists(mask_path):
        return None
    image = np.array(Image.open(image_path))     
    w, h, n = image.shape
    resized_image = np.array(Image.fromarray(image.astype('uint8')).resize((int(h), int(w))))
    mean_subtraction_value = 127.5
    resized_image = (resized_image/mean_subtraction_value) -1
    trained_image_width = int(512*np.ceil(resized_image.shape[0]/512))
    trained_image_height = int(512*np.ceil(resized_image.shape[1]/512))
    pad_x = int(trained_image_width - resized_image.shape[0])
    pad_y = int(trained_image_height - resized_image.shape[1])
    resized_image = np.pad(resized_image, ((0, pad_x), (0, pad_y), (0, 0)), mode='constant')
    res= np.zeros((1,int(trained_image_width), int(trained_image_height), 151),dtype=float)
    for i in range(int(trained_image_width//512)):
        for j in range(int(trained_image_height//512)):
            img = resized_image[512*(i):512*(i+1),512*(j):512*(j+1),:]
            res[0, 512*(i):512*(i+1),512*(j):512*(j+1),:] = model.predict(np.expand_dims(img, 0))
    if pad_x > 0:
      res = res[:,:-pad_x]
    if pad_y > 0:
      res = res[:,:,:-pad_y]
    return res



for img, mask in zip(tqdm.tqdm(images), masks):
    labels = get_mask(model, img,mask)
    if labels==None:
        print("Already reached")
        continue

    with open(mask,"wb") as f:
        pickle.dump(labels,f)