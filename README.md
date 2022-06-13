# 3DVision_SemanticSFM

## 1. Hloc
Make sure the Hloc repository and this repository are in the same directory.
Make sure to install Hloc with all its dependencies.
https://github.com/cvg/Hierarchical-Localization

## 2. Download the Dataset
Download the dataset into datasets/ and unzip it:
```
!wget http://cvg.ethz.ch/research/local-feature-evaluation/South-Building.zip -P datasets/
!unzip -q datasets/South-Building.zip -d datasets/
```

## 3. Ground Truth
Put the file ground_truth_adjusted.txt into the South-Building folder.

## 4. Segmentation
The overall method needs masks. The masks are stored as pkl file which can be generated from the segmentation model. You can download the weights that have been used from here: [weights link](https://github.com/ayoolaolafenwa/PixelLib/releases/download/1.3/deeplabv3_xception65_ade20k.h5).
Once the masks are generated, put the masks folder into the South-Building path.

## 5. Notebook
Open the Jupyter Notebook to run the code.
