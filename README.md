# 3DVision_SemanticSFM

## 1. Hloc
Make sure the Hloc repository and this repository are in the same directory.
Make sure to install Hloc with all its dependencies.
https://github.com/cvg/Hierarchical-Localization


## 2. Download the Dataset
Download the dataset into datasets/ and unzip it:
```
if not images.exists():
    !wget http://cvg.ethz.ch/research/local-feature-evaluation/South-Building.zip -P datasets/
    !unzip -q datasets/South-Building.zip -d datasets/
```

## 3. Ground Truth
Put the ground truth file into the South-Building folder.

## 4. Notebook
Open the Notebook to run the code.
