# ImageEncodings
Find original images of a subimage using CNNs

## Adding Datasets
- Make a dataset named as "ms_coco_resized128" and only add images in it of shape 128x128x3.
- Make another folder named as "test_resized128" and add one image resized to 64x64x3 that you want to test in it.
- You can use "test.py" to resize images and write them to folders.

## Usage
- First run "encoding.py" to generate dataset encodings from a keras CNN.
- Then run "test.py" to generate test encodings.
- Then finally execute "match.py" to get 4 best images that appear to have test image as part of them.
- You can compare the results with knn and kmeans also from "knn.py" and "kmeans.py"

## Requirements
- Python3
- Keras
- Tensorflow

## License
- It is a free tool to use.
- It is actually the implementation of this paper [Finding original Image of a subimage](https://arxiv.org/abs/1806.08078).
