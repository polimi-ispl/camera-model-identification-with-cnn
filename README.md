# First Steps Towards Camera Model Identification with Convolutional Neural Networks #
IEEE Signal Processing Letters ( Volume: PP, Issue: 99 )
[Available at IEEE] (http://ieeexplore.ieee.org/document/7786852/)

## Requirements ##
python2.7 with packages:

- [caffe](https://github.com/BVLC/caffe)
- opencv2
- numpy
- scipy
- multiprocessing
- tqdm

## Single image pipeline example ##
```
python2.7 single_image_pipeline.py demo/Kodak_M1063_0_10000.JPG
```

## Training from scratch ##
```python2.7 s01_download.py```

```python2.7 s02_extract_patches.py```

```python2.7 s03_split_generate.py```

```python2.7 s04_caffe_txt_generate.py```

```python2.7 s05_caffe_lmdb_generate.py```

```python2.7 s06_caffe_train.py```

```python2.7 s07_predict_patches.py```

```python2.7 s08_ppi_acc_plot.py```
