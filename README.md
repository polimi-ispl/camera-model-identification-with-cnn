# First Steps Towards Camera Model Identification with Convolutional Neural Networks #

IEEE Signal Processing Letters ( Volume: PP, Issue: 99 )

[Available at IEEE](http://ieeexplore.ieee.org/document/7786852/)

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

## Disclaimer notice ##
Users agree to the following restrictions on the utilization of this code:

- Redistribution: This code, in whole or in part, will not be further distributed, published, copied, or disseminated in any way or form whatsoever, whether for profit or not.
- Modification and Commercial Use: This code, in whole or in part, may not be modified or used for commercial purposes, without a formal agreement with the authors of the considered code.
- Citation: All documents and papers that report on activities that exploit this code must acknowledge its use by including an appropriate citation to the related publication.
- Indemnification: The authors are not responsible for any and all losses, expenses, damages, resulting from the use of this code.