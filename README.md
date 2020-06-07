# Thyroid Ultrasound Images Segmentation

## Introduction

With ultrasound, a number of soft-tissues (parenchymal), for example, thyroid
formations can be completely visualized, the state of which has important diagnostic value.
In thyroid the pathology can be diffused (affecting the entire
organ/structure), focal or mixed.
With the development of computer vision technologies, research has become
possible at the junction of the areas of deep learning and biomedicine to solve
such problems as medical image segmentation [[1, 2]](#references).
Today, analysis of thyroid formations is carried out by each doctor personally
and its information content is extremely operator-dependent. Our aim is to
provide them with algorithms that would highlight those formations in 2D US
images.
The ultrasound method is a manual method of arbitrary positioning by the
operator’s hand (hand-free) and, unlike CT and MRI (recording on a disc in
DICOM format), it is not standardized from a tomographic (full organ review)
point of view. At the same time, on modern ultrasonic devices it is possible to
save individual images (ultrasound slices).

## Purpose

Ultrasound is a form of radiation that, like X-rays, is useful for medical imaging because of a good balance between its penetration of and interaction with the body [[3]](#references). But the exact physical model of wave propagation through the insides of a person is still unknown. First of all we investigate different UNet-like models to solve thyroid ultrasound images segmentation problem. There are various approaches to the construction of such models [[4, 5]](#references). Our main goal is to take into account the relationship between neighboring ultrasound slices, so we proposed several methods aimed at this. We also compared our results with basic and state-of-the-art methods and suggest related idea for learnable data processing layers.

## Data

Dataset consists of pairs (image, mask): thyroid nodules (2D slices segmented
by experts) from National Center for Endocrinology.

![Data sample example](supplementary_files/sample_example.jpg)

## Method

What makes this project interesting for us is to account for
space information from neighbor (nearest) slices of space. Convolutional neural
networks have shown themselves very well in various tasks of computer vision.
So we have compiled an algorithm that uses the symbiosis of both ideas.

## Evaluation

To evaluate the quality of segmentation, we used the Dice coefficient, which is often used in biomedicine, and the weighted Dice coefficient, based on the Generalized Dice Loss (GDL) [[6]](#references). It can be seen that a weighted metric gives a more trustworthy result, since it takes into account the occupied area of the white spot (formation), relative to the black (background).

## Usage

### Requirements

The list of required libraries (can be installed by for example `pip install <package name>` or `conda install <package name>`):  

- `crfseg` - [Developed by team “2.5D Neural Networks for Medical Image Segmentation” in similar task](https://pypi.org/project/crfseg/)
- `numpy` (tested on 1.18.1)
- `pandas` (tested on 0.25.1)
- `pillow` (tested on 6.1.01)
- `pytorch` (tested on 1.3.1)
- `tensorboard` (tested on 2.1.1)
- `torchvision` (tested on 0.4.2)
- `opencv` (tested on opencv-python-headless 4.2.0.32)


### Prepare data
Raw data should be organized in a following way:   
```
raw_data
│
├───Patient_1
│   │
│   ├───Images
│   │   ├───001.tif
│   │   ├───002.tif
│   │   └─── ...
│   │
│   └───Masks
│       ├───001.labels.tif
│       ├───002.labels.tif
│       └─── ...
│
├───Patient_2
│   │
│   ...
...
```

- To preprocess data run:  

```python data_preprocessing.py --config_file="configs/config_preprocessing.txt"```  

- Where `config_preprocessing.txt` contains (including ":" and "#"):

```
root: <path to git repo> # root
raw data: <raw data> # path to folder relative to the root with raw data
preprocessed data: <data> # path to the folder relative to the root where preprocessed data will be
```

As a result, you will receive a `<data>` folder with prepared images with the same structure as the `<raw data>` folder, as well as a csv table `dataset.csv` with a description of the dataset.

### Train network

- To train the network after preparing your data run this command:  
```python train.py --config_file="config_train.txt"```

- Where `config_train.txt` contains (including ":" and "#"):

```
root: <path to git repo> # root
experiment_name: <Your experiment name> # experiment name, if None default name will be created
image size: <height, width> # model image input size (e.g. 256, 256)
batch size: <batch size> # batch size (e.g. 4)
lr: <learning rate> # learning rate (e.g. 0.001)
epochs number: <epochs number> # number of epochs (e.g. 40)
log dir: <path to log dir> # path to the folder relative to the root where the results of experiments from the tensorboard will be recorded
checkpoint_path: checkpoints/<checkpoint name> # checkpoint path if continue training, else None
```

---
**NOTE**

1. In file `train.py` you can change models (do not forget to import them if they are not imported already).

2. We compared our models with the state of the art architecture from [Deep Attentive Features for Prostate
Segmentation in 3D Transrectal Ultrasound](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8698868) paper. Since our data is not really 3d data we considered batch dimension of our original tensors as the third dimension of 3d images. 
To train it, you should change file `train.py` to `Train_DAF3D.py` in the command above.

---

#### List of available models 
##### in file `models.py`
1. ```UNet(n_channels, n_classes)``` - vanilla UNet with 2 down steps;
2. ```Unet_with_attention_right(n_channels, n_classes, height, width)``` - UNet with attention after each up step;
3. ```UNetTC<i>(n_channels, n_classes)``` - UNet with history concatenation in skip connections (TC stands for Triple Cat; `<i>` means number of down steps, available options are: "" for 2 down steps as in vanilla, "3" and "4" for 3 and 4 down steps correspondingly); 
4. ```UNetFourier(n_channels, n_classes, image_size, fourier_layer)``` - UNet with Fourier layers in skip connections;
5. ```UNet_crf(n_channels, n_classes)``` - UNet with [conditional random fields based layer](https://pypi.org/project/crfseg/) on top;
6. ```UNetCLSTMed()``` - UNet with convolutional LSTM cells in skip connections;

- Where:
	- `n_channels` states for number of color channels in the input image; 
	- `n_classes` - number of classes for output pixels (2 for binary segmentation);
	- `height, width`/`image_size` - spatial dimensions of the input image;
	- `fourier_layer` - type of transform in frequency space (`linear` or `non-linear`);

##### in file `DAF3D.py`
7. ```DAF3D()``` - model with [Deep Attentive Features](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8698868) [[7]](#references). Uses [ResNeXt](https://arxiv.org/pdf/1611.05431.pdf) [[8]](#references) as backbone and [FPN](https://arxiv.org/pdf/1612.03144.pdf) [[9]](#references) to combine multi-level features utilizing 3D atrous spatial pyramid pooling (ASPP) on top.

### Perform inference

Test data should be organized in a following way:   
```
test
│
├───Patient_1
│   │
│   ├───Images
│   │   ├───001.tif
│   │   ├───002.tif
│   │   └─── ...
│   │
│   └───Masks
│       ├───001.labels.tif
│       ├───002.labels.tif
│       └─── ...
│
├───Patient_2
│   │
│   ...
...
```

- To perform inference with the trained network run this command:  
```python inference.py --config_file="configs/config_inference.txt"```

- Where `config_inference.txt` contains (including ":" and "#"):

```
root: <path to git repo> # root
model path: <model.pth> # path to the model relative to the root
image size: <height, width> # model image input size for the loading model (e.g. 256, 256)
test data: <test> # path to the folder relative to the root with test data

```

---
**NOTE**

Model is supposed to be saved by the command ```torch.save(model.state_dict(), best_model_path)```

---


After running the above command folders `Animations` and `Predicted_Masks` will be created in the `test` folder:   
```
test
│
├───Patient_1
│   │
│   ├───Images
│   │   ├───001.tif
│   │   ├───002.tif
│   │   └─── ...
│   │
│   ├───Masks
│   │   ├───001.labels.tif
│   │   ├───002.labels.tif
│   │   └─── ...
│   │
│   ├───Animations
│   │   ├───001.gif
│   │   ├───002.gif
│   │   └─── ...
│   │
│   └───Predicted_Masks
│       ├───001.tif
│       ├───002.tif
│       └─── ...
│
├───Patient_2
│   │
│   ...
...
```

## Examples

- Gif image of prediction for UNetTC4 (UNet triple cat with 4 down steps) model:

![Inference example gif](supplementary_files/I0000028.gif)

<p style="text-align: center;">(US image, ground truth, prediction)</p>

- some predicted masks and ground truth masks for the same model:

![Inference example png 1](supplementary_files/inference1.PNG)
![Inference example png 2](supplementary_files/inference2.PNG)
![Inference example png 3](supplementary_files/inference3.PNG)
![Inference example png 4](supplementary_files/inference4.PNG)

## References

[1] Song, Junho & Chai, Young Jun & Masuoka, Hiroo & Park, Sun-Won &
Kim, Su-jin & Choi, June & Kong, Hyoun-Joong & Lee, Kyu Eun & Lee,
Joongseek & Kwak, Nojun & Yi, Ka & Miyauchi, Akira. (2019). Ultrasound
image analysis using deep learning algorithm for the diagnosis of thyroid
nodules. Medicine. 98. e15133. 10.1097/MD.0000000000015133.  
[2] Wang, Lei & Yang, Shujian & Yang, Shan & Zhao, Cheng & Tian, Guangye
& Gao, Yuxiu & Chen, Yongjian & Lu, Yun. (2019). Automatic thyroid
nodule recognition and diagnosis in ultrasound imaging with the YOLOv2
neural network. World Journal of Surgical Oncology. 17. 10.1186/s12957-
019-1558-z.  
[3] M. A. Flower. Webb’s Physics of Medical Imaging, Second Edition.  
[4] Azad, Reza & Asadi, Maryam & Fathy, Mahmood & Escalera, Sergio.
(2019). Bi-Directional ConvLSTM U-Net with Densley Connected Convolu-
tions.  
[5] Alom, Md. Zahangir & Hasan, Mahmudul & Yakopcic, Chris & Taha, Tarek
& Asari, Vijayan. (2018). Recurrent Residual Convolutional Neural Network
based on U-Net (R2U-Net) for Medical Image Segmentation.  
[6] Sudre, Carole & Li, Wenqi & Vercauteren, Tom & Ourselin, S´ebastien &
Cardoso, Manuel Jorge. (2017). Generalised Dice overlap as a deep learning
loss function for highly unbalanced segmentations.  
[7] Wang Y, Dou H, Hu X, et al. (2019) Deep Attentive Features for Prostate Segmentation
 in 3D Transrectal Ultrasound.  
[8] Xie, Saining et al. (2017) Aggregated Residual Transformations for Deep Neural Networks.
[9] Lin, Tsung-Yi et al. (2017) Feature Pyramid Networks for Object Detection.
