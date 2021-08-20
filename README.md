# Large Scale Multi-Illuminant (LSMI) Dataset for Developing White Balance Algorithm under Mixed Illumination (ICCV 2021)

<!-- ABOUT THE PROJECT -->
## About
[[Project site]](https://dykim.me/publication/lsmi/) [[Arxiv]]() [[Download Dataset]](https://drive.google.com/drive/folders/14RKJRiG0R5njERnrrsQePz30EJc08Sxd?usp=sharing)

This is an official repository of **"Large Scale Multi-Illuminant (LSMI) Dataset for Developing White Balance Algorithm under Mixed Illumination"**, which is accepted as a poster in ICCV 2021.

This repository provides  
1. Preprocessing code of "Large Scale Multi Illuminant (LSMI) Dataset"
2. Code of Pixel-level illumination inference U-Net
3. Pre-trained model parameter for testing U-Net

## Requirements
Our running environment is as follows:

- Python version 3.8.3
- Pytorch version 1.7.0
- CUDA version 11.2

We provide a docker image, which supports all extra requirements (ex. dcraw,rawpy,tensorboard...), including specified version of python, pytorch, CUDA above.

You can download the docker image [here](https://hub.docker.com/r/dongyoung95/torch1.7_lsmi).

The following instructions are assumed to run in a docker container that uses the docker image we provided.

<!-- GETTING STARTED -->
## Getting Started
### Clone this repo
In the docker container, clone this repository first.

```sh
git clone https://github.com/DY112/LSMI-dataset.git
```

### Download the LSMI dataset
You should first download the LSMI dataset from [here](https://www.kaggle.com/ciplab/datasets).

The dataset is composed of 3 sub-folers named "galaxy", "nikon", "sony".

Folders named by each camera include several scenes, and each scene folder contains full-resolution RAW files and JPG files that is converted to sRGB color space.

Move all three folders to the root of cloned repository.

### Preprocess the LSMI dataset

0. Convert raw images to tiff files  
   
   To convert original 1-channel bayer-pattern images to 3-channel RGB tiff images, run following code:

   ```sh
   python 0_cvt2tiff.py
   ```
   You should modify **SOURCE** and **EXT** variables properly.

   The converted tiff files are generated at the same location as the source file.

1. Make mixture map
   ```sh
   python 1_make_mixture_map.py
   ```
   Change the **CAMERA** variable properly to the target directory you want.
   
   .npy tpye mixture map data will be generated at each scene's directory.

2. Crop
   ```sh
   python 2_preprocess_data.py
   ```

   The image and the mixture map are resized as a square with a length of the **SIZE** variable inside the code, and the ground-truth image is also generated.

   We set the size to **256** to test the U-Net, and **512** for train the U-Net.

   Here, to test the pre-trained U-Net, set size to 256.

   The new dataset is created in a folder with the name of the CAMERA_SIZE. (Ex. galaxy_256)

### Use U-Net for pixel-level AWB

You can download pre-trained model parameter [here](https://drive.google.com/drive/folders/1m0Jt6vTkRJi_iMDDnhcW79QmE48MGVUP?usp=sharing).

Pre-trained model is trained on 512x512 data with random crop & random pixel level relighting augmentation method.

Locate downloaded **models** folder into **SVWB_Unet**.

- Test U-Net  
  ```sh
  cd SVWB_Unet
  sh test.sh
  ```

- Train U-Net
  ```sh
  cd SVWB_Unet
  sh train.sh
  ```


<!-- 
## Acknowledgements

* []()
* []()
* []()
 -->

