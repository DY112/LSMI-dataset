# Large Scale Multi-Illuminant (LSMI) Dataset for Developing White Balance Algorithm under Mixed Illumination (ICCV 2021)

<img width="600" alt="스크린샷 2021-08-21 오후 3 30 22" src="https://user-images.githubusercontent.com/24367643/130312876-5b2955c2-0176-4e87-ba90-7c466fa3961b.png">

<!-- ABOUT THE PROJECT -->
## Dataset License
<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.

## About
[[Paper]](https://dykim.me/publication/lsmi/LSMI.pdf)
[[Project site]](https://dykim.me/publication/lsmi/) 
[[Download Dataset]](https://drive.google.com/drive/folders/1m0Jt6vTkRJi_iMDDnhcW79QmE48MGVUP?usp=sharing)
[[Video]](https://youtu.be/i8OAdYryig0)

This is an official repository of **"Large Scale Multi-Illuminant (LSMI) Dataset for Developing White Balance Algorithm under Mixed Illumination"**, which is accepted as a poster in ICCV 2021.

This repository provides  
1. Preprocessing code of "Large Scale Multi Illuminant (LSMI) Dataset"
2. Code of Pixel-level illumination inference U-Net
3. Pre-trained model parameter for testing U-Net

If you use our code or dataset, please cite our paper:
```
@inproceedings{kim2021large,
  title={Large Scale Multi-Illuminant (LSMI) Dataset for Developing White Balance Algorithm Under Mixed Illumination},
  author={Kim, Dongyoung and Kim, Jinwoo and Nam, Seonghyeon and Lee, Dongwoo and Lee, Yeonkyung and Kang, Nahyup and Lee, Hyong-Euk and Yoo, ByungIn and Han, Jae-Joon and Kim, Seon Joo},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={2410--2419},
  year={2021}
}
```

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
You should first download the LSMI dataset from [here](https://drive.google.com/drive/).

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
   
   Note that the side of the image will be cropped to make the image shape square.
   
   If you don't want to crop the side of the image and just want to resize whole image anyway, use **SQUARE_CROP=False**

   We set the default test size to **256**, and set train size to **512**, and **SQUARE_CROP=True**.

   The new dataset is created in a folder with the name of the CAMERA_SIZE. (Ex. galaxy_512)

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

