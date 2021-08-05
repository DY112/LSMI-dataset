# Large Scale Multi-Illuminant (LSMI) Dataset for Developing White Balance Algorithm under Mixed Illumination (ICCV 2021)

<!-- TABLE OF CONTENTS -->
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>


<!-- ABOUT THE PROJECT -->
## About
[[Project site]](https://dykim.ml/publication/lsmi/) [[Arxiv]]() [[Download Dataset]](https://www.kaggle.com/ciplab/datasets)

This is an official repository of ICCV 2021 paper, **"Large Scale Multi-Illuminant (LSMI) Dataset for Developing White Balance Algorithm under Mixed Illumination"**.

This repository provides  
1. Preprocess code of "Large Scale Multi Illuminant (LSMI) Dataset"
2. Code of Pixel-level illumination inference U-Net
3. Pre-trained model parameter for testing U-Net

## Requirements
Our running environment is as follows:

- Python version 3.8.3
- Pytorch version 1.7.0
- CUDA version 11.2

We provide a docker image, which supports all extra requirements, including specified version of python, pytorch, CUDA above.

You can download the docker image [here](https://hub.docker.com/r/dongyoung95/torch1.7_lsmi).

In an environment that meets the above requirements, run the following code to install additional requirements:
```sh
pip install requirements.txt
```


<!-- GETTING STARTED -->
## Getting Started
### Download the LSMI dataset
You should first download the LSMI dataset from [here](https://www.kaggle.com/ciplab/datasets).

The dataset is composed of 3 sub-folers named "galaxy", "nikon", "sony".

Folders named by each camera include several scenes, and each scene folder contains full-resolution RAW files and JPG files that is converted to sRGB color space.

Move the downloaded dataset to each folder in the cloned repository (galaxy,nikon,sony).

### Convert raw type images into tiff type

1. Clone the repo
   ```sh
   git clone https://github.com/DY112/LSMI-dataset.git
   ```
2. Install NPM packages
   ```sh
   npm install
   ```



<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_



<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/DY112/LSMI-dataset/issues) for a list of proposed features (and known issues).

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Your Name - [@twitter_handle](https://twitter.com/twitter_handle) - email

Project Link: [https://github.com/DY112/LSMI-dataset](https://github.com/DY112/LSMI-dataset)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

* []()
* []()
* []()


