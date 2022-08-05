<p align="center">
  <img alt="🖼️_Image_Reconstructor" src="https://user-images.githubusercontent.com/62103572/183084803-ec31d4dd-8eff-4592-98a0-f2cf5f6bc7ab.png">
  <img alt="GitHub commit activity" src="https://img.shields.io/github/commit-activity/y/EliaFantini/Image-Reconstructor-FISTA-proximal-method-on-wavelets-trasnform">
  <img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/EliaFantini/Image-Reconstructor-FISTA-proximal-method-on-wavelets-trasnform">
  <img alt="GitHub code size" src="https://img.shields.io/github/languages/code-size/EliaFantini/Image-Reconstructor-FISTA-proximal-method-on-wavelets-trasnform">
  <img alt="GitHub repo size" src="https://img.shields.io/github/repo-size/EliaFantini/Image-Reconstructor-FISTA-proximal-method-on-wavelets-trasnform">
  <img alt="GitHub follow" src="https://img.shields.io/github/followers/EliaFantini?label=Follow">
  <img alt="GitHub fork" src="https://img.shields.io/github/forks/EliaFantini/Image-Reconstructor-FISTA-proximal-method-on-wavelets-trasnform?label=Fork">
  <img alt="GitHub watchers" src="https://img.shields.io/github/watchers/EliaFantini/Image-Reconstructor-FISTA-proximal-method-on-wavelets-trasnform?label=Watch">
  <img alt="GitHub star" src="https://img.shields.io/github/stars/EliaFantini/Image-Reconstructor-FISTA-proximal-method-on-wavelets-trasnform?style=social">
</p>


Implementation of an Image Reconstructor that applies fast proximal gradient method (FISTA) to the wavelet transform of an image using L1 and Total Variation (TV) regularizations. 

For a more detailed explanation of the terms mentioned above, please read *Exercise instructions.pdf*. 

The project was part of an assignment for the EPFL course [EE-556 Mathematics of data: from theory to computation](https://edu.epfl.ch/coursebook/en/mathematics-of-data-from-theory-to-computation-EE-556). The backbone of the code structure to run the experiments was already given by the professor and his assistants, what I had to do was to implement the core of the optimization steps, which are the ISTA and FISTA algorithms and other minor components. Hence, every code file is a combination of my personal code and the code that was given us by the professor.

The following image shows an example of the output of the code. The original image gets ruined by removing pixels randomly, then the model trained with the best lambda hyperparameter and L1 or TV regularizations tries to reconstruct the image. The plot under it shows the PSNR values (it's a denoising metric in dB, the higher the better) related to different lambda values. 

<p align="center">
<img width="auto" alt="Immagine 2022-08-05 155002" src="https://user-images.githubusercontent.com/62103572/183091131-c5849962-f382-4978-bae7-2c15d80c5d9d.png">
</p>

## Author
-  [Elia Fantini](https://github.com/EliaFantini)

## How to install and reproduce results
Download this repository as a zip file and extract it into a folder
The easiest way to run the code is to install Anaconda 3 distribution (available for Windows, macOS and Linux). To do so, follow the guidelines from the official
website (select python of version 3): https://www.anaconda.com/download/

Additional package required are: 
- matplotlib
- skimage
- pytorch
- pywt
- PIL

To install them write the following command on Anaconda Prompt (anaconda3):
```shell
cd *THE_FOLDER_PATH_WHERE_YOU_DOWNLOADED_AND_EXTRACTED_THIS_REPOSITORY*
```
Then write for each of the mentioned packages:
```shell
conda install *PACKAGE_NAME*
```
Some packages might require more complex installation procedures (especially pytorch). If the above command doesn't work for a package, just google "How to install *PACKAGE_NAME* on *YOUR_MACHINE'S_OS*" and follow those guides.

Finally, run **inpainting_template.py** to tune the lambda hyperparameter with both ISTA and FISTA and denoise an image (by default is *lauterbrunnen.jpg*, to change it you have to change the image path in the code) with the best model obtained:
```shell
python inpainting_template.py
```

## Files description

- **code/common/** : folder containing utils and other modular code to be used in the training
- **code/data/**: folder containing the images to denoise

- **code/examples**: folder containing code to test and visualize the wavelet transform and the TV norm

- **inpainting_template.py**: main code to run the training and testing

- **Answers.pdf**: pdf with the answers and plots to the assignment of the course

- **Exercise instructions.py**: pdf with the questions of the assignment of the course

## 🛠 Skills
Python, PyTorch, Matplotlib. Machine learning, proximal methods, hyperparameter tuning, denoising, wavelets transform, total variation and L1 regularization.

## 🔗 Links
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://github.com/EliaFantini/)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/-elia-fantini/)
