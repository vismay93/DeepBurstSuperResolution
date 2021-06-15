# Implementation of Deep Burst Super Resolution Model

I have used the preprocessing scripts and dataset scripts provided in https://github.com/goutamgmb/NTIRE21_BURSTSR and added my implementation of Deep Burst Super Resolution model and corresponding training and evaluation scripts.


### ** This repos has been forked from: https://github.com/goutamgmb/NTIRE21_BURSTSR  **

## Table of contents 
* [Introduction](#introduction)
* [Reference](#reference)
* [Dates](#dates)
* [Description](#description)
* [Track 1 - Synthetic](#track-1---synthetic)
* [Track 2 - Real-world](#track-2---real-world)
* [Toolkit](#toolkit)
* [Issues and Questions](#issues-and-questions)
* [Organizers](#organizers)
* [Terms and conditions](#terms-and-conditions)
* [Acknowledgements](#acknowledgements)

## Introduction
Burst Image Super-Resolution Challenge will be held as part of the [6th edition of 
NTIRE: New Trends in Image Restoration and Enhancement](https://data.vision.ee.ethz.ch/cvl/ntire21/) workshop to be held in conjunction 
with [CVPR 2021](http://cvpr2021.thecvf.com/). The task of this challenge is to generate 
a denoised, demosaicked, higher-resolution image, given a RAW burst as input. 
The challenge uses a new dataset and has 2 tracks, namely **Track 1: Synthetic** and 
**Track 2: Real-world**. The top ranked participants in each track will be awarded and all 
participants are invited to submit a paper describing their solution to the associated 
NTIRE workshop at CVPR 2021

## Reference

The provided **BurstSR dataset**, **code**, and evaluation protocol for this challenge was introduced in our following paper. Please cite it if you use any of these in your work.
Our paper also introduces a new method for Deep Burst Super-Resolution, which you can use as a baseline solution.

*Deep Burst Super-Resolution*  
Goutam Bhat, Martin Danelljan, Luc Van Gool, Radu Timofte  
arXiv:2101.10997  
[[paper](https://arxiv.org/abs/2101.10997)]


## Dates
* 2021.01.26 Release of train and validation data  
* 2021.02.01 Validation server online  
* 2021.03.15 Final test data release (inputs only)  
* 2021.03.20 Test output results submission deadline  
* 2021.03.20 Fact sheets and code/executable submission deadline  
* 2021.03.22 Preliminary test results released to the participants  
* 2021.04.02 Paper submission deadline for entries from the challenge  
* 2021.06.15 NTIRE workshop and challenges, results and award ceremony (CVPR 2021, Online)  


## Description
Given multiple noisy RAW images of a scene, the task in burst super-resolution is to 
predict a denoised higher-resolution RGB image by combining information from the 
multiple input frames. Concretely, the method will be provided a burst sequence 
containing 14 images, where each image contains the RAW sensor data from a bayer filter 
(RGGB) mosaic. The images in the burst have unknown offsets with respect to each other, 
and are corrupted by noise. The goal is to exploit the information from the multiple 
input images to predict a denoised, demosaicked RGB image having a 4 times higher 
resolution, compared to the input. The challenge will have two tracks, 
namely 1) Synthetic and 2) Real-world based on the source of the input data. The goal 
in both the tracks is to reconstruct the **original** image as well as possible, and 
not to artificially generate a plausible, visually pleasing image.





## Track 1 - Synthetic
In the synthetic track, the input bursts are generated from RGB images using a synthetic 
data generation pipeline. 

**Data generation:** The input sRGB image is first converted to linear sensor space 
using an inverse camera pipeline. A LR burst is then generated by applying random 
translations and rotations, followed by bilinear downsampling. The generated burst is 
then mosaicked and corrupted by random noise. 

**Training set:** We provide [code](datasets/synthetic_burst_train_set.py) to generate the synthetic 
bursts using any image dataset for training. Note that any image dataset except the 
test split of the [Zurich RAW to RGB dataset](http://people.ee.ethz.ch/~ihnatova/pynet.html#dataset) 
can be used to generate synthetic bursts for training.  

**Validation set:** The bursts in the validation set have been 
pre-generated with the [data generation code](datasets/synthetic_burst_train_set.py), 
using the RGB images from the test split of the 
[Zurich RAW to RGB dataset](http://people.ee.ethz.ch/~ihnatova/pynet.html#dataset). 

### Registration
If you wish to participate in the Synthetic track, please register for the challenge at the 
[codalab page](https://competitions.codalab.org/competitions/28078#participate) to get access to the evaluation server and receive email notifications for 
the challenge.

### Evaluation
The methods will be ranked using the fidelity (in terms of PSNR) with the high-resolution 
ground truth, i.e. the linear sensor space image used to generate the burst. The focus of 
the challenge is on learning to reconstruct the original high-resolution image, and not 
the subsequent post-processing. Hence, the PSNR computation will be computed in the 
linear sensor space, before post-processing steps such as color correction, 
white-balancing, gamma correction etc.


### Validation

The results on the **validation set** can be uploaded on the [Codalab server](https://competitions.codalab.org/competitions/28078#participate) (**live now**)
to obtain the performance measures, as well as a live leaderboard ranking. The results should be uploaded as a ZIP file
containing the network predictions for each burst. The predictions must be normalized to the range [0, 2^14] and saved
as 16 bit (uint16) png files. Please refer to [save_results_synburst_val.py](scripts/save_results_synburst_val.py) for
an example on how to save the results. An example submission file is available [here](https://data.vision.ee.ethz.ch/bhatg/syn_burst_example_submission.zip).

### Final Submission

The **test set** is now public. You can download the test set containing 500 synthetic bursts from [this link](https://data.vision.ee.ethz.ch/bhatg/track1_test_set.zip). You can use the dataset class provided in [synthetic_burst_test_set.py](datasets/synthetic_burst_test_set.py) in the latest commit to load the burst sequences.

For the final submission, you need to submit:
* The predicted outputs for each burst sequence as a zip folder, in the same format as used for uploading results to the codalab validation server (see [this](https://github.com/goutamgmb/NTIRE21_BURSTSR#validation) for details).
* The code and model files necessary to reproduce your results.
* A factsheet (both PDF and tex files) describing your method. The template for the factsheet is available [here](https://data.vision.ee.ethz.ch/bhatg/NTIRE_BURSTSR_TEMPLATE.zip).  

The results, code, and factsheet should be submitted via the [google form](https://docs.google.com/forms/d/e/1FAIpQLSeil_Q3jrkcmy5ZmA3xEOW3-KzmA1Q0TFFU218JvRDopy_Jdg/viewform?usp=sf_link)

**NOTE:** Training on the validation split is **NOT** allowed for test set submissions.

## Track 2 - Real-world
This track deals with the problem of real-world burst super-resolution. For this purpose, 
we introduce a new dataset BurstSR consisting of real-world bursts.

**BurstSR dataset:** BurstSR consists of 200 RAW burst sequences, and 
corresponding high-resolution ground truths (a reference for the paper introducing the 
dataset will be made available soon). Each burst sequence contains 14 RAW images 
captured using a handheld smartphone camera. For each burst sequence, we also capture 
a high-resolution image using a DSLR camera mounted on a tripod to serve as ground truth. 
We extract 160x160 crops from the bursts to obtain a training set consisting of
5405 crops, and a validation set consisting of 882 crops. A detailed description of the 
BurstSR dataset is available in the paper ["Deep Burst Super-Resolution"](https://arxiv.org/pdf/2101.10997.pdf). 
Please cite the [paper](https://arxiv.org/pdf/2101.10997.pdf) if you use the BurstSR dataset in your work. 

**Challenges:** Since the burst and ground 
truth images are captured using different cameras, there exists a spatial mis-alignment, 
as well as color mis-match between the images. Thus, in addition to designing network 
architectures, developing effective training strategies to utilize mis-aligned training 
data is a key challenge in this track.

### Registration
If you wish to participate in the Real-world track, please register for the challenge at the 
[codalab page](https://competitions.codalab.org/competitions/28079#participate) to receive email notifications for 
the challenge.

### Evaluation
Due to the spatial and color mis-alignments between the input burst and the ground truth, 
it is difficult to estimate similarity between the network prediction and the ground 
truth. We introduce **AlignedPSNR** metric for this purpose. A user study will also be conducted as a complement to 
determine the final ranking on the test set.

**AlignedPSNR:** AlignedPSNR first spatially 
aligns the network prediction to the ground truth, using pixel-wise optical flow 
estimated using [PWC-Net](https://arxiv.org/abs/1709.02371). A linear color mapping between the input burst and the ground 
truth, modeled as a 3x3 color correction matrix, is then estimated and used to transform 
the spatially aligned network prediction to the same color space as the ground truth. 
Finally, PSNR is computed between the spatially aligned and color corrected network 
prediction and the ground truth. More description of the AlignedPSNR metric is available in 
the paper ["Deep Burst Super-Resolution"](https://arxiv.org/pdf/2101.10997.pdf). 



**User study:** The emphasis of the user study 
will be on which method can best reconstruct the **original** high-frequency details. The 
goal is thus not to generate more pleasing images by modifying the output color space 
or generating artificial high frequency content not existing in the high-resolution 
ground truth.


### Validation

The will be no evaluation server for Track 2. Instead, the ground 
truth images for the **validation set** are provided and the methods can be evaluated locally 
using the provided implementation of [AlignedPSNR](utils/metrics.py). Please refer to 
[evaluate_burstsr_val.py](scripts/evaluate_burstsr_val.py) script for an example on 
how to evaluate on BurstSR validation set. 

NOTE: The [evaluate_burstsr_val.py](scripts/evaluate_burstsr_val.py) script computes the AlignedPSNR score in the linear sensor space, before white-balancing or any intensity scaling. Since the RAW sensor values in the captured bursts are generally small (mean of around 0.1 - 0.2), the computed PSNR values are generally very high (> 47), since the maximum signal value in the PSNR computation is still assumed to be 1.0. In the ["Deep Burst Super-Resolution"](https://arxiv.org/pdf/2101.10997.pdf) paper, we computed the final scores on the white-balanced image, after scaling the image intensities to be between [0, 1]. Thus the scores computed by [evaluate_burstsr_val.py](scripts/evaluate_burstsr_val.py) cannot be compared with the scores reported in the paper.


### Final Submission

The **test set** is now public. You can download the test set containing 639 real-world bursts from [this link](https://data.vision.ee.ethz.ch/bhatg/track2_test_set.zip). You can use the dataset class provided in [burstsr_test_dataset.py](datasets/burstsr_test_dataset.py) in the latest commit to load the burst sequences.

For the final submission, you need to submit:
* The predicted outputs for each burst sequence as a zip folder. The predictions must be normalized to the range [0, 2^14] and saved as 16 bit (uint16) png files. Please refer to [save_results_burstsr_test.py](scripts/save_results_burstsr_test.py) for an example on how to save the results.
* The code and model files necessary to reproduce your results.
* A factsheet (both PDF and tex files) describing your method. The template for the factsheet is available [here](https://data.vision.ee.ethz.ch/bhatg/NTIRE_BURSTSR_TEMPLATE.zip).  

The results, code, and factsheet should be submitted via the [google form](https://docs.google.com/forms/d/e/1FAIpQLSeil_Q3jrkcmy5ZmA3xEOW3-KzmA1Q0TFFU218JvRDopy_Jdg/viewform?usp=sf_link)

**NOTE:** Training on the validation split is **NOT** allowed for test set submissions.


## Toolkit
We also provide a Python toolkit which includes the necessary data loading and 
evaluation scripts. The toolkit contains the following modules.

* [data_processing](data_processing): Contains the forward and inverse camera pipeline 
  employed in [“Unprocessing images for learned raw denoising”](https://arxiv.org/abs/1811.11127), 
  as well as the [script](data_processing/synthetic_burst_generation.py) to generate a 
  synthetic burst from a single RGB image.
* [datasets](datasets): Contains the PyTorch dataset classes useful for the challenge. 
    * [synthetic_burst_train_set](datasets/synthetic_burst_train_set.py) provides the SyntheticBurst dataset which generates synthetic bursts using any image dataset. 
    * [zurich_raw2rgb_dataset](datasets/zurich_raw2rgb_dataset.py) can be used to load 
      the RGB images Zurich RAW to RGB mapping dataset. This can be used along with SyntheticBurst dataset to generate synthetic bursts for training.  	
    * [synthetic_burst_val_set](datasets/synthetic_burst_val_set.py) can be used to load 
      the pre-generated synthetic validation set.
    * [synthetic_burst_test_set](datasets/synthetic_burst_test_set.py) can be used to load 
      the pre-generated synthetic test set.
    * [burstsr_dataset](datasets/burstsr_dataset.py) provides the BurstSRDataset class which can be used to load the RAW bursts and high-resolution ground truths for the real-world track.
    * [burstsr_test_dataset](datasets/burstsr_test_dataset.py) provides the BurstSRTestDataset class which can be used to load the test split in the real-world track 2.
* [pwcnet](pwcnet): The code for the optical flow network [PWC-Net](https://arxiv.org/abs/1811.11127) 
  borrowed from [pytorch-pwc](https://github.com/sniklaus/pytorch-pwc). The network weights can be 
  downloaded from [here](https://data.vision.ee.ethz.ch/bhatg/pwcnet-network-default.pth).
* [scripts](scripts): Includes useful example scripts.
    * [download_burstsr_dataset](scripts/download_burstsr_dataset.py) can be used to 
      download and unpack the BurstSR dataset.
    * [test_synthetic_burst](scripts/test_synthetic_bursts.py) provides an example on how
  to use the [SyntheticBurst](datasets/synthetic_burst_train_set.py) dataset.
    * [test_burstsr_dataset](scripts/test_burstsr_dataset.py) provides an example on how
  to use the [BurstSR](datasets/burstsr_dataset.py) dataset.
    * [save_results_synburst_val](scripts/save_results_synburst_val.py) provides an example
      on how to save the results on [SyntheticBurstVal](datasets/synthetic_burst_val_set.py) 
      dataset for submission on the evaluation server.
    * [save_results_burstsr_test](scripts/save_results_burstsr_test.py) provides an example
      on how to save the results on the test set for the final submission for Track 2.
    * [evaluate_burstsr_val](scripts/evaluate_burstsr_val.py) provides an example on how
      to evaluate a method on the [BurstSR](datasets/burstsr_dataset.py) validation set.
* [utils](utils): Contains the [AlignedPSNR](utils/metrics.py) metric, as well as other utility functions.

**Installation:** The toolkit requires [PyTorch](https://pytorch.org/) and [OpenCV](https://opencv.org/) 
for track 1, and additionally [exifread](https://pypi.org/project/ExifRead/) and 
[cupy](https://cupy.dev/) for track 2. The necessary packages can be installed with 
[anaconda](https://www.anaconda.com/), using the [install.sh](install.sh) script. 


## Data
We provide the following data as part of the challenge. 

**Synthetic validation set:** The official validation set for track 1. The dataset contains 300 synthetic bursts, each containing 
14 RAW images. The synthetic bursts are generated from the RGB images from the test split of the Zurich RAW to RGB mapping dataset. 
The dataset can be downloaded from [here](https://data.vision.ee.ethz.ch/bhatg/syn_burst_val.zip).

**Synthetic test set:** The official test set for track 1. The dataset contains 500 synthetic bursts, each containing 
14 RAW images. The dataset can be downloaded from [here](https://data.vision.ee.ethz.ch/bhatg/track1_test_set.zip).

**BurstSR train and validation set:** The training and validation set for track 2. 
The dataset has been split into 10 parts and can be downloaded and unpacked using the 
[download_burstsr_dataset.py](scripts/download_burstsr_dataset.py) script. In case of issues with the script, the download links 
are available [here](burstsr_links.md).

**BurstSR test set:** The test set for track 2. The dataset contains 639 bursts, each containing 14 RAW images. The dataset can be downloaded from [here](https://data.vision.ee.ethz.ch/bhatg/track2_test_set.zip).

**Zurich RAW to RGB mapping set:** The RGB images from the training split of the 
[Zurich RAW to RGB mapping dataset](http://people.ee.ethz.ch/~ihnatova/pynet.html#dataset) 
can be downloaded from [here](https://data.vision.ee.ethz.ch/bhatg/zurich-raw-to-rgb.zip). These RGB images can be 
used to generate synthetic bursts for training using  the SyntheticBurst class.

Additionally, the weights for the [PWC-Net](https://arxiv.org/abs/1811.11127) network used 
in the evaluation for Track 2 can be downloaded from [here](https://data.vision.ee.ethz.ch/bhatg/pwcnet-network-default.pth).

## Issues and questions: 
In case of any questions about the challenge or the toolkit, feel free to open an issue on Github.

## Organizers
* [Goutam Bhat](https://goutamgmb.github.io/) (goutam.bhat@vision.ee.ethz.ch)
* [Martin Danelljan](https://martin-danelljan.github.io/) (martin.danelljan@vision.ee.ethz.ch)
* [Radu Timofte](http://people.ee.ethz.ch/~timofter/) (radu.timofte@vision.ee.ethz.ch)

## Terms and conditions
The terms and conditions for participating in the challenge are provided [here](terms_and_conditions.md)


## Acknowledgements
The toolkit uses the forward and inverse camera pipeline code from [unprocessing](https://github.com/timothybrooks/unprocessing),
as well as the PWC-Net code from [pytorch-pwc](https://github.com/sniklaus/pytorch-pwc).
