# kaggle_Google_Smartphone_Decimeter_Challenge

Thanks to kaggle and organizers hosting a nice competition.

## Overview
* Predicting the Noise, `Noise = Ground Truth - Baseline`, like denoising in computer vision
* Using the speed `latDeg(t + dt) - latDeg(t)/dt`  as input  instead of the absolute position for preventing overfitting on the train dataset.
* Making 2D image input with Short Time Fourie Transform, STFT, and then using ImageNet convolutional neural network

![image-20210806172801198](./images/pipeline.png)

## STFT and Conv Network Part
* Input: Using [librosa](https://librosa.org/doc/latest/index.html),  generating STFT for both latDeg&lngDeg speeds.
    + Each phone sequence are split into 256 seconds sequence then STFT with `n_tft=256`, `hop_length=1` and `win_length=16` , result in (256, 127, 2) feature for each degree. The following 2D images are generated  from 1D sequence.

![image-20210806174449510](./images/stft_images.png)

* Model: Regression and Segmentation
    * Regression: EfficientNet B3, predict latDeg&lngDeg noise, 
    * Segmentation: Unet ++ with EfficientNet encoder([segmentation pyroch](https://github.com/qubvel/segmentation_models.pytorch)) , predict stft  noise
        * segmentation prediction + input STFT ->  inverse STFT -> prediction of latDeg&lngDeg speeds

        * this speed prediction was used for:
            1. Low speed mask;  The points of low speed area are replaced with its median.
            2. Speed disagreement mask: If the speed from position prediction and this speed prediction differ a lot, remove such points and interpolate.
            
## LigthGBM Part
  * Input: IMU data excluding magnetic filed feature
      * also excluding y acceleration and z gyro because of phone mounting condition
      * adding moving average as additional features, `window_size=5, 15, 45`
  * Predict latDeg&lngDeg noise

## KNN at downtown Part
similar to [Snap to Grid](https://www.kaggle.com/robikscube/indoor-navigation-snap-to-grid-post-processing), but using both global and local feature. Local re-ranking comes from the  host baseline of [GLR2021](https://www.kaggle.com/c/landmark-retrieval-2020)
* Use train ground truth as database
* Global search: query(latDeg&lngDeg) -> find 10 candidates
* Local re-ranking: query(latDeg&lngDeg speeds and its moving averages) -> find 3 candidates -> taking mean over candidates