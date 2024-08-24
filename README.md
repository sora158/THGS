# TalkingAvatar Dataset for Paper THGS

This document provides detailed steps on how to access and use data hosted on Google Drive and Baidu Netdisk.

## Google Drive
1. Visit the link: [Google Drive](https://drive.google.com/drive/folders/15Ly9UUoeltswIsJU9Fp7tAuaIlcS98gI?usp=sharing)

## Baidu Netdisk
1. Visit the link: [Baidu Netdisk](https://pan.baidu.com/s/1nwiieKtYzNtgfMkuQERpaw?pwd=THGS)
2. Enter the password: THGS
   
# Fah Dataset Structure Overview

The Fah dataset is organized into multiple directories, each corresponding to a 10-second video segment. Below is a detailed breakdown of the dataset's structure and contents.

## Data Structure Example
```
Fah/
├── Fah_output000/ (data folder corresponding to a 10s video)
│   ├── Fah_output000.mp4
│   ├── aud.wav
│   ├── aud_eo.npy (audio feature)
│   ├── deca/
│   ├── deca_vis/
│   ├── face_mask/
│   ├── fan/
│   ├── image (ground truth)
│   ├── keypoints.npy (2d keypoints)
│   ├── ours_exp (estimated SMPL-X params)
│   ├── parsing (foreground mask)
│   ├── pixie/
│   └── pixie_vis/
├── Fah_output001/ (similar data folder structure)
├── Fah_output002/ (similar data folder structure)
├── Fah_output003/ (similar data folder structure)
...
├── optimized_camera_params.npz (camera parameters)
└── poses_optimized.npz (refined body parameters through Joint Optimization Strategy, Stage 1)
```

## Using the Data

Once the data is downloaded, it can be decompressed in Linux using the following command:
```bash
tar -xzvf Fah.tar.gz
```

## Important Notes
Ensure compliance with all relevant usage and distribution policies.
If there are any problems, please contact [chenchuang010@gmail.com] or directly raise an issue.
