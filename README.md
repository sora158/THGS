# Data Access Instructions

This document provides detailed steps on how to access and use data hosted on Google Drive and Baidu Netdisk.

## Google Drive

### Accessing Data in Google Drive
1. Google Drive link (will come soon).

## Baidu Netdisk
1. Visit the link: [Baidu Netdisk](https://pan.baidu.com/s/1nwiieKtYzNtgfMkuQERpaw?pwd=THGS)
2. Enter the password: THGS
   
### Data Structure

The directory structure for the project is outlined below. Each subfolder contains data related to a specific 10-second video segment, along with various associated files and subdirectories that store different types of data processed and generated during the analysis.

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
### Using the Data
Once downloaded, the data can be decompressed in Liunx using command like: tar -xzvf Trading.tar.gz

## Important Notes
- Ensure compliance with all relevant usage and distribution policies.
- If there are any problems, please contact [chenchuang010@gmail.com] or directly raise a issue.
