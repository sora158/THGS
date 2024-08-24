from pathlib import Path
import os
import hydra
import torch
import cv2
import glob
from torch.utils.data import DataLoader
from scipy.spatial.transform import Rotation as R
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import torch
import os
import pickle
import json
from pathlib import Path
import os
import numpy as np
import hydra
import torch
import glob
from torch.utils.data import DataLoader
import glob
import cv2

from scipy.interpolate import interp1d
from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera

from scipy.ndimage import zoom
device = torch.device("cuda:0")

label_colors = [
    [0, 0, 0],       # 背景
    [255, 0, 0],     # 脸部
    [0, 255, 0],     # 左眉毛
    [0, 0, 255],     # 右眉毛
    [255, 255, 0],   # 左眼
    [255, 0, 255],   # 右眼
    [0, 255, 255],   # 鼻子
    [128, 128, 128], # 上唇
    [128, 0, 0],     # 下唇
    [0, 128, 0],     # 嘴
    [0, 0, 128],     # 头发
    [128, 128, 0],   # 左耳
    [128, 0, 128],   # 右耳
    [0, 128, 128],   # 颈部
    [192, 192, 192], # 衣服
    [64, 64, 64],    # 眼镜
    [255, 128, 0],   # 帽子
    [0, 255, 128]    # 耳环
]

def interpolate_array(data, new_length):
    old_length = data.shape[0]
    new_data = np.zeros((new_length, data.shape[1], data.shape[2]))

    # 创建旧的和新的时间戳
    old_timestamps = np.linspace(0, 1, old_length)
    new_timestamps = np.linspace(0, 1, new_length)

    # 对每一个特征进行插值
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            interpolator = interp1d(old_timestamps, data[:, i, j], kind='linear')
            new_data[:, i, j] = interpolator(new_timestamps)

    return new_data


def read_image(file_path):
    # Read the image using a suitable library (e.g., PIL)
    image = cv2.imread(file_path)
    return image

def numeric_order(filename):
    # Extract the numeric part of the filename
    numeric_part = ''.join(filter(str.isdigit, filename))
    # Convert the numeric part to an integer
    return int(numeric_part)



def load_smplx_param(data):
    # Load data from the .npz file into the smpl_params dictionary

    smpl_params = data
    betas = smpl_params.get("betas", None)
    transl = smpl_params.get("transl", None)  # Assuming transl is directly stored in smpl_params
    global_orient = smpl_params.get("global_orient") 
    body_pose = smpl_params.get("body_pose_axis") 
    expression = smpl_params.get("expression", None)
    left_hand_pose = smpl_params.get("left_hand_pose", None)
    right_hand_pose = smpl_params.get("right_hand_pose", None)
    jaw_pose = smpl_params.get("jaw_pose", None)
    leye_pose = smpl_params.get("leye_pose", None)
    reye_pose = smpl_params.get("reye_pose", None)
    # Return organized data
    return {
        "betas": betas.astype(np.float32) if betas is not None else None,
        "transl": transl.astype(np.float32) if transl is not None else None,
        "global_orient": global_orient.astype(np.float32) if global_orient is not None else None,
        "body_pose": body_pose.astype(np.float32) if body_pose is not None else None,
        "expression": expression.astype(np.float32) if expression is not None else None,
        "left_hand_pose": left_hand_pose.astype(np.float32) if left_hand_pose is not None else None,
        "right_hand_pose": right_hand_pose.astype(np.float32) if right_hand_pose is not None else None,
        "jaw_pose": jaw_pose.astype(np.float32) if jaw_pose is not None else None,
        "leye_pose": leye_pose.astype(np.float32) if leye_pose is not None else None,
        "reye_pose": reye_pose.astype(np.float32) if reye_pose is not None else None,
    }

import torch.multiprocessing as mp

def permute_transform_func(x):
    return np.transpose(x, (1, 2, 0))

import numpy as np
from scipy.spatial.transform import Rotation as R

import numpy as np
from scipy.spatial.transform import Rotation as R

def fill_missing_images(ori_imgs_path, parsing_left_imgs_path):
    # 获取两个文件夹中的文件列表
    ori_img_files = glob.glob(os.path.join(ori_imgs_path, '*'))
    parsing_left_img_files = glob.glob(os.path.join(parsing_left_imgs_path, '*'))

    # 提取文件名中的数字编号
    ori_img_numbers = {int(os.path.basename(file).split('.')[0]) for file in ori_img_files}
    parsing_left_img_numbers = {int(os.path.basename(file).split('.')[0]) for file in parsing_left_img_files}

    # 查找在 ori_imgs_path 中存在但在 parsing_left_imgs_path 中不存在的文件编号
    missing_img_numbers = ori_img_numbers - parsing_left_img_numbers

    # 创建一张全黑的图像
    height, width = 720, 1280  # 替换为实际图像的高度和宽度
    black_image = np.zeros((height, width, 3), dtype=np.uint8)  # 如果图像是彩色的

    # 根据缺失的文件编号创建全黑的文件
    for img_number in missing_img_numbers:
        img_filename = f'{img_number:06d}.jpg'  # 根据命名逻辑生成文件名
        img_path = os.path.join(parsing_left_imgs_path, img_filename)

        # 保存全黑图像到文件
        cv2.imwrite(img_path, black_image)
        print(f"已创建缺失的文件: {img_filename}")

def setup_camera(w, h, k, w2c, near=0.01, far=100):
    fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
    w2c = torch.tensor(w2c).cuda().float()
    cam_center = torch.inverse(w2c)[:3, 3]
    w2c = w2c.unsqueeze(0).transpose(1, 2)
    opengl_proj = torch.tensor([[2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
                                [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
                                [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
                                [0.0, 0.0, 1.0, 0.0]]).cuda().float().unsqueeze(0).transpose(1, 2)
    full_proj = w2c.bmm(opengl_proj)
    cam = Camera(
        image_height=h,
        image_width=w,
        tanfovx=w / (2 * fx),
        tanfovy=h / (2 * fy),
        bg=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=w2c,
        projmatrix=full_proj,
        sh_degree=0,
        campos=cam_center,
        prefiltered=False,
        debug = False,
    )
    return cam

def resize_image_to_720p(image, target_width, target_height):
    original_height, original_width = image.shape[:2]
    
    # Compute the zoom factors for height and width
    zoom_factor_height = target_height / original_height
    zoom_factor_width = target_width / original_width
    
    # Apply zoom to the image
    resized_image = zoom(image, (zoom_factor_height, zoom_factor_width, 1), order=1)  # order=1 for bilinear interpolation
    
    return resized_image

class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir,subject,split,opt):
        self.root_dir =root_dir
        self.subject = subject
        subdirectories = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        subdirectories = sorted(subdirectories)
        permute_transform = transforms.Lambda(lambda x: np.transpose(x, (1, 2, 0)))
        self.transform = transforms.Compose([
            permute_transform,
        ])
        self.parsing_transform = transforms.Compose([
            permute_transform,
        ])
        self.rays_o_cache = {} 
        self.rays_d_cache = {} 
        self.downscale = opt.downscale
        self.split = split
        self.near = None
        self.far =  None
        self.rays_dict ={}
        self.data={}
        self.ori_imgs_path={}
        self.parsing_imgs_path= {}
        self.parsing_left_imgs_path= {}
        self.parsing_right_imgs_path = {}
        self.parsing_head_imgs_path = {}
        self.face_mask_path = {}
        self.keypoints_2d =[]
        self.audio_data = []
        self.min_distances = {}
        self.max_distances = {}
        
        self.msk_path = {} #用于切片
        self.face_parsing_path = {}
        self.img_path = {}
        self.smplx_params  = {}
        self.rays ={}
        self.keypoints ={}   
        self.audio={}    
        self.keypoints_valid={}
        all_camera_translations = []
        num = 0
        for subdirectory in subdirectories:
            ori_imgs_path = os.path.join(self.root_dir, subdirectory, "image")
            face_mask_path = os.path.join(self.root_dir, subdirectory, "face_mask")
            print(subdirectory)
            num_imgs = len(os.listdir(ori_imgs_path))
            ori_imgs_path  = glob.glob(os.path.join(ori_imgs_path, '*'))
            ori_imgs_path = sorted(ori_imgs_path, key=lambda x: int(x.split('/')[-1].split('.')[0]))
            self.ori_imgs_path.update({i: (ori_imgs_path[i-num]) for i in range(num, num + num_imgs)}) 
            parsing_imgs_path = os.path.join(self.root_dir, subdirectory, "parsing_full_body")
            parsing_imgs_path  = glob.glob(os.path.join(parsing_imgs_path, '*'))
            parsing_imgs_path = sorted(parsing_imgs_path, key=lambda x: int(x.split('/')[-1].split('.')[0]))
            self.parsing_imgs_path.update({i: (parsing_imgs_path[i-num]) for i in range(num, num + num_imgs)})

            face_mask_path  = glob.glob(os.path.join(face_mask_path, '*'))
            face_mask_path = sorted(face_mask_path, key=lambda x: int(x.split('/')[-1].split('.')[0]))
            self.face_mask_path.update({i: (face_mask_path[i-num]) for i in range(num, num + num_imgs)})
            
            self.keypoints_2d.append(np.load(f"{self.root_dir}/{subdirectory}/keypoints.npy"))
            self.audio_data.append(interpolate_array(np.load(f"{self.root_dir}/{subdirectory}/aud_eo.npy"), num_imgs))
            num += num_imgs
        
        def get_smplx_to_o3d_R() -> np.ndarray:
            R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) #origin
            return R
        camera = np.load(f"{self.root_dir}/optimized_camera_params.npz")
        focal_length  = camera["focal_length"]
        camera_transl = camera["avg_transl"]
        
        self.height,self.width,_ = read_image(self.ori_imgs_path[0]).shape    
        print("camera_transl:",camera_transl,"focal_length",focal_length, self.height, self.width)
        self.image_shape = (self.height, self.width)
        K = np.array([[focal_length, 0, self.width/2], [0, focal_length, self.height/2]]) 
        c2w = np.eye(4)
        c2w[:3,:3] = get_smplx_to_o3d_R() 
        c2w[:3,3] =  camera_transl 
        self.cam = setup_camera(self.width,self.height,K, np.linalg.inv(c2w), near=1.0, far=100)
        start = opt.start
        end = opt.end + 1
        self.w_bg = opt.w_bg
        skip = opt.get("skip", 1)
        
        data = np.load(f"{self.root_dir}/poses_optimized.npz")
        self.smpl_params = {key: data[key] for key in data}
        self.smpl_params["global_orient"] = self.smpl_params["global_orient"][:,0]
        for k, v in self.smpl_params.items():
            if k != "betas":
                self.smpl_params[k] = v[start:end:skip]
                
        self.keypoints_2d = np.vstack(self.keypoints_2d)
        self.audio_data = np.vstack(self.audio_data)

        for i in range(start, end, skip):
            self.msk_path.update({(i-start)//skip: self.parsing_imgs_path[i]})
            self.face_parsing_path.update({(i-start)//skip: self.face_mask_path[i]})
            self.img_path.update({(i-start)//skip: self.ori_imgs_path[i]})
            self.keypoints.update({(i-start)//skip: self.keypoints_2d[i]})
            self.audio.update({(i-start)//skip: self.audio_data[i]})
        print(start,end,skip)
        
    def get_SMPL_params(self):
        return {
            k: torch.from_numpy(v.copy()) for k, v in self.smpl_params.items()
        }
        
    def process_face_msk(self,face_msk):
        face_parsing = np.zeros((face_msk.shape[1], face_msk.shape[2]), dtype=np.bool)
        for i in range(face_msk.shape[0]):
            for label_idx in range(11,14):
                face_parsing += (face_msk[i] == label_idx).astype(np.bool) 
        return face_parsing
        return len(self.img_path)
    
    def __getitem__(self, idx):
        img = cv2.imread(self.img_path[idx])
        msk = cv2.imread(self.msk_path[idx])
        #bg =  (bg[..., :3] / 255).astype(np.float32)
        #cc modify 0517 for 720p
        target_width, target_height = 1280, 720

        # Resize the image and mask to the target resolution
        if self.downscale > 1:
            img = cv2.resize(img, dsize=None, fx=1/self.downscale, fy=1/self.downscale)
            msk = cv2.resize(msk, dsize=None, fx=1/self.downscale, fy=1/self.downscale)

        img = (img[..., :3] / 255).astype(np.float32)
        msk = (msk[..., 0] / 255).astype(np.bool)
       
        msk = msk.astype(np.float32)
        
        img = img * msk[..., None] + (1 - msk[..., None]) # background here
        
        face_msk = np.load(self.face_parsing_path[idx])['arr_0']
        face_msk = self.process_face_msk(face_msk)
        keypoints = self.keypoints[idx]
        
        full_pose = np.concatenate((self.smpl_params["global_orient"][idx],
                                    self.smpl_params["body_pose"][idx],
                                    self.smpl_params["jaw_pose"][idx],
                                    self.smpl_params["leye_pose"][idx],
                                    self.smpl_params["reye_pose"][idx],
                                    self.smpl_params["left_hand_pose"][idx],
                                    self.smpl_params["right_hand_pose"][idx],
                                    ))[None,...]
        exp = self.smpl_params["expression"][idx]
        ret = {
                # GS3D
                "rgb": img, #(291600, 3)
                "mask":msk,#(291600,)
                "cam":self.cam,
                # SMPLX parameters
                "smpl_beta": self.smpl_params["betas"][0],  
                "smpl_trans": self.smpl_params["transl"][idx],
                "smpl_pose": full_pose,
                "smpl_exp": exp,
                "idx": idx,
                "keypoints":keypoints,
                "audio":self.audio[idx],
                "face_parsing":face_msk,
            }
        meta_info = {
            "video": self.subject,
        }
        viz_id = f"video{self.subject}_dataidx{idx}"
        meta_info["viz_id"] = viz_id
        return ret, meta_info
