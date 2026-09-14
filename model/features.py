"""The inference-time image pipeline, without nuplan.

navsim's SparseDriveFeatureBuilder imports nuplan for its target builder and
annotations. The *image* path needs none of that -- PIL, numpy and some
calibration algebra -- so it is reimplemented here and the TT process reads the
feature cache directly instead of handing tensors across the two-interpreter
seam.

Only test mode. The augmentations collapse:

    resize  = max(256/1080, 512/1920) = 0.2667   fixed
    crop    = (0, 32, 512, 288)                  centred, fixed
    flip    = False,  rotate = 0
    ego_rotation      returns unchanged at angle 0
    photo_metric      returns unchanged in test mode

so the transform is a scale and a translation, and it multiplies into lidar2img
the same way for every frame.

Graded against the navsim builder itself in tools/poc_features.py -- this is a
reimplementation, so it is only worth having if it is bit-comparable.
"""

import copy

import numpy as np
import torch
from PIL import Image

CAMS = ("cam_l0", "cam_f0", "cam_r0")
H_SRC, W_SRC = 1080, 1920
FINAL_H, FINAL_W = 256, 512
IMG_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
IMG_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


def _aug():
    resize = max(FINAL_H / H_SRC, FINAL_W / W_SRC)
    new_w, new_h = int(W_SRC * resize), int(H_SRC * resize)
    crop_h = int(new_h) - FINAL_H          # bot_pct_lim is (0, 0)
    crop_w = int(max(0, new_w - FINAL_W) / 2)
    return resize, (new_w, new_h), (crop_w, crop_h, crop_w + FINAL_W, crop_h + FINAL_H)


def camera_params(frame):
    """calibration -> lidar2img per camera, in CAMS order."""
    l2i = []
    for cam in CAMS:
        info = frame[cam]
        c2l = np.eye(4)
        c2l[:3, :3] = info["sensor2lidar_rotation"]
        c2l[:3, 3] = info["sensor2lidar_translation"]
        l2c = np.eye(4)
        r = np.linalg.inv(info["sensor2lidar_rotation"])
        l2c[:3, :3] = r.T
        l2c[3, :3] = -(info["sensor2lidar_translation"] @ r.T)
        k = copy.deepcopy(info["intrinsics"])
        viewpad = np.eye(4)
        viewpad[:k.shape[0], :k.shape[1]] = k
        l2i.append(viewpad @ l2c.T)
    return l2i


def build(feature: dict):
    """cache entry -> (imgs [C,3,256,512], status [8], proj [C,4,4], iwh [C,2])."""
    frame = feature["camera_feature"][-1]        # only the current frame is used
    l2i = camera_params(frame)
    resize, resize_dims, crop = _aug()

    mat = np.eye(3)
    mat[:2, :2] *= resize
    mat[:2, 2] -= np.array(crop[:2])             # rotate 0 and flip False drop out
    ext = np.eye(4)
    ext[:3, :3] = mat

    imgs = []
    for i, cam in enumerate(CAMS):
        im = Image.open(str(frame[cam]["image_path"]))
        im = im.resize(resize_dims).crop(crop)
        a = np.array(im).astype(np.float32)
        a = (a - IMG_MEAN.reshape(1, -1)) / IMG_STD.reshape(1, -1)
        imgs.append(a.transpose(2, 0, 1))
        l2i[i] = ext @ l2i[i]

    return (torch.from_numpy(np.ascontiguousarray(np.stack(imgs))).float(),
            feature["status_feature"].float(),
            torch.from_numpy(np.stack(l2i)).float(),
            torch.tensor([[FINAL_W, FINAL_H]] * len(CAMS), dtype=torch.float32))
