# *************************************************************************
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0 
# *************************************************************************
import os
import os.path as osp
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from einops import rearrange
from PIL import Image


def save_videos_from_pil(pil_images, path, fps=8, crf=None):
    """Save PIL images as video using OpenCV."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    width, height = pil_images[0].size
    
    # Use mp4v codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    
    for pil_image in pil_images:
        frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        writer.write(frame)
    
    writer.release()


def save_videos_grid(videos_, path: str, rescale=False, n_rows=6, fps=8, crf=None):
    """Save video tensor as grid."""
    if not isinstance(videos_, list):
        videos_ = [videos_]

    outputs = []
    vid_len = videos_[0].shape[2]
    
    for i in range(vid_len):
        output = []
        for videos in videos_:
            videos = rearrange(videos, "b c t h w -> t b c h w")
            x = torchvision.utils.make_grid(videos[i], nrow=n_rows)
            x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
            if rescale:
                x = (x + 1.0) / 2.0
            x = (x * 255).numpy().astype(np.uint8)
            output.append(x)

        output = Image.fromarray(np.concatenate(output, axis=0))
        outputs.append(output)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    save_videos_from_pil(outputs, path, fps, crf)


def read_frames(video_path):
    """Read video frames using OpenCV."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
    
    cap.release()
    return frames


def get_fps(video_path):
    """Get video FPS using OpenCV."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps
