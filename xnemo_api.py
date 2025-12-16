# *************************************************************************
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
#
# X-NeMo Python API for Video Generation
# *************************************************************************
"""
X-NeMo Video Generation API

Simple API for generating reenacted portrait videos.

Args:
    --source: User's uploaded video (motion source)
    --reference: Identity's face image
    --output: Reenacted video output

Example:
    from xnemo_api import XNemoGenerator
    
    generator = XNemoGenerator()
    generator.generate(reference="face.jpg", source="motion.mp4", output="output.mp4")
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

import cv2
import mediapipe as mp
import numpy as np
import torch
from decord import VideoReader
from diffusers import DDIMScheduler, DPMSolverMultistepScheduler, AutoencoderKLTemporalDecoder
from omegaconf import OmegaConf
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from skimage import img_as_ubyte
from skimage.transform import resize
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPVisionModelWithProjection

from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.models.motion_encoder.encoder import MotEncoder_withExtra as MotEncoder
from src.pipelines.pipeline_pose2vid_motenc_long import Pose2VideoPipeline
from src.utils.util import save_videos_grid


@dataclass
class GenerationConfig:
    """Configuration for video generation.
    
    All parameters have sensible defaults - just use GenerationConfig() for most cases.
    """
    width: int = 512
    height: int = 512
    steps: int = 25
    guidance_scale: float = 2.5
    seed: int = 42
    max_frames: Optional[int] = None
    context_frames: int = 24
    context_overlap: int = 8
    context_batch_size: int = 1
    vae_batch_size: int = 32  # Increased for faster decoding
    num_workers: int = 4


class XNemoGenerator:
    """X-NeMo Video Generator.
    
    Loads models once, then can generate multiple videos efficiently.
    
    Example:
        generator = XNemoGenerator()
        generator.generate(reference="face.jpg", source="motion.mp4", output="output.mp4")
    """
    
    def __init__(
        self,
        pretrained_model: str = "pretrained_weights/sd-image-variations-diffusers",
        vae_path: str = "pretrained_weights/stable-video-diffusion-img2vid-xt/vae",
        denoising_unet: str = "pretrained_weights/xnemo_denoising_unet.pth",
        temporal_module: str = "pretrained_weights/xnemo_temporal_module.pth",
        face_detector: str = "blaze_face_short_range.tflite",
        inference_config: str = "configs/inference/inference_xnemo_stage2.yaml",
        device: str = "cuda",
        dtype: str = "fp16",
        verbose: bool = True,
    ):
        self.device = device
        self.verbose = verbose
        self.face_detector_path = face_detector
        
        self.weight_dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }.get(dtype, torch.float32)
        
        if self.verbose:
            print(f"{'=' * 50}")
            print(f"X-NeMo Generator")
            print(f"{'=' * 50}")
            print(f"Device: {device}, Dtype: {dtype}")
        
        if "cuda" in device:
            self._setup_cuda()
        
        self._load_models(
            pretrained_model, vae_path, denoising_unet,
            temporal_module, inference_config
        )
        
        if self.verbose:
            print(f"✓ Ready")
            print(f"{'=' * 50}")
    
    def _setup_cuda(self):
        """Setup CUDA optimizations for maximum speed."""
        if torch.cuda.is_available():
            # Enable TF32 for faster matrix operations on Ampere+ GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Enable cuDNN autotuning for optimal kernel selection
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False  # Faster, non-deterministic
            
            torch.set_default_dtype(torch.float16)
            
            # Enable all SDPA backends for maximum flexibility
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_math_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(True)
            
            # Disable gradient computation globally for inference
            torch.set_grad_enabled(False)
            
            if self.verbose:
                gpu = torch.cuda.get_device_name(self.device)
                mem = torch.cuda.get_device_properties(self.device).total_memory / 1024**3
                print(f"GPU: {gpu} ({mem:.1f} GB)")
    
    def _load_models(self, pretrained_model, vae_path, denoising_unet_path,
                     temporal_module_path, inference_config_path):
        """Load all models."""
        if self.verbose:
            print("Loading models...")
        
        self.infer_config = OmegaConf.load(inference_config_path)
        
        # VAE
        self.vae = AutoencoderKLTemporalDecoder.from_pretrained(
            vae_path
        ).to(self.device, dtype=self.weight_dtype)
        
        # Reference UNet
        self.reference_unet = UNet2DConditionModel.from_pretrained(
            pretrained_model, subfolder="unet"
        ).to(device=self.device, dtype=self.weight_dtype)
        
        # Denoising UNet
        self.denoising_unet = UNet3DConditionModel.from_pretrained_2d(
            pretrained_model, "", subfolder="unet",
            unet_additional_kwargs=self.infer_config.unet_additional_kwargs,
        ).to(dtype=self.weight_dtype, device=self.device)
        
        # Motion Encoder
        self.motion_encoder = MotEncoder().to(dtype=self.weight_dtype, device=self.device)
        self.motion_encoder.eval()
        
        # Image Encoder
        self.image_enc = CLIPVisionModelWithProjection.from_pretrained(
            os.path.join(pretrained_model, "image_encoder")
        ).to(dtype=self.weight_dtype, device=self.device)
        
        # Load weights
        self.denoising_unet.load_state_dict(
            torch.load(denoising_unet_path, map_location="cpu", weights_only=True),
            strict=False
        )
        self.reference_unet.load_state_dict(
            torch.load(denoising_unet_path.replace('denoising_unet', 'reference_unet'),
                      map_location="cpu", weights_only=True),
            strict=True
        )
        self.motion_encoder.load_state_dict(
            torch.load(denoising_unet_path.replace('denoising_unet', 'motion_encoder'),
                      map_location="cpu", weights_only=True),
            strict=True
        )
        self.denoising_unet.load_state_dict(
            torch.load(temporal_module_path, map_location="cpu", weights_only=True),
            strict=False
        )
        
        # Scheduler
        sched_kwargs = OmegaConf.to_container(self.infer_config.noise_scheduler_kwargs)
        self.scheduler = DPMSolverMultistepScheduler(
            beta_start=sched_kwargs.get("beta_start", 0.00085),
            beta_end=sched_kwargs.get("beta_end", 0.02),
            beta_schedule=sched_kwargs.get("beta_schedule", "scaled_linear"),
            algorithm_type="dpmsolver++",
            solver_order=2,
            prediction_type=sched_kwargs.get("prediction_type", "epsilon"),
        )
        
        # Pipeline
        self.pipe = Pose2VideoPipeline(
            vae=self.vae,
            image_encoder=self.image_enc,
            reference_unet=self.reference_unet,
            denoising_unet=self.denoising_unet,
            motion_encoder=self.motion_encoder,
            scheduler=self.scheduler,
        ).to(self.device, dtype=self.weight_dtype)
        
        self.pipe.enable_vae_slicing()
        self.pipe.enable_vae_tiling()
        self.pipe.enable_xformers_memory_efficient_attention()
        
        # Apply torch.compile for faster inference (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            try:
                self.denoising_unet = torch.compile(
                    self.denoising_unet, 
                    mode="reduce-overhead",
                    fullgraph=False
                )
                if self.verbose:
                    print("✓ torch.compile enabled for UNet")
            except Exception as e:
                if self.verbose:
                    print(f"⚠ torch.compile not available: {e}")
        
        if self.verbose:
            print(f"✓ Models loaded")
    
    def generate(
        self,
        reference: Union[str, Path, Image.Image, np.ndarray],
        source: Union[str, Path],
        output: str,
        config: Optional[GenerationConfig] = None,
    ) -> str:
        """Generate reenacted video.
        
        Args:
            reference: Identity's face image (path, PIL Image, or numpy array)
            source: User's uploaded video (motion source)
            output: Path for reenacted video output
            config: Optional generation config (uses defaults if None)
        
        Returns:
            Path to generated video
        """
        start = time.time()
        config = config or GenerationConfig()
        
        # Validate inputs
        ref_path = str(reference) if isinstance(reference, (str, Path)) else None
        source_path = str(source)
        
        if ref_path and not os.path.exists(ref_path):
            raise FileNotFoundError(f"Reference not found: {ref_path}")
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source video not found: {source_path}")
        
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        
        if self.verbose:
            print(f"\nGenerating:")
            print(f"  Reference: {ref_path or 'PIL/numpy'}")
            print(f"  Source: {source_path}")
            print(f"  Output: {output}")
        
        # Preprocess (also returns source video FPS)
        data = self._preprocess(reference, source_path, config)
        
        # Generate
        with torch.inference_mode():
            video = self.pipe(
                data['ref_image'],
                data['pose_images'],
                data['ref_pose'],
                config.width, config.height,
                len(data['pose_images']),
                num_inference_steps=config.steps,
                guidance_scale=config.guidance_scale,
                generator=data['generator'],
                init_latents=None,
                mot_bbox_param=data['bbox_params'],
                context_frames=config.context_frames,
                context_overlap=config.context_overlap,
                context_batch_size=config.context_batch_size,
                decode_chunk_size=config.vae_batch_size,
            ).videos
        
        # Save using the source video's FPS for correct playback speed
        save_videos_grid(video, output, n_rows=1, fps=data["fps"], crf=18)
        
        if self.verbose:
            elapsed = time.time() - start
            fps = len(data['pose_images']) / elapsed
            print(f"✓ Done in {elapsed:.1f}s ({fps:.1f} fps)")
        
        return output
    
    def _preprocess(self, reference, source_path, config):
        """Preprocess inputs for generation.
        
        Returns a dict containing:
            - ref_image: cropped reference face image (PIL)
            - ref_pose: reference pose image (PIL)
            - pose_images: list of pose images (PIL)
            - bbox_params: tensor of bbox parameters
            - generator: torch.Generator
            - fps: source video frames-per-second (float)
        """
        width, height = config.width, config.height
        
        transform = transforms.Compose([
            transforms.Resize((height, width)),
            transforms.ToTensor()
        ])
        
        # Face detector
        options = mp.tasks.vision.FaceDetectorOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self.face_detector_path),
        )
        img_detector = mp.tasks.vision.FaceDetector.create_from_options(options)
        
        # Load reference image
        if isinstance(reference, (str, Path)):
            path = str(reference)
            if path.endswith('.mp4'):
                ref_img = VideoReader(path)[0].asnumpy()
            else:
                ref_img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        elif isinstance(reference, Image.Image):
            ref_img = np.array(reference.convert('RGB'))
        else:
            ref_img = reference if reference.ndim == 3 else reference[..., :3]
        
        # Load source video
        control = VideoReader(source_path)
        fps = float(control.get_avg_fps())
        total = min(len(control), config.max_frames or len(control))
        
        if self.verbose:
            print(f"  Frames: {total}")
        
        control = control.get_batch(list(range(total))).asnumpy()
        
        # Process reference
        ref_bbox = self._extract_bbox(ref_img, None, img_detector)
        left, top, right, bot = self._scale_bbox(ref_bbox, 1.1, ref_img.shape[:2])
        ref_pose = Image.fromarray(ref_img[int(top):int(bot), int(left):int(right)]).convert("RGB")
        
        ref_img = self._crop_face([ref_img], ref_bbox)[0]
        ref_pil = Image.fromarray(ref_img).convert("RGB")
        
        # Generator
        generator = torch.Generator(device=self.device)
        generator.manual_seed(config.seed)
        
        # Video face detector
        options = mp.tasks.vision.FaceDetectorOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self.face_detector_path),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
        )
        detector = mp.tasks.vision.FaceDetector.create_from_options(options)
        
        # Process frames
        centers = []
        bbox_params = []
        fix_length = None
        ts = 0
        
        for i in tqdm(range(len(control)), desc="Face detection", disable=not self.verbose):
            frame = control[i][:, :control[0].shape[0]]
            ts += int(1000 / fps)
            bbox = self._extract_bbox(frame, None, detector, ts)
            
            param = torch.ones(3, dtype=torch.float32)
            param[:2] = 0
            bbox_params.append(param)
            
            if i == 0:
                bbox = self._scale_bbox(bbox, 1.1, frame.shape[:2])
                fix_length = (round(bbox[2] - bbox[0]) // 2 * 2,
                             round(bbox[3] - bbox[1]) // 2 * 2)
            
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            centers.append([cx, cy])
        
        centers = gaussian_filter1d(np.array(centers, dtype=np.float32), sigma=5.0, axis=0)
        bbox_params = torch.stack(bbox_params)
        
        # Parallel crop
        args = [(i, control[i], control[0].shape[0], centers, fix_length) for i in range(len(control))]
        with ThreadPoolExecutor(max_workers=config.num_workers) as ex:
            results = list(tqdm(ex.map(self._process_pose, args), total=len(control),
                               desc="Processing", disable=not self.verbose))
        
        pose_images = [r[1] for r in sorted(results, key=lambda x: x[0])]
        
        return {
            "ref_image": ref_pil,
            "ref_pose": ref_pose,
            "pose_images": pose_images,
            "bbox_params": bbox_params,
            "generator": generator,
            "fps": fps,
        }
    
    @staticmethod
    def _extract_bbox(frame, refbbox, detector, ts=None):
        """Extract face bounding box."""
        if refbbox is None:
            refbbox = (0, 0, frame.shape[1], frame.shape[0])
        
        image = mp.Image(image_format=mp.ImageFormat.SRGB,
                        data=np.asarray(frame, dtype=np.uint8).copy())
        
        result = detector.detect(image) if ts is None else detector.detect_for_video(image, ts)
        
        if result.detections:
            det = max(result.detections, key=lambda d: d.categories[0].score)
            b = det.bounding_box
            bbox = np.array([b.origin_x, b.origin_y, b.origin_x + b.width, b.origin_y + b.height])
        else:
            h, w = frame.shape[:2]
            size = min(h, w) * 0.8
            bbox = np.array([w/2 - size/2, h/2 - size/2, w/2 + size/2, h/2 + size/2])
        
        return np.maximum(bbox, 0)
    
    @staticmethod
    def _scale_bbox(bbox, scale, size):
        """Scale bounding box."""
        left, top, right, bot = bbox[:4]
        length = max(right - left, bot - top) * scale
        cx, cy = (left + right) / 2, (top + bot) / 2
        new = [cx - length/2, cy - length/2, cx + length/2, cy + length/2]
        
        if new[0] < 0 or new[1] < 0 or new[2] > size[1]-1 or new[3] > size[0]-1:
            return bbox
        return np.array(new)
    
    @staticmethod
    def _crop_face(frames, bbox, increase=0.5):
        """Crop face from frames."""
        h, w = frames[0].shape[:2]
        left, top, right, bot = bbox
        width, height = right - left, bot - top
        
        wi = max(increase, ((1 + 2*increase) * height - width) / (2 * width))
        hi = max(increase, ((1 + 2*increase) * width - height) / (2 * height))
        
        left = int(left - wi * width)
        top = int(top - hi * height)
        right = int(right + wi * width)
        bot = int(bot + hi * height)
        
        pad = max(-left, right - w, -top, bot - h, 0)
        if pad > 0:
            frames = [np.pad(f, ((pad, pad), (pad, pad), (0, 0)), 'constant') for f in frames]
            left, top, right, bot = left + pad, top + pad, right + pad, bot + pad
        
        return [img_as_ubyte(resize(f[top:bot, left:right], (512, 512), anti_aliasing=True))
                for f in frames]
    
    @staticmethod
    def _get_bbox_from_center(center, length, size):
        """Get bbox from center point."""
        cx, cy = center
        w, h = length
        box = [cx - w/2, cy - h/2, cx + w/2, cy + h/2]
        
        if box[0] < 0 or box[1] < 0 or box[2] > size[1]-1 or box[3] > size[0]-1:
            xoff = max(-box[0], 0) + min(size[1]-1 - box[2], 0)
            yoff = max(-box[1], 0) + min(size[0]-1 - box[3], 0)
            return np.array(box) + np.array([xoff, yoff, xoff, yoff])
        return np.array(box)
    
    @staticmethod
    def _process_pose(args):
        """Process single pose frame."""
        idx, frame, shape, centers, fix_length = args
        pose = frame[:, :shape]
        bbox = XNemoGenerator._get_bbox_from_center(centers[idx], fix_length, pose.shape[:2])
        left, top, right, bot = bbox
        cropped = pose[int(top):int(bot), int(left):int(right)]
        return idx, Image.fromarray(cropped).convert("RGB")


# CLI
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="X-NeMo Video Generation")
    parser.add_argument("--source", required=True, help="User's uploaded video (motion source)")
    parser.add_argument("--reference", required=True, help="Identity's face image")
    parser.add_argument("--output", required=True, help="Reenacted video output")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--guidance", type=float, default=2.5)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    
    args = parser.parse_args()
    
    config = GenerationConfig(
        width=args.width,
        height=args.height,
        steps=args.steps,
        guidance_scale=args.guidance,
        max_frames=args.max_frames,
        seed=args.seed,
    )
    
    generator = XNemoGenerator()
    generator.generate(args.reference, args.source, args.output, config)
