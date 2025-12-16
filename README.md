# X-NeMo Video Generation

Generate portrait videos driven by motion from source videos.

## Setup

```bash
# Create environment
conda create -n xnemo python=3.9
conda activate xnemo
pip install -r requirements.txt
```

## Download Models

Download and place in `pretrained_weights/`:

1. [stable-video-diffusion-img2vid-xt](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt) → `pretrained_weights/stable-video-diffusion-img2vid-xt/`
2. [sd-image-variations-diffusers](https://huggingface.co/lambdalabs/sd-image-variations-diffusers) → `pretrained_weights/sd-image-variations-diffusers/`
3. [X-NeMo weights](https://drive.google.com/drive/folders/1RdjBYYbstO7SOchDg7oimoAwu03g_-mI) → `pretrained_weights/xnemo_*.pth`

The face detector model (`blaze_face_short_range.tflite`) is included in the repo.

## Usage

### Python API

```python
from xnemo_api import XNemoGenerator

generator = XNemoGenerator()
generator.generate(reference="face.jpg", source="motion.mp4", output="output.mp4")
```

### CLI

```bash
export PYTHONPATH="$(pwd):$PYTHONPATH"

python xnemo_api.py \
    --source motion.mp4 \
    --reference face.jpg \
    --output output.mp4
```

### Arguments

| Argument | Description |
|----------|-------------|
| `--source` | User's uploaded video (motion source) |
| `--reference` | Identity's face image |
| `--output` | Reenacted video output |

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--steps` | 25 | Denoising steps |
| `--guidance` | 2.5 | Guidance scale |
| `--max_frames` | all | Limit frames |
| `--seed` | 42 | Random seed |
| `--width` | 512 | Output width |
| `--height` | 512 | Output height |

**Note**: The output video's FPS automatically matches the source video's FPS for natural playback speed. There is no separate FPS override option.

## Apptainer Container (Cluster)

Build and run in a container on cluster:

```bash
# Load cluster modules first
module load CUDA/12.2.0
module load FFmpeg

# Build
apptainer build xnemo.sif xnemo.def

# Run
apptainer run --nv \
    --bind /path/to/pretrained_weights:/app/pretrained_weights \
    --bind /path/to/data:/data \
    xnemo.sif --source /data/motion.mp4 --reference /data/face.jpg --output /data/output.mp4

# Interactive shell
apptainer shell --nv --bind /path/to/pretrained_weights:/app/pretrained_weights xnemo.sif
```

## License

Apache License 2.0 for code. Model weights under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) for academic research only.

## Citation

```bibtex
@article{zhao2025x,
  title={X-NeMo: Expressive neural motion reenactment via disentangled latent attention},
  author={Zhao, Xiaochen and Xu, Hongyi and Song, Guoxian and Xie, You and Zhang, Chenxu and Li, Xiu and Luo, Linjie and Suo, Jinli and Liu, Yebin},
  journal={arXiv preprint arXiv:2507.23143},
  year={2025}
}
```
