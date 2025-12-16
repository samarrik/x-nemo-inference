#!/usr/bin/env python3
"""
Standalone X-Nemo runner script.

This script runs in isolation to avoid import conflicts with other ML repos.

Usage:
    python run_xnemo.py --source video.mp4 --reference face.jpg --output output.mp4
"""

import argparse
import json
import os
import sys

# Ensure we're using this repo's src
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from xnemo_api import GenerationConfig, XNemoGenerator


def main():
    parser = argparse.ArgumentParser(description="X-Nemo Video Generation (Standalone)")
    parser.add_argument("--source", required=True, help="User's video (motion source)")
    parser.add_argument("--reference", required=True, help="Identity's face image")
    parser.add_argument("--output", required=True, help="Output video path")

    # Model paths (optional, uses defaults if not specified)
    parser.add_argument(
        "--pretrained-model", default=None, help="Path to sd-image-variations-diffusers"
    )
    parser.add_argument("--vae-path", default=None, help="Path to VAE")
    parser.add_argument(
        "--denoising-unet", default=None, help="Path to denoising UNet weights"
    )
    parser.add_argument(
        "--temporal-module", default=None, help="Path to temporal module weights"
    )

    # Generation config
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--guidance", type=float, default=2.5)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-frames", type=int, default=None)

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    # Build generator kwargs
    gen_kwargs = {
        "device": args.device,
        "dtype": args.dtype,
        "verbose": not args.quiet,
    }

    if args.pretrained_model:
        gen_kwargs["pretrained_model"] = args.pretrained_model
    if args.vae_path:
        gen_kwargs["vae_path"] = args.vae_path
    if args.denoising_unet:
        gen_kwargs["denoising_unet"] = args.denoising_unet
    if args.temporal_module:
        gen_kwargs["temporal_module"] = args.temporal_module

    # Create generator and run
    generator = XNemoGenerator(**gen_kwargs)

    config = GenerationConfig(
        width=args.width,
        height=args.height,
        steps=args.steps,
        guidance_scale=args.guidance,
        fps=args.fps,
        seed=args.seed,
        max_frames=args.max_frames,
    )

    output_path = generator.generate(
        reference=args.reference,
        source=args.source,
        output=args.output,
        config=config,
    )

    # Output result as JSON for parsing
    result = {"success": True, "output": output_path}
    print(f"\n__RESULT_JSON__:{json.dumps(result)}")


if __name__ == "__main__":
    main()
