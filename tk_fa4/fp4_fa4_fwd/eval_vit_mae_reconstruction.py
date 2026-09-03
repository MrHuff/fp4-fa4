#!/usr/bin/env python3
"""Paired BF16/FP4 ViT-MAE reconstruction evaluation.

Only the twelve encoder self-attention layers are replaced. The decoder,
mask, model weights, and input pixels are identical between the two runs.
This is an accuracy harness; dynamic input quantization is intentionally not
included in the kernel timings reported elsewhere.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
from typing import Any

try:
    from .eval_regular_attention import (
        DEFAULT_HAO_ROOT,
        MX_QUANT_MODES,
        RegularAttentionRunner,
        authenticate_asset_manifest,
        install_vit_attention,
        load_extension,
        portable_file_identity,
        tensor_metrics,
    )
except ImportError:  # direct script execution
    from eval_regular_attention import (
        DEFAULT_HAO_ROOT,
        MX_QUANT_MODES,
        RegularAttentionRunner,
        authenticate_asset_manifest,
        install_vit_attention,
        load_extension,
        portable_file_identity,
        tensor_metrics,
    )


DEFAULT_OUTPUT = Path(
    "../../results/fp4_fa4_reconstruction_20260805/fast.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="facebook/vit-mae-base")
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument(
        "--asset-manifest",
        type=Path,
        required=True,
        help="fa4_external_assets_v1 manifest rechecked before model use.",
    )
    parser.add_argument("--model-asset", required=True)
    parser.add_argument("--image-asset", required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--mask-value", type=float, default=10.0)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument(
        "--extension-module",
        default="_C_tk_hao_direct_fp4pv",
    )
    parser.add_argument(
        "--attention-backend",
        choices=("tk", "hao-native", "hao-fp8"),
        default="tk",
    )
    parser.add_argument("--hao-root", type=Path, default=DEFAULT_HAO_ROOT)
    parser.add_argument(
        "--global-anchor-kv",
        action="store_true",
    )
    parser.add_argument(
        "--global-anchor-samples",
        type=int,
        choices=(32, 64, 128),
        default=32,
    )
    parser.add_argument(
        "--mx-q-quant-mode",
        choices=tuple(MX_QUANT_MODES),
        default="rte",
    )
    parser.add_argument(
        "--mx-k-quant-mode",
        choices=tuple(MX_QUANT_MODES),
        default="rte",
    )
    parser.add_argument(
        "--mx-v-quant-mode",
        choices=tuple(MX_QUANT_MODES),
        default="rte",
    )
    parser.add_argument(
        "--nv-qk-fold-k64-scales",
        choices=("auto", "none", "q", "k", "both"),
        default="auto",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--grid", type=Path)
    return parser.parse_args()


def image_files(root: Path) -> list[Path]:
    suffixes = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted(
        path for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in suffixes
    )


def denormalize(image: Any, mean: list[float], std: list[float]) -> Any:
    import torch

    mean_tensor = torch.tensor(
        mean, device=image.device, dtype=torch.float32
    ).view(1, 3, 1, 1)
    std_tensor = torch.tensor(
        std, device=image.device, dtype=torch.float32
    ).view(1, 3, 1, 1)
    return image.float() * std_tensor + mean_tensor


def patch_mask_image(model: Any, mask: Any) -> Any:
    patch_size = int(model.config.patch_size)
    channels = int(model.config.num_channels)
    expanded = mask.unsqueeze(-1).repeat(
        1, 1, patch_size * patch_size * channels
    )
    return model.unpatchify(expanded).float()


def masked_mse(actual: Any, target: Any, mask: Any) -> float:
    error = (actual.float() - target.float()).square().mean(dim=-1)
    value = (error * mask.float()).sum() / mask.sum().clamp_min(1)
    return float(value.item())


def image_metrics(
    prediction: Any,
    target: Any,
    mask_image: Any,
) -> dict[str, float]:
    import torch

    squared = (prediction - target).square() * mask_image
    denominator = (mask_image.sum() * target.shape[1]).clamp_min(1)
    mse = squared.sum() / denominator
    psnr = -10.0 * torch.log10(mse.clamp_min(torch.finfo(torch.float32).tiny))
    return {
        "masked_rgb_mse": float(mse.item()),
        "masked_rgb_psnr_db": float(psnr.item()),
    }


def tensor_to_pil(tensor: Any) -> Any:
    import torch
    from PIL import Image

    array = (
        tensor.detach()
        .float()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to("cpu", dtype=torch.uint8)
        .permute(1, 2, 0)
        .numpy()
    )
    return Image.fromarray(array)


def write_grid(rows: list[dict[str, Any]], path: Path) -> None:
    from PIL import Image, ImageDraw

    if not rows:
        return
    labels = ("Input", "75% masked", "BF16", "FP4", "8x difference")
    tile_width, tile_height = rows[0]["input"].size
    label_height = 24
    canvas = Image.new(
        "RGB",
        (tile_width * len(labels), label_height + tile_height * len(rows)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for column, label in enumerate(labels):
        draw.text((column * tile_width + 5, 5), label, fill="black")
    for row_index, row in enumerate(rows):
        y = label_height + row_index * tile_height
        for column, key in enumerate(
            ("input", "masked", "baseline", "fp4", "difference")
        ):
            canvas.paste(row[key], (column * tile_width, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def mean_metrics(records: list[dict[str, Any]], key: str) -> dict[str, float]:
    names = records[0][key]
    return {
        name: statistics.fmean(record[key][name] for record in records)
        for name in names
    }


def main() -> None:
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, ViTMAEForPreTraining

    args = parse_args()
    model_root = Path(args.model).resolve()
    image_root = args.image_dir.resolve()
    model_identity, asset_manifest_identity = authenticate_asset_manifest(
        args.asset_manifest,
        args.model_asset,
        "model",
        model_root,
    )
    image_identity, second_manifest_identity = authenticate_asset_manifest(
        args.asset_manifest,
        args.image_asset,
        "image set",
        image_root,
    )
    if second_manifest_identity != asset_manifest_identity:
        raise RuntimeError("asset manifest changed while it was being authenticated")
    extension_identity = portable_file_identity(args.extension)
    if args.samples < 1:
        raise ValueError("--samples must be positive")
    paths = image_files(image_root)
    if len(paths) < args.samples:
        raise ValueError(
            f"requested {args.samples} images, found {len(paths)} in "
            f"{image_root}"
        )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.set_device(0)
    extension = load_extension(
        args.extension.resolve(), args.extension_module
    )
    runner = RegularAttentionRunner(
        extension,
        attention_backend=args.attention_backend,
        hao_root=args.hao_root,
        mask_value=args.mask_value,
        scale_factors=[],
        scale_sweep_samples=0,
        global_anchor=args.global_anchor_kv,
        global_anchor_samples=args.global_anchor_samples,
        mx_q_quant_mode=MX_QUANT_MODES[args.mx_q_quant_mode],
        mx_k_quant_mode=MX_QUANT_MODES[args.mx_k_quant_mode],
        mx_v_quant_mode=MX_QUANT_MODES[args.mx_v_quant_mode],
        nv_qk_fold_k64_scales=args.nv_qk_fold_k64_scales,
    )
    processor = AutoImageProcessor.from_pretrained(
        str(model_root),
        local_files_only=True,
    )
    model = ViTMAEForPreTraining.from_pretrained(
        str(model_root),
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).eval().cuda()
    install_vit_attention(model, runner)

    image_size = int(model.config.image_size)
    patch_size = int(model.config.patch_size)
    patch_count = (image_size // patch_size) ** 2
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    records: list[dict[str, Any]] = []
    grid_rows: list[dict[str, Any]] = []

    with torch.inference_mode():
        for index, path in enumerate(paths[:args.samples]):
            source = Image.open(path).convert("RGB")
            pixel_values = processor(
                images=source, return_tensors="pt"
            ).pixel_values.to(device="cuda", dtype=torch.bfloat16)
            noise = torch.rand(
                (1, patch_count), device="cuda", generator=generator
            )

            runner.enabled = False
            baseline = model(pixel_values=pixel_values, noise=noise)
            runner.begin_sample(index)
            runner.enabled = True
            fp4 = model(pixel_values=pixel_values, noise=noise)

            if not torch.equal(baseline.mask, fp4.mask):
                raise RuntimeError("paired reconstruction masks differ")
            target_patches = model.patchify(pixel_values.float())
            baseline_logits = baseline.logits.float()
            fp4_logits = fp4.logits.float()
            mask = baseline.mask.float()
            mask_image = patch_mask_image(model, mask)

            target_image = denormalize(
                pixel_values, processor.image_mean, processor.image_std
            ).clamp(0.0, 1.0)
            baseline_image = denormalize(
                model.unpatchify(baseline_logits),
                processor.image_mean,
                processor.image_std,
            ).clamp(0.0, 1.0)
            fp4_image = denormalize(
                model.unpatchify(fp4_logits),
                processor.image_mean,
                processor.image_std,
            ).clamp(0.0, 1.0)
            baseline_composite = (
                target_image * (1.0 - mask_image)
                + baseline_image * mask_image
            )
            fp4_composite = (
                target_image * (1.0 - mask_image)
                + fp4_image * mask_image
            )

            baseline_quality = image_metrics(
                baseline_image, target_image, mask_image
            )
            baseline_quality["masked_normalized_mse"] = masked_mse(
                baseline_logits, target_patches, mask
            )
            fp4_quality = image_metrics(
                fp4_image, target_image, mask_image
            )
            fp4_quality["masked_normalized_mse"] = masked_mse(
                fp4_logits, target_patches, mask
            )
            records.append(
                {
                    "index": index,
                    "image": path.name,
                    "baseline_quality": baseline_quality,
                    "fp4_quality": fp4_quality,
                    "fp4_vs_bf16_patches": tensor_metrics(
                        fp4_logits, baseline_logits
                    ),
                    "fp4_vs_bf16_reconstruction": tensor_metrics(
                        fp4_composite, baseline_composite
                    ),
                }
            )

            if args.grid is not None:
                masked = target_image * (1.0 - mask_image) + 0.5 * mask_image
                difference = (
                    (fp4_composite - baseline_composite).abs() * 8.0
                )
                grid_rows.append(
                    {
                        "input": tensor_to_pil(target_image[0]),
                        "masked": tensor_to_pil(masked[0]),
                        "baseline": tensor_to_pil(baseline_composite[0]),
                        "fp4": tensor_to_pil(fp4_composite[0]),
                        "difference": tensor_to_pil(difference[0]),
                    }
                )
            print(
                f"[{index + 1}/{args.samples}] {path.name}: "
                f"BF16 {baseline_quality['masked_rgb_psnr_db']:.3f} dB, "
                f"FP4 {fp4_quality['masked_rgb_psnr_db']:.3f} dB",
                flush=True,
            )

    result = {
        "schema": "tk_fp4_vit_mae_reconstruction_v2",
        "model": model_identity,
        "images_asset": image_identity,
        "asset_manifest": asset_manifest_identity,
        "seed": args.seed,
        "images": [path.name for path in paths[:args.samples]],
        "mask_ratio": float(model.config.mask_ratio),
        "replaced_attention_layers": len(model.vit.encoder.layer),
        "model_attention_shape": {
            "visible_tokens_with_cls": int(
                patch_count * (1.0 - model.config.mask_ratio) + 1
            ),
            "heads": int(model.config.num_attention_heads),
            "head_dim": int(
                model.config.hidden_size / model.config.num_attention_heads
            ),
        },
        "kernel_shape": {
            "seqlen": runner.target_seqlen,
            "heads": runner.target_heads,
            "head_dim": runner.target_dim,
        },
        "extension": extension_identity,
        "topology": runner.topology,
        "summary": {
            "baseline_quality": mean_metrics(records, "baseline_quality"),
            "fp4_quality": mean_metrics(records, "fp4_quality"),
            "fp4_vs_bf16_patches": mean_metrics(
                records, "fp4_vs_bf16_patches"
            ),
            "fp4_vs_bf16_reconstruction": mean_metrics(
                records, "fp4_vs_bf16_reconstruction"
            ),
            "attention": runner.summary(),
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(args.output)
    if args.grid is not None:
        write_grid(grid_rows, args.grid)
    print(json.dumps(result["summary"], indent=2), flush=True)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
