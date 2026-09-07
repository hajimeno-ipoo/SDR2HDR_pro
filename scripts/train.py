from __future__ import annotations

import argparse
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import torch
import numpy as np
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from sdr2hdr.dataset import HDRSDRPairDataset
from sdr2hdr.model import EnhancementUNet, HDRGuidedEnhancementUNet


def set_training_mode(model: nn.Module, freeze_shared: bool) -> None:
    # BatchNorm statistics are buffers, not gradient-controlled parameters.
    model.train(not freeze_shared)


def initialize_v2_guidance_heads(model: HDRGuidedEnhancementUNet) -> None:
    """Give the new v2 heads a conservative starting point.

    A v2 model initialized from the legacy three-map model has no learned
    luminance or reconstruction heads.  The default random initialization
    produces a mid-range sigmoid output, which can brighten dark frames before
    the new heads have seen enough paired data.  A small-weight, low-confidence
    prior keeps the first epochs close to the legacy result while the heads
    learn from the HDR targets.
    """
    def logit(probability: float) -> float:
        return math.log(probability / (1.0 - probability))

    with torch.no_grad():
        model.luma_head.weight.mul_(0.05)
        model.luma_head.bias.fill_(logit(0.10))
        model.reconstruction_head.weight.mul_(0.05)
        model.reconstruction_head.bias.fill_(logit(0.03))


def split_train_validation_indices(length: int, validation_size: int) -> tuple[list[int], list[int]]:
    """Select validation frames across the sequence instead of only at its end."""
    if length <= 0:
        return [], []
    validation_size = min(max(validation_size, 1), length)
    if validation_size == 1:
        val_indices = [length - 1]
    else:
        val_indices = sorted(
            {
                round(index * (length - 1) / (validation_size - 1))
                for index in range(validation_size)
            }
        )
    val_set = set(val_indices)
    train_indices = [index for index in range(length) if index not in val_set]
    return train_indices or val_indices, val_indices


def split_train_validation_indices_by_group(
    groups: Sequence[str], validation_size: int
) -> tuple[list[int], list[int]]:
    """Split samples by content group so variants of one scene never cross splits."""
    length = len(groups)
    if length <= 1:
        return split_train_validation_indices(length, validation_size)

    grouped_indices: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        grouped_indices[str(group)].append(index)
    ordered_groups = sorted(grouped_indices)
    if len(ordered_groups) <= 1:
        return split_train_validation_indices(length, validation_size)

    validation_size = min(max(validation_size, 1), length - 1)
    group_count = max(
        1,
        min(
            len(ordered_groups) - 1,
            round(len(ordered_groups) * validation_size / length),
        ),
    )
    if group_count == 1:
        selected_positions = [len(ordered_groups) // 2]
    else:
        selected_positions = [
            round(index * (len(ordered_groups) - 1) / (group_count - 1))
            for index in range(group_count)
        ]
    validation_groups = {ordered_groups[position] for position in selected_positions}
    val_indices = sorted(
        index
        for group in validation_groups
        for index in grouped_indices[group]
    )
    val_set = set(val_indices)
    train_indices = [index for index in range(length) if index not in val_set]
    return train_indices, val_indices


def resolve_training_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def validate_content_separation(train_names: Sequence[str], validation_names: Sequence[str]) -> None:
    if not train_names or not validation_names:
        raise ValueError("Training and validation directories must both contain samples")
    overlap = set(train_names) & set(validation_names)
    if overlap:
        raise ValueError(f"Training/validation content overlap: {sorted(overlap)}")


def total_variation_loss(tensor: torch.Tensor) -> torch.Tensor:
    dx = torch.abs(tensor[:, :, :, 1:] - tensor[:, :, :, :-1]).mean()
    dy = torch.abs(tensor[:, :, 1:, :] - tensor[:, :, :-1, :]).mean()
    return dx + dy


def compute_loss(pred: torch.Tensor, target_maps: torch.Tensor, clip_mask: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    confidence = 1.0 - clip_mask * 0.8
    pred_exp, pred_con, pred_pro = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]
    tgt_exp, tgt_con, tgt_pro = target_maps[:, 0:1], target_maps[:, 1:2], target_maps[:, 2:3]
    protected_weight = 1.0 + torch.clamp(tgt_pro, min=0.0) * 1.5
    overdrive = torch.clamp(pred_exp - tgt_exp, min=0.0)
    l_exp = (confidence * protected_weight * torch.abs(pred_exp - tgt_exp)).mean() + (protected_weight * overdrive).mean() * 0.5
    l_con = (confidence * (1.0 + torch.clamp(tgt_pro, min=0.0)) * torch.abs(pred_con - tgt_con)).mean()
    l_pro = torch.abs(pred_pro - tgt_pro).mean()
    l_tv = total_variation_loss(pred_exp) * 0.01
    total = 1.0 * l_exp + 0.5 * l_con + 0.75 * l_pro + l_tv
    return total, {"exp": l_exp, "con": l_con, "pro": l_pro, "tv": l_tv}


def compute_loss_v2(
    pred: torch.Tensor,
    target_maps: torch.Tensor,
    target_luma_rel: torch.Tensor,
    reconstruction_target: torch.Tensor,
    identity_mask: torch.Tensor,
    clip_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pred_maps = pred[:, 0:3]
    pred_luma = pred[:, 3:4]
    pred_recon = pred[:, 4:5]

    # 1. 3-map base loss
    l_map, map_losses = compute_loss(pred_maps, target_maps, clip_mask)

    # 2. log-luma loss
    eps = 1e-4
    l_luma = torch.abs(torch.log(pred_luma + eps) - torch.log(target_luma_rel + eps)).mean()

    # 3. reconstruction loss (for clipped areas)
    l_recon = (reconstruction_target * torch.abs(pred_luma - target_luma_rel)).mean()
    l_conf = nn.functional.binary_cross_entropy(pred_recon, reconstruction_target)

    # 4. identity loss (for non-clipped / normal areas)
    identity_weight = identity_mask * (1.0 + torch.clamp(target_maps[:, 2:3], min=0.0))
    l_identity = (identity_weight * torch.abs(pred_luma - target_luma_rel)).mean()

    # 5. gradient loss (to reconstruct local structures in highlights)
    pred_grad_x = torch.abs(pred_luma[:, :, :, 1:] - pred_luma[:, :, :, :-1])
    pred_grad_y = torch.abs(pred_luma[:, :, 1:, :] - pred_luma[:, :, :-1, :])
    tgt_grad_x = torch.abs(target_luma_rel[:, :, :, 1:] - target_luma_rel[:, :, :, :-1])
    tgt_grad_y = torch.abs(target_luma_rel[:, :, 1:, :] - target_luma_rel[:, :, :-1, :])
    l_grad = (torch.abs(pred_grad_x - tgt_grad_x)).mean() + (torch.abs(pred_grad_y - tgt_grad_y)).mean()

    total = (
        1.0 * l_map
        + 1.0 * l_luma
        + 1.5 * (l_recon + l_conf)
        + 0.8 * l_identity
        + 0.5 * l_grad
    )
    return total, {
        "map": l_map,
        "luma": l_luma,
        "recon": l_recon,
        "conf": l_conf,
        "identity": l_identity,
        "grad": l_grad,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train enhancement map estimator.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--validation-dir", help="Separate, content-disjoint validation NPZ directory")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-version", choices=["legacy", "v2"], default="v2")
    parser.add_argument(
        "--init-checkpoint",
        help="Optional legacy or v2 TorchScript/checkpoint used to initialize shared weights.",
    )
    parser.add_argument(
        "--freeze-shared",
        action="store_true",
        help="When initializing v2 from a legacy checkpoint, train only the new guidance heads.",
    )
    parser.add_argument(
        "--guidance-head-lr",
        type=float,
        default=5e-3,
        help="Learning rate for v2 luminance/reconstruction heads when --freeze-shared is used.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    if args.freeze_shared and (args.model_version != "v2" or not args.init_checkpoint):
        parser.error("--freeze-shared requires --model-version v2 and --init-checkpoint")
    if args.guidance_head_lr <= 0.0:
        parser.error("--guidance-head-lr must be greater than zero")

    output_dir = Path(args.output_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_source = HDRSDRPairDataset(args.data_dir, patch_size=args.patch_size, training=True, seed=args.seed)
    val_source = HDRSDRPairDataset(args.validation_dir or args.data_dir, patch_size=args.patch_size, training=False)
    validation_size = max(1, len(train_source) // 10)
    content_group_count = len(set(train_source.content_names))
    train_indices, val_indices = split_train_validation_indices_by_group(
        train_source.content_names,
        validation_size,
    )
    split_mode = "content-group" if content_group_count > 1 else "frame-fallback (one content group)"
    if args.validation_dir:
        try:
            validate_content_separation(train_source.content_names, val_source.content_names)
        except ValueError as error:
            parser.error(str(error))
        train_indices = list(range(len(train_source)))
        val_indices = list(range(len(val_source)))
        split_mode = "explicit content-disjoint directories"
    print(
        f"validation split: mode={split_mode} content_groups={content_group_count} "
        f"train_samples={len(train_indices)} val_samples={len(val_indices)}"
    )
    train_dataset = Subset(train_source, train_indices)
    val_dataset = Subset(val_source, val_indices)

    output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = resolve_training_device(args.device)
    if args.model_version == "v2":
        model: nn.Module = HDRGuidedEnhancementUNet().to(device)
    else:
        model = EnhancementUNet().to(device)

    if args.init_checkpoint:
        checkpoint_path = Path(args.init_checkpoint)
        try:
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            source_state = payload.get("model", payload) if isinstance(payload, dict) else payload
        except Exception:
            source_state = torch.jit.load(str(checkpoint_path), map_location="cpu").state_dict()
        if hasattr(source_state, "state_dict"):
            source_state = source_state.state_dict()
        source_is_v2 = any(
            key.startswith("luma_head.") or key.startswith("reconstruction_head.")
            for key in source_state.keys()
        )
        if args.model_version == "v2":
            # Reuse the legacy three-map head for v2's first three outputs.
            if "head.0.weight" in source_state:
                source_state = dict(source_state)
                source_state["map_head.weight"] = source_state["head.0.weight"]
                source_state["map_head.bias"] = source_state["head.0.bias"]
                source_state.pop("head.0.weight", None)
                source_state.pop("head.0.bias", None)
            incompatible = model.load_state_dict(source_state, strict=False)
        else:
            incompatible = model.load_state_dict(source_state, strict=False)
        print(
            f"initialized from {checkpoint_path}: "
            f"missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)}"
        )

        if args.model_version == "v2" and not source_is_v2:
            initialize_v2_guidance_heads(model)

    if args.freeze_shared:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("luma_head.") or name.startswith("reconstruction_head.")

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(
        trainable_parameters,
        lr=args.guidance_head_lr if args.freeze_shared else 3e-4,
        weight_decay=1e-4,
    )
    warmup = LinearLR(optimizer, start_factor=0.2, total_iters=3)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - 3))
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[3])
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler(device=device.type, enabled=amp_enabled)
    best_val = float("inf")

    for epoch in range(args.epochs):
        # Frozen shared blocks contain BatchNorm buffers as well as parameters.
        # eval() preserves those buffers while gradients still train the heads.
        set_training_mode(model, args.freeze_shared)
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"train {epoch+1}/{args.epochs}"):
            sdr_linear = batch["sdr_linear"].to(device)
            target_maps = batch["target_maps"].to(device)
            clip_mask = batch["clip_mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                pred = model(sdr_linear)
                if args.model_version == "v2":
                    target_luma_rel = batch["target_luma_rel"].to(device)
                    reconstruction_target = batch["reconstruction_target"].to(device)
                    identity_mask = batch["identity_mask"].to(device)
                    loss, _ = compute_loss_v2(
                        pred,
                        target_maps,
                        target_luma_rel,
                        reconstruction_target,
                        identity_mask,
                        clip_mask,
                    )
                else:
                    loss, _ = compute_loss(pred, target_maps, clip_mask)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(loss.detach().cpu())
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"val {epoch+1}/{args.epochs}"):
                sdr_linear = batch["sdr_linear"].to(device)
                target_maps = batch["target_maps"].to(device)
                clip_mask = batch["clip_mask"].to(device)
                pred = model(sdr_linear)
                if args.model_version == "v2":
                    target_luma_rel = batch["target_luma_rel"].to(device)
                    reconstruction_target = batch["reconstruction_target"].to(device)
                    identity_mask = batch["identity_mask"].to(device)
                    loss, _ = compute_loss_v2(
                        pred,
                        target_maps,
                        target_luma_rel,
                        reconstruction_target,
                        identity_mask,
                        clip_mask,
                    )
                else:
                    loss, _ = compute_loss(pred, target_maps, clip_mask)
                val_loss += float(loss.detach().cpu())
        train_loss /= max(len(train_loader), 1)
        val_loss /= max(len(val_loader), 1)
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_version": args.model_version,
                    "output_channels": 5 if args.model_version == "v2" else 3,
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                output_dir / "best.pt",
            )

    writer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
