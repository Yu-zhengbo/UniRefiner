"""UniRefiner algorithm flow."""

from __future__ import annotations

import logging
import os

from einops import repeat
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from unirefiner.models.channel_mask import apply_channel_mask
from unirefiner.spurious_filtering import (
    analyze_adaptive_spurious_detection,
    analyze_attention_hijacking,
    analyze_fp_gp_similarity,
)

from .crops import (
    compose_crop_with_background,
    roi_align_feature_map,
    roi_align_token_grid,
    sample_random_crop_boxes,
)
from .losses import (
    compute_refinement_loss,
    compute_spatial_consistency_loss,
)
from .registers import normalize_register_fill, surround_image_with_registers

def reset_teacher_hook_cache(teacher_hook_cache: dict) -> None:
    teacher_hook_cache.clear()
    teacher_hook_cache["record"] = False
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class UniRefinerMethod:
    """Open-source UniRefiner training method."""

    def __init__(self):
        self.teacher_hook_cache = {"record": False}
        self.global_iteration = 0
        self.pca_eigenvectors = None

    def ensure_teacher_hook_ready(self, teacher_model, args) -> None:
        if not getattr(args, "_teacher_hook_ready", False):
            teacher_model.hook_prepare(self.teacher_hook_cache)
            args._teacher_hook_ready = True

    def maybe_render_pca_snapshot(self, model, teacher_model, args, reg_factor_scale: int = 1) -> None:
        vis_interval = getattr(args, "vis_pca_interval", 0)
        vis_test_image = getattr(args, "vis_pca_test_image", None)
        if (
            getattr(args, "rank", 0) == 0
            and vis_interval > 0
            and vis_test_image
            and self.global_iteration % vis_interval == 0
        ):
            self.save_pca_snapshot(model, teacher_model, vis_test_image, reg_factor_scale=reg_factor_scale, args=args)
        self.global_iteration += 1

    @torch.no_grad()
    def save_pca_snapshot(self, model, teacher_model, test_image_path: str, reg_factor_scale: int, args) -> None:
        if not os.path.isfile(test_image_path):
            return

        from unirefiner.diagnostics.pca import pca_visualize

        model_eval = model.module if hasattr(model, "module") else model
        teacher_eval = teacher_model.module if hasattr(teacher_model, "module") else teacher_model
        patch_size = model_eval.patch_size
        base_tokens = 64
        base_image_size = base_tokens * patch_size
        register_size = max(base_tokens // args.reg_factor, 1) * reg_factor_scale
        save_dir = args.vis_pca_save_dir or os.path.join(args.logs_path, args.name, "pca_vis")
        os.makedirs(save_dir, exist_ok=True)

        device = next(model_eval.parameters()).device
        dtype = next(model_eval.parameters()).dtype
        image = Image.open(test_image_path).convert("RGB")
        transform = Compose(
            [
                Resize((base_image_size, base_image_size)),
                ToTensor(),
                Normalize(mean=model_eval.image_mean, std=model_eval.image_std),
            ]
        )
        image_tensor = transform(image).unsqueeze(0).to(device=device, dtype=dtype, non_blocking=True)
        image_with_registers, _ = surround_image_with_registers(
            image_tensor,
            patch_size=patch_size,
            register_size=register_size,
            register_fill=normalize_register_fill(args.register_fill),
        )

        student_tokens = F.normalize(
            apply_channel_mask(model_eval.encode_dense(image_with_registers), args=args)[0],
            dim=-1,
        )
        teacher_tokens = F.normalize(
            apply_channel_mask(teacher_eval.encode_dense(image_with_registers), args=args)[0],
            dim=-1,
        )
        height = image_with_registers.shape[-2] // patch_size
        width = image_with_registers.shape[-1] // patch_size

        if self.pca_eigenvectors is None:
            student_pca_image, self.pca_eigenvectors = pca_visualize(
                student_tokens.float(),
                width=width,
                height=height,
                output_size=512,
                return_eigen=True,
            )
        else:
            student_pca_image = pca_visualize(
                student_tokens.float(),
                width=width,
                height=height,
                output_size=512,
                eigen=self.pca_eigenvectors,
            )
        teacher_pca_image = pca_visualize(
            teacher_tokens.float(),
            width=width,
            height=height,
            output_size=512,
            eigen=self.pca_eigenvectors,
        )

        merged = Image.new(
            "RGB",
            (
                student_pca_image.width + teacher_pca_image.width,
                max(student_pca_image.height, teacher_pca_image.height),
            ),
        )
        merged.paste(student_pca_image, (0, 0))
        merged.paste(teacher_pca_image, (student_pca_image.width, 0))
        output_path = os.path.join(save_dir, f"pca_iter_{self.global_iteration:07d}.png")
        merged.save(output_path)
        logging.info("saved PCA snapshot: %s", output_path)

        if getattr(args, "wandb_run", None) is not None:
            try:
                import wandb

                args.wandb_run.log(
                    {
                        "diagnostics/pca": wandb.Image(
                            merged,
                            caption=f"left=student right=teacher iter={self.global_iteration}",
                        ),
                        "diagnostics/pca_path": output_path,
                    },
                    step=getattr(args, "global_step", self.global_iteration),
                    commit=False,
                )
            except Exception as error:
                logging.warning("wandb PCA logging failed: %s", error)

    def forward(self, batch, model, teacher_model, device, cast_dtype, distributed, args):
        images, background_images = batch
        images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
        background_images = background_images.to(device=device, dtype=cast_dtype, non_blocking=True)
        batch_size = images.shape[0]
        image_size = images.shape[-1]

        if distributed:
            model = model.module

        self.ensure_teacher_hook_ready(teacher_model, args)
        self.maybe_render_pca_snapshot(model, teacher_model, args=args, reg_factor_scale=1)

        try:
            # Global register image and random local crop proposals.
            register_size = max(image_size // model.patch_size // args.reg_factor, 1)
            images_with_registers, register_token_mask = surround_image_with_registers(
                images,
                patch_size=model.patch_size,
                register_size=register_size,
                register_fill=normalize_register_fill(args.register_fill),
            )

            num_proposals = int(getattr(args, "num_proposals", 3))
            if num_proposals <= 0:
                raise ValueError(f"num_proposals must be positive, got {num_proposals}.")
            crop_scale = (0.25, 0.5)
            crop_boxes = sample_random_crop_boxes(
                scale=crop_scale,
                ratio=(2 / 3, 3 / 2),
                num_boxes=num_proposals * batch_size,
            )
            crop_boxes = torch.from_numpy(crop_boxes).to(device=device).reshape(batch_size, -1, 4)

            crop_images = roi_align_feature_map(
                images.float(),
                crop_boxes,
                size=image_size,
            ).to(dtype=cast_dtype)

            # Student features: crop tokens, register tokens, and full-image tokens.
            student_dense_tokens = apply_channel_mask(model.encode_dense(images_with_registers), args=args)
            _, _, feature_dim = student_dense_tokens.shape
            student_register_tokens = student_dense_tokens[register_token_mask].reshape(
                batch_size,
                1,
                -1,
                feature_dim,
            )
            student_register_tokens = student_register_tokens.expand(-1, num_proposals, -1, -1).reshape(
                batch_size * num_proposals,
                -1,
                feature_dim,
            )

            student_full_image_tokens = student_dense_tokens[~register_token_mask].reshape(
                batch_size,
                -1,
                feature_dim,
            )
            student_full_image_tokens = F.normalize(student_full_image_tokens, dim=-1)
            student_crop_tokens = roi_align_token_grid(student_full_image_tokens, crop_boxes)
            student_crop_tokens = F.normalize(student_crop_tokens, dim=-1)
            student_register_tokens = F.normalize(student_register_tokens, dim=-1)
            repeated_student_full_image_tokens = repeat(
                student_full_image_tokens,
                "b n c -> (b p) n c",
                p=num_proposals,
            )

            # Teacher features for composited crops and spurious-token filters.
            with torch.no_grad():
                composited_crop_images, foreground_token_mask = compose_crop_with_background(
                    crop_images,
                    background_images,
                    ratio=0.35,
                    patch_size=model.patch_size,
                    square=bool(getattr(teacher_model, "requires_square_inputs", False)),
                )

                self.teacher_hook_cache["record"] = True
                teacher_composited_tokens = apply_channel_mask(
                    teacher_model.encode_dense(composited_crop_images),
                    args=args,
                )
                self.teacher_hook_cache["record"] = False
                teacher_composited_tokens = F.normalize(teacher_composited_tokens, dim=-1)

                teacher_background_tokens = teacher_composited_tokens[~foreground_token_mask].reshape(
                    batch_size * num_proposals,
                    -1,
                    feature_dim,
                )
                teacher_foreground_crop_tokens = teacher_composited_tokens[foreground_token_mask].reshape(
                    batch_size * num_proposals,
                    -1,
                    feature_dim,
                )

                teacher_full_image_tokens = F.normalize(
                    apply_channel_mask(teacher_model.encode_dense(images), args=args),
                    dim=-1,
                )
                teacher_full_image_tokens = repeat(
                    teacher_full_image_tokens,
                    "b n c -> (b p) n c",
                    p=num_proposals,
                )

                # Map crop-teacher foreground tokens back to the
                # original full-image teacher space. This keeps the supervision
                # target on the teacher's normal image-token distribution while
                # the local teacher path still exposes which
                # tokens are clean or spurious.
                crop_to_full_teacher_match_indices = (
                    teacher_foreground_crop_tokens @ teacher_full_image_tokens.transpose(1, 2)
                ).argmax(dim=-1)
                teacher_matched_crop_targets = torch.gather(
                    teacher_full_image_tokens,
                    1,
                    crop_to_full_teacher_match_indices.unsqueeze(-1).expand(-1, -1, feature_dim),
                )

                fp_gp_inter_image_tokens = F.normalize(
                    apply_channel_mask(teacher_model.encode_dense(background_images[:1]), args=args),
                    dim=-1,
                )
                fp_gp_inter_image_clean_mask, sampled_fp_gp_inter_image_tokens = analyze_fp_gp_similarity(
                    teacher_foreground_crop_tokens,
                    fp_gp_inter_image_tokens,
                    thres_sigma=args.fp_gp_sigma,
                    thres_cos=args.fp_gp_cosine_threshold,
                    sampled_token_count=student_register_tokens.shape[1],
                )
                # FP-GP filtering uses two criteria. The inter-image path
                # compares with another image, while the intra-image path
                # compares with unrelated background regions in the same
                # composited crop. These criteria expose the shared unreliable
                # pattern of fixed-pattern/global-proxy phenomena.
                fp_gp_intra_image_clean_mask, sampled_fp_gp_intra_image_tokens = analyze_fp_gp_similarity(
                    teacher_foreground_crop_tokens,
                    teacher_background_tokens,
                    thres_sigma=args.fp_gp_sigma,
                    thres_cos=args.fp_gp_cosine_threshold,
                    sampled_token_count=student_register_tokens.shape[1],
                )

                adaptive_register_sample_count = max(student_register_tokens.shape[1] // 3, 1)
                sampled_register_indices = torch.randperm(
                    student_register_tokens.shape[1],
                    device=student_register_tokens.device,
                )[
                    :adaptive_register_sample_count
                ]
                adaptive_register_clean_mask = analyze_adaptive_spurious_detection(
                    student_register_tokens[:, sampled_register_indices, :],
                    teacher_matched_crop_targets,
                    thres_cos=args.adaptive_spurious_detector_cosine_threshold,
                )

                # NCE supervises tokens that pass all filters.
                pre_ah_clean_mask = (
                    fp_gp_inter_image_clean_mask
                    & fp_gp_intra_image_clean_mask
                    & adaptive_register_clean_mask
                )
                if not args.disable_attention_hijack_filter:
                    attention_hijack_clean_mask = analyze_attention_hijacking(
                        self.teacher_hook_cache,
                        layer_start=args.attention_hijack_layer_start,
                        layer_end=args.attention_hijack_layer_end,
                        thres_sigma=args.attention_hijack_sigma,
                        foreground_mask=foreground_token_mask,
                        rejected_mask=~pre_ah_clean_mask,
                    )
                else:
                    attention_hijack_clean_mask = torch.ones_like(pre_ah_clean_mask, dtype=torch.bool)

                clean_token_mask = attention_hijack_clean_mask & pre_ah_clean_mask
                useful_ratio = (clean_token_mask.sum(-1) / clean_token_mask.shape[1]).mean()
                sampled_absorption_teacher_tokens = torch.cat(
                    [sampled_fp_gp_inter_image_tokens, sampled_fp_gp_intra_image_tokens],
                    dim=1,
                )

            disable_student_teacher_matching = bool(
                getattr(args, "disable_student_teacher_matching", False)
            )
            # Default: use composited-crop tokens for filtering, but supervise
            # with matched full-image teacher tokens to reduce distribution shift.
            # If disabled, use composited-crop teacher tokens directly.
            teacher_target_tokens = (
                teacher_foreground_crop_tokens if disable_student_teacher_matching else teacher_matched_crop_targets
            )
            teacher_negative_pool_tokens = (
                teacher_foreground_crop_tokens if disable_student_teacher_matching else teacher_full_image_tokens
            )
            refinement_loss, mean_alignment, register_absorption_loss, register_uniformity = compute_refinement_loss(
                student_crop_tokens,
                teacher_target_tokens,
                clean_token_mask,
                student_register_tokens,
                sampled_absorption_teacher_tokens,
                repeated_student_full_image_tokens,
                teacher_full_image_tokens,
                teacher_negative_pool_tokens=teacher_negative_pool_tokens,
                teacher_spurious_rematch_pool_tokens=teacher_full_image_tokens,
                args=args,
            )
            total_loss = refinement_loss + 0.1 * register_absorption_loss
            spatial_consistency_loss = total_loss.new_zeros(())

            # SCD starts after a short NCE warmup.
            spatial_consistency_start_stage = 0.1
            if args.train_stage > spatial_consistency_start_stage:
                spatial_source_images = crop_images
                spatial_register_size = max(register_size, 4)
                crop_images_with_registers, crop_register_token_mask = surround_image_with_registers(
                    spatial_source_images,
                    patch_size=model.patch_size,
                    register_size=spatial_register_size,
                    register_fill=normalize_register_fill(args.register_fill),
                )

                with torch.no_grad():
                    register_expanded_crop_tokens = apply_channel_mask(
                        model.encode_dense(crop_images_with_registers),
                        args=args,
                    )
                    register_expanded_crop_tokens = register_expanded_crop_tokens[~crop_register_token_mask].reshape(
                        crop_images_with_registers.shape[0],
                        -1,
                        feature_dim,
                    )
                    register_expanded_crop_tokens = F.normalize(
                        register_expanded_crop_tokens[: batch_size * num_proposals],
                        dim=-1,
                    )

                spatial_consistency_loss = compute_spatial_consistency_loss(
                    student_crop_tokens,
                    register_expanded_crop_tokens,
                )
                total_loss = total_loss + 0.4 * spatial_consistency_loss

            losses = {
                "loss_final": total_loss,
                "nce_loss": refinement_loss,
                "loss_scd": spatial_consistency_loss,
                "loss_cos": mean_alignment,
                "align_reg": register_absorption_loss,
                "uniform_reg": register_uniformity.detach().mean(),
                "useful_ratio": useful_ratio,
            }
            return losses, batch_size
        finally:
            reset_teacher_hook_cache(self.teacher_hook_cache)

    def __call__(self, batch, model, teacher_model, device, cast_dtype, distributed, args):
        return self.forward(batch, model, teacher_model, device, cast_dtype, distributed, args)
