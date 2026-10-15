"""
Adaptive Ratio Dataset
======================
Dynamically adjusts the normal vs wrong-caption sample ratio during training,
based on per-step advantage dead rates.

Usage in training config yaml:
    data:
      custom_cls:
        path: perception/utils/adaptive_dataset.py
        name: AdaptiveRatioDataset
      adaptive_target_ratio: 2.0    # target effective normal:wrong ratio
      adaptive_ema_alpha: 0.1       # EMA smoothing coefficient
"""

import json
import logging
import random
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl import DataProto
from verl.utils.dataset.rl_dataset import RLHFDataset

logger = logging.getLogger(__name__)


class AdaptiveRatioDataset(RLHFDataset):
    """Dataset with dynamic normal/wrong-caption ratio adjustment.

    Monitors advantage dead rates per sample type and adjusts the probability
    of degrading wrong-caption samples to normal ones, ensuring the effective
    (gradient-producing) sample ratio stays at the target.

    Degradation is reversible: it only affects __getitem__ return values,
    never the underlying dataset.
    """

    def __init__(
        self,
        data_files,
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
    ):
        super().__init__(data_files, tokenizer, config, processor, max_samples)
        print("################################################################################################## Hello#")
        # Configuration
        self._target_ratio = config.get("adaptive_target_ratio", 2.0)
        self._ema_alpha = config.get("adaptive_ema_alpha", 0.1)

        # State
        self._degrade_prob = 0.0
        self._ema_normal_alive = 0.5
        self._ema_wrong_alive = 0.5

        # Count original types in dataset
        self._count_types()

        # Metrics buffer (read by trainer after on_batch_end)
        self._last_metrics = {}

        logger.info(
            f"[AdaptiveRatio] Initialized: N_normal={self._n_normal}, N_wrong={self._n_wrong}, "
            f"target_ratio={self._target_ratio}, ema_alpha={self._ema_alpha}"
        )

    def maybe_filter_out_long_prompts(self, dataframe):
        """Override to avoid cloudpickle serializing 'self' inside the doc2len closure.

        Even with num_proc=1, HuggingFace Dataset.filter() may pickle doc2len and its
        captured variables. Since doc2len closes over 'self', and AdaptiveRatioDataset is
        loaded via load_extern_object (non-importable __module__), cloudpickle cannot
        reconstruct the inherited class properly → _build_messages disappears.

        Fix: extract all required state into plain Python objects before building doc2len,
        so 'self' is never captured in the closure.
        """
        if not self.filter_overlong_prompts:
            return dataframe

        # ── Snapshot all needed state as plain objects (no 'self' reference) ──
        tokenizer        = self.tokenizer
        processor        = self.processor
        prompt_key       = self.prompt_key
        image_key        = self.image_key
        video_key        = self.video_key
        image_patch_size = self.image_patch_size
        max_prompt_length = self.max_prompt_length
        apply_chat_template_kwargs = dict(self.apply_chat_template_kwargs)
        tool_schemas     = self.tool_schemas

        if processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_video
            import re as _re
            import traceback as _tb
            from io import BytesIO as _BytesIO
            from PIL import Image as _Image

            def _build_messages_local(doc, _prompt_key, _image_key, _video_key):
                """Standalone version of RLHFDataset._build_messages using only free vars."""
                import re as _re2
                from io import BytesIO as _BytesIO2
                from PIL import Image as _Image2
                messages = list(doc[_prompt_key])  # shallow copy
                images   = list(doc.get(_image_key, []) or [])
                videos   = list(doc.get(_video_key, []) or [])
                image_offset, video_offset = 0, 0
                for message in messages:
                    if not images and not videos:
                        continue
                    content = message.get("content", "")
                    if not isinstance(content, str):
                        continue
                    content_list = []
                    segments = _re2.split("(<image>|<video>)", content)
                    segments = [s for s in segments if s]
                    for segment in segments:
                        if segment == "<image>":
                            image = images[image_offset]
                            if isinstance(image, _Image2.Image):
                                image = image.convert("RGB")
                            elif isinstance(image, dict) and "bytes" in image:
                                image = _Image2.open(_BytesIO2(image["bytes"])).convert("RGB")
                            content_list.append({"type": "image", "image": image})
                            image_offset += 1
                        elif segment == "<video>":
                            content_list.append({"type": "video", "video": videos[video_offset]})
                            video_offset += 1
                        else:
                            content_list.append({"type": "text", "text": segment})
                    message["content"] = content_list
                return messages

            def doc2len(doc) -> int:
                try:
                    messages = _build_messages_local(doc, prompt_key, image_key, video_key)
                    apply_kwargs = dict(apply_chat_template_kwargs)
                    if tool_schemas is not None:
                        apply_kwargs["tools"] = tool_schemas
                    raw_prompt = processor.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=False, **apply_kwargs
                    )
                    images = [
                        process_image(img, image_patch_size=image_patch_size)
                        for img in (doc.get(image_key) or [])
                    ] or None
                    if doc.get(video_key):
                        videos, video_metadata = zip(
                            *[
                                process_video(v, image_patch_size=image_patch_size, return_video_metadata=True)
                                for v in doc[video_key]
                            ],
                            strict=True,
                        )
                        videos_kwargs = {"video_metadata": list(video_metadata), "do_sample_frames": False}
                        videos = list(videos)
                    else:
                        videos, videos_kwargs = None, {}
                    return len(
                        processor(text=[raw_prompt], images=images, videos=videos, videos_kwargs=videos_kwargs)[
                            "input_ids"
                        ][0]
                    )
                except Exception:
                    print("Error processing one of the samples, skipping...")
                    import traceback; traceback.print_exc()
                    return max_prompt_length + 1
        else:
            def doc2len(doc) -> int:
                try:
                    apply_kwargs = dict(apply_chat_template_kwargs)
                    if tool_schemas is not None:
                        apply_kwargs["tools"] = tool_schemas
                    return len(
                        tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True, **apply_kwargs)
                    )
                except Exception:
                    print("Error processing one of the samples, skipping...")
                    import traceback; traceback.print_exc()
                    return max_prompt_length + 1

        # Always run filter in main process (num_proc=1) to avoid any pickling of closures
        dataframe = dataframe.filter(
            lambda doc: doc2len(doc) <= max_prompt_length,
            num_proc=1,
            desc=f"Filtering prompts longer than {max_prompt_length} tokens",
        )
        print(f"filter dataset len: {len(dataframe)}")
        return dataframe

    def _count_types(self):
        """Count original normal (type=0) and wrong-caption (type=1) samples."""
        self._n_normal = 0
        self._n_wrong = 0
        for i in range(len(self.dataframe)):
            row = self.dataframe[i]
            extra = row.get("extra_info", {})
            if isinstance(extra, str):
                try:
                    extra = json.loads(extra)
                except Exception:
                    extra = {}
            t = extra.get("type", 0) if isinstance(extra, dict) else 0
            if t == 0:
                self._n_normal += 1
            elif t == 1:
                self._n_wrong += 1
        logger.info(
            f"[AdaptiveRatio] Dataset type counts: normal={self._n_normal}, wrong={self._n_wrong}"
        )

    def __getitem__(self, item):
        """Return data with probabilistic degradation of wrong->normal samples."""
        row_dict = super().__getitem__(item)

        extra_info = row_dict.get("extra_info", {})
        sample_type = extra_info.get("type", 0) if isinstance(extra_info, dict) else 0

        if sample_type == 1 and self._degrade_prob > 0 and random.random() < self._degrade_prob:
            # Reversible degradation: modify returned values only, not the underlying data
            row_dict["extra_info"] = {**extra_info, "type": 0, "_degraded": True}
            # Remove the last assistant turn (wrong caption)
            if "raw_prompt" in row_dict and isinstance(row_dict["raw_prompt"], list):
                if len(row_dict["raw_prompt"]) > 2 and row_dict["raw_prompt"][-1].get("role") == "assistant":
                    row_dict["raw_prompt"] = row_dict["raw_prompt"][:-1]

        return row_dict

    def on_batch_end(self, batch: DataProto):
        """Called by trainer after each training step.

        Computes dead rates and updates degrade_prob for the next step.
        """
        try:
            metrics = self._compute_dead_rates(batch)
            self._update_degrade_prob(metrics)
            self._last_metrics = metrics
        except Exception as e:
            logger.warning(
                f"[AdaptiveRatio] on_batch_end failed: {e}. "
                f"Keeping current degrade_prob={self._degrade_prob:.4f}"
            )

    def get_metrics(self) -> dict:
        """Return metrics for wandb logging. Called by trainer after on_batch_end."""
        return self._last_metrics

    def _compute_dead_rates(self, batch: DataProto) -> dict:
        """Compute advantage dead rates grouped by sample type."""
        advantages = batch.batch["advantages"]         # (B*n, response_length)
        response_mask = batch.batch["response_mask"]   # (B*n, response_length)
        uids = batch.non_tensor_batch["uid"]           # (B*n,)

        # Per-rollout scalar advantage
        adv_scalar = (advantages * response_mask).sum(-1)  # (B*n,)

        # Get type for each rollout from extra_info
        extra_infos = batch.non_tensor_batch.get("extra_info", np.array([{}] * len(uids)))
        types = []
        for i in range(len(uids)):
            ei = extra_infos[i]
            if isinstance(ei, dict):
                types.append(ei.get("type", 0))
            else:
                types.append(0)

        # Group by uid
        uid_groups: dict[str, dict] = defaultdict(lambda: {"advs": [], "type": 0})
        for i, uid in enumerate(uids):
            uid_key = str(uid)
            uid_groups[uid_key]["advs"].append(adv_scalar[i].item())
            uid_groups[uid_key]["type"] = types[i]

        # Compute dead rates per type
        normal_total, normal_dead = 0, 0
        wrong_total, wrong_dead = 0, 0

        for info in uid_groups.values():
            advs = info["advs"]
            t = info["type"]
            is_dead = np.var(advs) < 1e-8 if len(advs) > 1 else True

            if t == 0:
                normal_total += 1
                normal_dead += int(is_dead)
            elif t == 1:
                wrong_total += 1
                wrong_dead += int(is_dead)

        current_normal_alive = 1.0 - (normal_dead / max(normal_total, 1))
        current_wrong_alive = 1.0 - (wrong_dead / max(wrong_total, 1))

        # EMA smoothing
        self._ema_normal_alive = (
            self._ema_alpha * current_normal_alive
            + (1 - self._ema_alpha) * self._ema_normal_alive
        )
        self._ema_wrong_alive = (
            self._ema_alpha * current_wrong_alive
            + (1 - self._ema_alpha) * self._ema_wrong_alive
        )

        return {
            "actor/adaptive_normal_dead_rate": 1.0 - current_normal_alive,
            "actor/adaptive_wrong_dead_rate": 1.0 - current_wrong_alive,
            "actor/adaptive_ema_normal_alive": self._ema_normal_alive,
            "actor/adaptive_ema_wrong_alive": self._ema_wrong_alive,
            "actor/adaptive_batch_normal_count": normal_total,
            "actor/adaptive_batch_wrong_count": wrong_total,
        }

    def _update_degrade_prob(self, metrics: dict):
        """Solve for degrade probability that achieves target effective ratio.

        Target: effective_normal / effective_wrong = R
        Where:
            effective_normal = (N_0 + N_1 * p) * alpha_0
            effective_wrong  = N_1 * (1 - p) * alpha_1

        Solving for p:
            p = (R * alpha_1 - (N_0/N_1) * alpha_0) / (alpha_0 + R * alpha_1)
        """
        alpha_0 = self._ema_normal_alive
        alpha_1 = self._ema_wrong_alive
        R = self._target_ratio
        N_0 = self._n_normal
        N_1 = self._n_wrong

        if N_1 == 0:
            self._degrade_prob = 0.0
            metrics["actor/degrade_prob"] = 0.0
            return

        r = N_0 / N_1
        numerator = R * alpha_1 - r * alpha_0
        denominator = alpha_0 + R * alpha_1

        if denominator < 1e-8:
            p = 0.0
        else:
            p = numerator / denominator

        p = max(0.0, min(1.0, p))

        old_p = self._degrade_prob
        self._degrade_prob = p
        metrics["actor/degrade_prob"] = self._degrade_prob

        # Compute effective counts for logging
        effective_normal = (N_0 + N_1 * p) * alpha_0
        effective_wrong = N_1 * (1 - p) * alpha_1
        actual_ratio = effective_normal / max(effective_wrong, 1e-8)
        metrics["actor/adaptive_effective_ratio"] = actual_ratio

        if abs(old_p - p) > 0.01:
            logger.info(
                f"[AdaptiveRatio] degrade_prob: {old_p:.4f} -> {p:.4f} | "
                f"ema_alive: normal={self._ema_normal_alive:.3f}, wrong={self._ema_wrong_alive:.3f} | "
                f"effective_ratio={actual_ratio:.2f} (target={R})"
            )


# ── Pickle / cloudpickle compatibility ──────────────────────────────────────
# AdaptiveRatioDataset is loaded dynamically via load_extern_object, which sets
# __module__ to a non-importable name. When cloudpickle serializes instances
# across Ray / HuggingFace worker boundaries it stores (__module__, __qualname__)
# and tries to re-import on the other side. If __module__ is wrong, the worker
# reconstructs a bare object with no inherited methods → AttributeError on any
# parent-class method (e.g. _build_messages).
#
# Fix: register this module under its real importable path so every worker can
# do `from perception.utils.adaptive_dataset import AdaptiveRatioDataset`
# and get the fully-inherited class back.
import sys as _sys

_IMPORTABLE = "perception.utils.adaptive_dataset"
_this_module = _sys.modules.get(__name__)
if _this_module is not None and _IMPORTABLE not in _sys.modules:
    _sys.modules[_IMPORTABLE] = _this_module

# Patch the class's __module__ so pickle/cloudpickle uses the importable path.
AdaptiveRatioDataset.__module__ = _IMPORTABLE
