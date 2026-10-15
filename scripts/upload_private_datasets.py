#!/usr/bin/env python3
"""Upload the selected research datasets to one private Hugging Face dataset repo.

The file list is deliberately explicit. Preview is offline; network access and
uploads require --execute. Source files are read directly and never changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_DIR = ".dataset_archive"


@dataclass(frozen=True)
class DatasetFile:
    archive_path: str
    source_path: str
    role: str


# Main training data, controls/ablation inputs, and the additional Qwen3-VL-8B
# training pair. These are only the parquet files referenced by the paper's
# training scripts or the explicitly selected rebuttal runs.
DATA_FILES = (
    DatasetFile("training/main_augmented/train.parquet",
                "_preprocessed_dataset/0416_agumented/train.parquet",
                "Main augmented training set used by the main run and several ablations."),
    DatasetFile("training/vanilla_grpo_control/train.parquet",
                "_preprocessed_dataset/dataset_20K_0423_VANILLA/standard_NONE_20260423_024929/train.parquet",
                "Vanilla GRPO control training set."),
    DatasetFile("training/gpt54mini_augmented/train.parquet",
                "_preprocessed_dataset/0424_dataset_gpt_5_4_mini_18k_augmented/train.parquet",
                "GPT-5.4-mini augmented training set."),
    DatasetFile("training/gpt54_augmented/train.parquet",
                "_preprocessed_dataset/gpt_5_4_augmented/train.parquet",
                "GPT-5.4 augmented training set used for the reported expert-model result."),
    DatasetFile("rebuttal/qwen3_vl_8b_augmented/train.parquet",
                "_preprocessed_dataset/virl39k_mmk12_gpt54_augmented_qwen3-vl-8b_20260727_122328/train.parquet",
                "Qwen3-VL-8B augmented training set."),
    DatasetFile("rebuttal/qwen3_vl_8b_augmented/test.parquet",
                "_preprocessed_dataset/virl39k_mmk12_gpt54_augmented_qwen3-vl-8b_20260727_122328/test.parquet",
                "Companion split stored with the Qwen3-VL-8B augmented set; the launch script configures validation separately."),
    DatasetFile("rebuttal/qwen3_vl_8b_vanilla/train.parquet",
                "_preprocessed_dataset/virl39k_mmk12_gpt54_vanilla/train.parquet",
                "Qwen3-VL-8B vanilla GRPO control training set."),
    DatasetFile("ablations/data_ratio/ratio_000/train.parquet",
                "_preprocessed_dataset/ablation_ratio_3/ratio_000/train.parquet",
                "Ratio-ablation parquet; semantic ratio was not independently verified."),
    DatasetFile("ablations/data_ratio/ratio_030/train.parquet",
                "_preprocessed_dataset/ablation_ratio_3/ratio_030/train.parquet",
                "Ratio-ablation parquet; semantic ratio was not independently verified."),
    DatasetFile("ablations/data_ratio/ratio_050/train.parquet",
                "_preprocessed_dataset/ablation_ratio_3/ratio_050/train.parquet",
                "Ratio-ablation parquet; semantic ratio was not independently verified."),
    DatasetFile("ablations/data_ratio/ratio_070/train.parquet",
                "_preprocessed_dataset/ablation_ratio_3/ratio_070/train.parquet",
                "Ratio-ablation parquet; semantic ratio was not independently verified."),
)


class ArchiveError(RuntimeError):
    pass


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
    if not path.exists() or path.read_text(encoding="utf-8") != content:
        path.write_text(content, encoding="utf-8")


def resolve_files(source_root: Path) -> list[tuple[DatasetFile, Path]]:
    root = source_root.resolve(strict=True)
    resolved = []
    seen_archive_paths = set()
    for item in DATA_FILES:
        archive_path = PurePosixPath(item.archive_path)
        source_relative = PurePosixPath(item.source_path)
        if (archive_path.is_absolute() or ".." in archive_path.parts or
                source_relative.is_absolute() or ".." in source_relative.parts):
            raise ArchiveError("Dataset paths must be relative and cannot traverse directories")
        if item.archive_path in seen_archive_paths:
            raise ArchiveError("Duplicate destination path in the explicit dataset allowlist")
        path = root / item.source_path
        if path.is_symlink():
            raise ArchiveError(f"Refusing symlinked dataset file: {item.source_path}")
        try:
            actual = path.resolve(strict=True)
        except FileNotFoundError as exc:
            raise ArchiveError(f"Required dataset file is missing: {item.source_path}") from exc
        if not actual.is_relative_to(root) or not actual.is_file() or actual.suffix != ".parquet":
            raise ArchiveError(f"Dataset source is not a regular parquet under source-root: {item.source_path}")
        seen_archive_paths.add(item.archive_path)
        resolved.append((item, actual))
    return resolved


def sha256(path: Path, cache: dict[str, dict]) -> str:
    stat = path.stat()
    signature = [str(path.resolve()), stat.st_size, stat.st_mtime_ns]
    old = cache.get(str(path))
    if old and old.get("signature") == signature:
        return old["sha256"]
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if path.stat().st_size != stat.st_size or path.stat().st_mtime_ns != stat.st_mtime_ns:
        raise ArchiveError("Source dataset changed during checksum calculation")
    value = digest.hexdigest()
    cache[str(path)] = {"signature": signature, "sha256": value}
    return value


def make_readme(entries: list[dict[str, Any]]) -> str:
    lines = [
        "# Private research data archive",
        "",
        "This private repository stores selected training and ablation parquet files for continuity and reproducibility.",
        "It does not contain experiment result JSON, rollout outputs, logs, caches, checkpoints, or evaluation predictions.",
        "",
        "The training data is based on ViRL-39K and MMK-12 and includes derived visual-context augmentation. "
        "Confirm the source datasets' terms before making any part of this archive public or redistributing it.",
        "",
        "The ratio-ablation files are preserved under their local ratio labels. Their accompanying `stats.json` files "
        "were absent from the source workspace, so the intended ratio semantics have not been verified here.",
        "",
        "## Files",
        "",
        "| Archive path | Description | Size (bytes) | SHA-256 |",
        "|---|---|---:|---|",
    ]
    for entry in entries:
        desc = entry["role"].replace("|", "\\|")
        lines.append(f"| `{entry['archive_path']}` | {desc} | {entry['size']} | `{entry['sha256']}` |")
    lines.append("")
    lines.append("`ARCHIVE_MANIFEST.json` records the original workspace-relative source paths. Keep this repository private.")
    lines.append("")
    return "\n".join(lines)


def build_archive_metadata(entries: list[dict[str, Any]], repo_id: str) -> dict[str, str]:
    manifest = {
        "version": 1,
        "repo_id": repo_id,
        "visibility": "private",
        "files": entries,
        "excluded": ["evaluation results", "rollouts", "logs", "caches", "checkpoints", "source code"],
        "notes": [
            "The ratio-ablation parquet semantics were not verified because stats.json files were absent.",
            "Source dataset redistribution terms must be checked before any public release.",
        ],
    }
    manifest_text = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    readme = make_readme(entries)
    sums = "".join(f"{entry['sha256']}  {entry['archive_path']}\n" for entry in entries)
    return {
        "ARCHIVE_MANIFEST.json": manifest_text,
        "README.md": readme,
        "SHA256SUMS": sums,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="Hugging Face namespace/repository")
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/hf-private-dataset-archive"))
    parser.add_argument("--execute", action="store_true", help="Create/check the private repo and upload")
    return parser.parse_args(argv)


def ensure_private_dataset_repo(api, repo_id: str, expected_paths: set[str]) -> None:
    from huggingface_hub.errors import RepositoryNotFoundError

    try:
        info = api.repo_info(repo_id, repo_type="dataset")
    except RepositoryNotFoundError:
        api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
        info = api.repo_info(repo_id, repo_type="dataset")
    if not bool(info.private):
        raise ArchiveError(f"Repository {repo_id} is not private; refusing to upload or change visibility")
    actual = {file.rfilename for file in (info.siblings or [])}
    unexpected = actual - expected_paths - {".gitattributes"}
    if unexpected:
        raise ArchiveError(
            f"Repository {repo_id} already contains unrelated files; use a new empty private dataset repository"
        )


def git_blob_sha1(path: Path) -> str:
    size = path.stat().st_size
    digest = hashlib.sha1(f"blob {size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_remote(api, repo_id: str, expected: dict[str, dict[str, Any]]) -> None:
    info = api.repo_info(repo_id, repo_type="dataset", files_metadata=True)
    if not bool(info.private):
        raise ArchiveError(f"Visibility check failed for {repo_id}")
    remote = {file.rfilename: file for file in (info.siblings or [])}
    for name, wanted in expected.items():
        found = remote.get(name)
        if found is None or found.size != wanted["size"]:
            raise ArchiveError(f"Remote file missing or size mismatch: {name}")
        lfs = getattr(found, "lfs", None)
        remote_sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        if remote_sha and remote_sha != wanted["sha256"]:
            raise ArchiveError(f"Remote SHA-256 mismatch: {name}")
        if not remote_sha:
            if wanted["size"] > 20 * 1024 * 1024:
                raise ArchiveError(f"Hub did not expose a verifiable LFS SHA-256 for large file: {name}")
            if git_blob_sha1(wanted["path"]) != found.blob_id:
                raise ArchiveError(f"Remote Git blob hash mismatch: {name}")
    print(f"Verified private visibility, {len(expected)} files and remote sizes/hashes: {repo_id}", flush=True)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", args.repo_id):
        raise ArchiveError("--repo-id must be a Hugging Face namespace/repository")
    entries_and_paths = resolve_files(args.source_root)
    total = sum(path.stat().st_size for _, path in entries_and_paths)
    print(f"Private dataset archive target: {args.repo_id}")
    print(f"Explicit parquet allowlist: {len(entries_and_paths)} files, {total / 1024**3:.2f} GiB")
    for item, path in entries_and_paths:
        print(f"  {path.stat().st_size / 1024**2:8.1f} MiB  {item.archive_path}  <- {item.source_path}")
    if not args.execute:
        print("Offline preview only. No token, network, staging, or upload was used. Add --execute to run.")
        return 0

    from huggingface_hub import HfApi, get_token

    token = get_token()
    if not token:
        raise ArchiveError("Authenticate first with `hf auth login` or set HF_TOKEN")
    os.umask(0o077)
    work = args.work_dir / args.repo_id.replace("/", "__")
    work.mkdir(parents=True, exist_ok=True, mode=0o700)
    cache_file = work / "checksums.local.json"
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}

    entries = []
    for item, path in entries_and_paths:
        print(f"Hashing {item.archive_path}", flush=True)
        entries.append({
            "archive_path": item.archive_path,
            "source_path": item.source_path,
            "role": item.role,
            "size": path.stat().st_size,
            "sha256": sha256(path, cache),
        })
        write_json(cache_file, cache)

    metadata_text = build_archive_metadata(entries, args.repo_id)
    metadata_files = {}
    for name, content in metadata_text.items():
        destination = work / name
        if not destination.exists() or destination.read_text(encoding="utf-8") != content:
            destination.write_text(content, encoding="utf-8")
        metadata_files[name] = destination

    expected = {
        entry["archive_path"]: {
            "size": entry["size"], "sha256": entry["sha256"],
            "path": source_path,
        }
        for entry, (_, source_path) in zip(entries, entries_and_paths)
    }
    expected.update({
        name: {"size": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "path": path}
        for name, path in metadata_files.items()
    })

    api = HfApi(token=token)
    ensure_private_dataset_repo(api, args.repo_id, set(expected))
    api.create_repo(args.repo_id, repo_type="dataset", private=True, exist_ok=True)
    ensure_private_dataset_repo(api, args.repo_id, set(expected))

    # Upload source parquet files directly; no second local 16+ GiB staging copy.
    # Rerunning the same command safely retries/replaces only these allowlisted paths.
    for item, path in entries_and_paths:
        print(f"Uploading {item.archive_path}", flush=True)
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=item.archive_path,
            repo_id=args.repo_id,
            repo_type="dataset",
            token=token,
            commit_message=f"Archive {item.archive_path}",
        )
    for name, path in metadata_files.items():
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=name,
            repo_id=args.repo_id,
            repo_type="dataset",
            token=token,
            commit_message="Add private dataset archive manifest",
        )
    ensure_private_dataset_repo(api, args.repo_id, set(expected))
    verify_remote(api, args.repo_id, expected)
    write_json(work / "verified.local.json", {
        "repo_id": args.repo_id,
        "visibility": "private",
        "status": "verified",
        "files": len(expected),
        "parquet_files": len(entries),
    })
    print("Upload and remote checks completed. Source datasets were not modified or deleted.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ArchiveError, OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
