#!/usr/bin/env python3
"""Archive selected HF checkpoints; preview is offline, uploads require --execute.

The private source-to-neutral-ID mapping is read from an ignored local manifest.
Only explicitly listed model artifacts can be uploaded. Source weights are never
modified or deleted. See upload_selected_checkpoints.md for usage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
INDEX = "model.safetensors.index.json"
SUPPORTED_BASES = {
    "Qwen/Qwen2.5-VL-7B-Instruct": "qwen2_5_vl",
    "Qwen/Qwen2-VL-7B-Instruct": "qwen2_vl",
    "Qwen/Qwen3-VL-4B-Instruct": "qwen3_vl",
    "Qwen/Qwen3-VL-8B-Instruct": "qwen3_vl",
}
LICENSE_SHA256 = "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
# These official model cards declare Apache-2.0, but their model repositories
# may omit a standalone LICENSE. Copies below came from Qwen's code repositories.
LICENSE_FALLBACKS = {
    "Qwen/Qwen2.5-VL-7B-Instruct": (
        "Qwen2.5-VL-LICENSE.txt", "https://github.com/QwenLM/Qwen2.5-VL/blob/main/LICENSE"
    ),
    "Qwen/Qwen3-VL-4B-Instruct": (
        "Qwen3-VL-LICENSE.txt", "https://github.com/QwenLM/Qwen3-VL/blob/main/LICENSE"
    ),
    "Qwen/Qwen3-VL-8B-Instruct": (
        "Qwen3-VL-LICENSE.txt", "https://github.com/QwenLM/Qwen3-VL/blob/main/LICENSE"
    ),
}
MODEL_FILES = {
    "config.json", "generation_config.json", "tokenizer_config.json",
    "tokenizer.json", "vocab.json", "merges.txt", "added_tokens.json",
    "special_tokens_map.json", "preprocessor_config.json",
    "processor_config.json", "video_preprocessor_config.json",
    "chat_template.jinja", "chat_template.json", INDEX,
}
# Vocabulary/token contents must remain byte-for-byte unchanged.
CONFIG_FILES = MODEL_FILES - {
    "tokenizer.json", "vocab.json", "merges.txt", "added_tokens.json",
    "special_tokens_map.json", "chat_template.jinja", "chat_template.json", INDEX,
}
DROP_KEYS = {
    "_name_or_path", "name_or_path", "_commit_hash", "cache_dir", "local_dir",
    "model_path", "checkpoint_path", "source_path", "output_dir", "logging_dir",
    "experiment_name", "project_name", "run_name", "wandb", "wandb_run_id",
    "api_key", "api-key", "access_token", "hf_token", "password",
    "secret_key", "authorization", "azure_endpoint", "api_url", "base_url",
}
CREDENTIAL = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{16,}|hf_[A-Za-z0-9]{16,}|"
    r"ghp_[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16})\b"
)
INTERNAL = re.compile(
    r"(?:/(?:mnt|home|root|workspace)/|hdfs://|file://|"
    r"\b10\.(?:\d{1,3}\.){2}\d{1,3}\b|"
    r"(?:byteintl|tiktok-row|byted)\.(?:net|org))", re.I
)


class ArchiveError(RuntimeError):
    pass


@dataclass
class Model:
    id: str
    visibility: str
    source: Path
    source_relative: str
    base_model: str
    files: list[Path]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
    # Keep mtime stable on reruns so the Hub upload cache can be reused.
    if not path.exists() or path.read_text() != text:
        path.write_text(text)


def load_models(manifest: Path, source_root: Path) -> list[Model]:
    data = json.loads(manifest.read_text())
    if data.get("version") != 1:
        raise ArchiveError("Unsupported manifest version")
    models = []
    ids, sources = set(), set()
    for row in data["models"]:
        visibility, model_id = row["visibility"], row["id"]
        if visibility not in {"private", "public"}:
            raise ArchiveError("Unknown visibility in manifest")
        expected_id = r"main" if visibility == "private" else r"model-\d{3}"
        if not re.fullmatch(expected_id, model_id) or model_id in ids:
            raise ArchiveError("Invalid or duplicate neutral model ID")
        relative = PurePosixPath(row["source"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ArchiveError("Model source must be a relative path within source-root")
        source = (source_root / str(relative)).resolve(strict=True)
        if not source.is_relative_to(source_root.resolve()) or source in sources:
            raise ArchiveError("Model sources must be distinct directories within source-root")
        if row["base_model"] not in SUPPORTED_BASES:
            raise ArchiveError("Unsupported base model; review its loading files and license before adding it")
        config = json.loads((source / "config.json").read_text())
        if config.get("model_type") != SUPPORTED_BASES[row["base_model"]]:
            raise ArchiveError(f"Base model architecture mismatch for {model_id}")
        index = json.loads((source / INDEX).read_text())
        shards = sorted(set(index["weight_map"].values()))
        if not shards:
            raise ArchiveError(f"Empty weight index: {model_id}")
        for name in shards:
            if Path(name).name != name or not re.fullmatch(r"[A-Za-z0-9_.-]+\.safetensors", name):
                raise ArchiveError(f"Unsafe weight filename: {model_id}")
        required = {"config.json", "tokenizer_config.json", "tokenizer.json", "preprocessor_config.json", INDEX}
        names = required | set(shards)
        missing = sorted(name for name in names if not (source / name).is_file())
        if missing:
            raise ArchiveError(f"Missing files in {model_id}: {', '.join(missing)}")
        files = [source / name for name in sorted(MODEL_FILES | set(shards)) if (source / name).is_file()]
        ids.add(model_id)
        sources.add(source)
        models.append(Model(model_id, visibility, source, str(relative), row["base_model"], files))
    if sum(m.visibility == "private" for m in models) != 1:
        raise ArchiveError("Exactly one main private checkpoint is required")
    if not any(m.visibility == "public" for m in models):
        raise ArchiveError("No public checkpoints selected")
    return models


def sanitize_config(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: sanitize_config(item) for key, item in value.items() if key.lower() not in DROP_KEYS}
    if isinstance(value, list):
        return [sanitize_config(item) for item in value]
    return value


def check_provenance_text(text: str, model: Model, forbidden_names: list[str]) -> None:
    # Never print matched text: a failed check could contain an actual credential.
    if CREDENTIAL.search(text) or INTERNAL.search(text):
        raise ArchiveError(f"Unresolved credential/internal reference in metadata for {model.id}")
    if model.visibility == "public" and (
        re.search(r"\b(?:MIRA|NeurIPS|rebuttal)\b", text, re.I)
        or any(name in text for name in forbidden_names)
    ):
        raise ArchiveError(f"Unresolved experiment identifier in metadata for {model.id}")


def check_weight_header(path: Path, model: Model, forbidden_names: list[str]) -> None:
    with path.open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ArchiveError(f"Truncated safetensors file in {model.id}")
        length = struct.unpack("<Q", raw)[0]
        if length <= 0 or length > 100_000_000:
            raise ArchiveError(f"Invalid safetensors header in {model.id}")
        header = json.loads(stream.read(length))
    check_provenance_text(json.dumps(header), model, forbidden_names)
    # Reject truncated payloads without loading the tensor data.
    payload_size = path.stat().st_size - 8 - length
    offsets = [v.get("data_offsets", []) for k, v in header.items() if k != "__metadata__"]
    if not offsets or any(len(v) != 2 or not 0 <= v[0] <= v[1] <= payload_size for v in offsets):
        raise ArchiveError(f"Invalid tensor offsets in {model.id}")


def model_card(model: Model) -> str:
    return (
        "---\nlicense: apache-2.0\nlibrary_name: transformers\n"
        f"base_model: {model.base_model}\npipeline_tag: image-text-to-text\n---\n\n"
        f"# {model.id}\n\n"
        f"Fine-tuned vision-language checkpoint derived from [{model.base_model}]"
        f"(https://huggingface.co/{model.base_model}).\n\n"
        "The model weights were fine-tuned from the base model. Loading configuration "
        "metadata was cleaned of local provenance fields; tokenizer content and archived "
        "weight bytes are preserved. This is a derivative checkpoint, not an official "
        "upstream release. See LICENSE and any accompanying upstream notices.\n\n"
        "Load this directory with compatible Transformers or vLLM tooling. "
        "For a model in a collection repository, pass this directory name as "
        "`subfolder` when using Transformers. Generation quality is not guaranteed.\n"
    )


def attach_license(model: Model, dest: Path, token: str | None) -> list[Path]:
    from huggingface_hub import hf_hub_download, list_repo_files

    # Fetch upstream legal files, not its full model or model card.
    names = list_repo_files(model.base_model, token=token)
    legal = [n for n in names if "/" not in n and re.fullmatch(r"(?:LICEN[CS]E|NOTICE|COPYING)(?:\.[\w.-]+)?", n, re.I)]
    has_license = any(re.fullmatch(r"(?:LICEN[CS]E|COPYING)(?:\.[\w.-]+)?", n, re.I) for n in legal)
    fallback = None
    if not has_license:
        spec = LICENSE_FALLBACKS.get(model.base_model)
        if spec is None:
            raise ArchiveError(f"Upstream license missing and no verified fallback exists for {model.id}")
        readme = Path(hf_hub_download(model.base_model, "README.md", token=token)).read_text(encoding="utf-8-sig")
        frontmatter = re.match(r"\s*---\s*\n(.*?)\n---(?:\s|$)", readme, re.S)
        declaration = re.search(
            r"^license:\s*['\"]?apache-2\.0['\"]?\s*(?:#.*)?$",
            frontmatter.group(1), re.M | re.I,
        ) if frontmatter else None
        if not declaration:
            raise ArchiveError(f"Cannot confirm Apache-2.0 in the upstream model card for {model.id}")
        filename, license_url = spec
        fallback = Path(__file__).resolve().parent / "checkpoint_archive_licenses" / filename
        if not fallback.is_file() or hashlib.sha256(fallback.read_bytes()).hexdigest() != LICENSE_SHA256:
            raise ArchiveError("Verified license copy is missing or changed; restore scripts/checkpoint_archive_licenses")
        print(f"  {model.id}: using official Apache-2.0 copy (upstream model card verified)", flush=True)
    result = []
    for name in legal:
        source = Path(hf_hub_download(model.base_model, name, token=token))
        target = dest / name
        if not target.exists() or target.read_bytes() != source.read_bytes():
            shutil.copyfile(source, target)
        result.append(target)
    if fallback is not None:
        target = dest / "LICENSE"
        if not target.exists() or target.read_bytes() != fallback.read_bytes():
            shutil.copyfile(fallback, target)
        provenance = dest / "UPSTREAM_LICENSE.md"
        text = (
            f"Base model: [{model.base_model}](https://huggingface.co/{model.base_model}).\n\n"
            "The upstream model card declares `license: apache-2.0`, but its model "
            "repository does not include a standalone license file. The accompanying "
            f"LICENSE is copied verbatim from [the official Qwen license]({license_url}).\n\n"
            f"License text SHA-256: `{LICENSE_SHA256}`.\n"
        )
        if not provenance.exists() or provenance.read_text() != text:
            provenance.write_text(text)
        result.extend([target, provenance])
    return result


def sha256(path: Path, cache: dict) -> str:
    info = path.stat()
    signature = [str(path.resolve()), info.st_size, info.st_mtime_ns]
    old = cache.get(str(path))
    if old and old["signature"] == signature:
        return old["sha256"]
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if path.stat().st_size != info.st_size or path.stat().st_mtime_ns != info.st_mtime_ns:
        raise ArchiveError("Source changed during checksum calculation")
    value = digest.hexdigest()
    cache[str(path)] = {"signature": signature, "sha256": value}
    return value


def prepare(models: list[Model], work: Path, token: str | None, private_id: str, public_id: str) -> dict:
    stages = {name: work / name for name in ("private", "public")}
    for stage in stages.values():
        stage.mkdir(parents=True, exist_ok=True)
    cache_path = work / "checksums.local.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    forbidden = [m.source.name for m in models]
    expected: dict[str, dict[str, dict]] = {"private": {}, "public": {}}
    mapping = []
    for model in models:
        dest = stages[model.visibility] / (model.id if model.visibility == "public" else "")
        dest.mkdir(parents=True, exist_ok=True)
        output = []
        print(f"Preparing {model.visibility}/{model.id}", flush=True)
        for source in model.files:
            target = dest / source.name
            if source.suffix == ".safetensors":
                check_weight_header(source, model, forbidden)
                if target.is_symlink():
                    if target.resolve() != source.resolve():
                        raise ArchiveError("Existing staging symlink points to a different checkpoint")
                elif target.exists():
                    raise ArchiveError("Existing staging weight is not the expected symlink; use a fresh work-dir")
                else:
                    target.symlink_to(source.resolve())
            elif source.name in CONFIG_FILES:
                cleaned = sanitize_config(json.loads(source.read_text()))
                check_provenance_text(json.dumps(cleaned), model, forbidden)
                write_json(target, cleaned)
            else:
                if source.name in {INDEX, "chat_template.jinja", "chat_template.json"}:
                    check_provenance_text(source.read_text(), model, forbidden)
                if not target.exists() or target.read_bytes() != source.read_bytes():
                    shutil.copyfile(source, target)
            output.append(target)
        card = dest / "README.md"
        text = model_card(model)
        if not card.exists() or card.read_text() != text:
            card.write_text(text)
        output.extend([card, *attach_license(model, dest, token)])
        checksums = {}
        for path in sorted(output):
            # The large weights are hashed once here; reruns reuse the local cache.
            print(f"  checksum {path.name}", flush=True)
            checksums[path.name] = sha256(path, cache)
            write_json(cache_path, cache)
        sums = dest / "SHA256SUMS"
        text = "".join(f"{value}  {name}\n" for name, value in sorted(checksums.items()))
        if not sums.exists() or sums.read_text() != text:
            sums.write_text(text)
        output.append(sums)
        for path in output:
            name = path.relative_to(stages[model.visibility]).as_posix()
            expected[model.visibility][name] = {"size": path.stat().st_size, "sha256": sha256(path, cache)}
        mapping.append({"id": model.id, "visibility": model.visibility,
                        "source": model.source_relative, "base_model": model.base_model})
    public_card = stages["public"] / "README.md"
    text = "---\nlicense: apache-2.0\n---\n\n# Vision-language checkpoint collection\n\n"
    text += "Each subdirectory contains a separate fine-tuned checkpoint, loading files, "
    text += "upstream licensing information, and SHA256SUMS. See its README for the base model.\n\n"
    text += "| Directory | Base model |\n|---|---|\n"
    text += "".join(f"| {m.id} | {m.base_model} |\n" for m in models if m.visibility == "public")
    if not public_card.exists() or public_card.read_text() != text:
        public_card.write_text(text)
    private_manifest = stages["private"] / "archive_manifest.private.json"
    write_json(private_manifest, {"version": 1, "private_repo": private_id, "public_repo": public_id, "models": mapping})
    for visibility, path in [("public", public_card), ("private", private_manifest)]:
        expected[visibility][path.name] = {"size": path.stat().st_size, "sha256": sha256(path, cache)}
    write_json(cache_path, cache)
    write_json(work / "expected.local.json", expected)
    return {"stages": stages, "expected": expected}


def ensure_repo(api, repo_id: str, private: bool, expected: dict) -> None:
    from huggingface_hub.errors import RepositoryNotFoundError

    try:
        info = api.model_info(repo_id)
    except RepositoryNotFoundError:
        api.create_repo(repo_id, private=private, repo_type="model", exist_ok=True)
        info = api.model_info(repo_id)
    if bool(info.private) != private:
        raise ArchiveError(f"Visibility mismatch for {repo_id}; refusing to change an existing repository")
    extra = {f.rfilename for f in info.siblings or []} - set(expected) - {".gitattributes"}
    if extra:
        raise ArchiveError(f"Existing repository {repo_id} contains unexpected files; choose an empty repository")


def verify_remote(api, repo_id: str, expected: dict, stage: Path, private: bool) -> None:
    info = api.model_info(repo_id, files_metadata=True)
    if bool(info.private) != private:
        raise ArchiveError(f"Visibility mismatch after upload for {repo_id}")
    remote = {f.rfilename: f for f in info.siblings or []}
    for name, wanted in expected.items():
        found = remote.get(name)
        if found is None or found.size != wanted["size"]:
            raise ArchiveError(f"Missing file or size mismatch in {repo_id}: {name}")
        lfs = getattr(found, "lfs", None)
        remote_sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        if remote_sha:
            if remote_sha != wanted["sha256"]:
                raise ArchiveError(f"SHA-256 mismatch in {repo_id}: {name}")
        else:
            # Small Git-backed files expose Git blob SHA-1 rather than LFS SHA-256.
            if wanted["size"] > 20 * 1024 * 1024:
                raise ArchiveError(f"Remote hash unavailable for large file in {repo_id}: {name}")
            contents = (stage / name).read_bytes()
            digest = hashlib.sha1(f"blob {len(contents)}\0".encode() + contents).hexdigest()
            if digest != found.blob_id:
                raise ArchiveError(f"Git blob hash mismatch in {repo_id}: {name}")
    print(f"Verified {repo_id}: visibility, {len(expected)} file sizes and content hashes", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", required=True, help="Hugging Face user or organization")
    parser.add_argument("--private-repo", default="checkpoint-main-private", help="Repo name within namespace")
    parser.add_argument("--public-repo", default="vl-checkpoint-archive", help="Repo name within namespace")
    parser.add_argument("--manifest", type=Path, default=ROOT / ".checkpoint_archive/manifest.private.json")
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/hf-checkpoint-archive"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--execute", action="store_true", help="Prepare, hash, upload and verify; otherwise offline preview only")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    for value in [args.namespace, args.private_repo, args.public_repo]:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
            raise ArchiveError("Namespace and repository names must be single HF name components")
    if args.private_repo == args.public_repo or args.workers < 1:
        raise ArchiveError("Use different public/private repository names and a positive worker count")
    if re.search(r"mira|neurips|rebuttal|paper", args.public_repo, re.I):
        raise ArchiveError("Use a neutral public repository name")
    models = load_models(args.manifest, args.source_root)
    private_id = f"{args.namespace}/{args.private_repo}"
    public_id = f"{args.namespace}/{args.public_repo}"
    for visibility, repo_id in [("private", private_id), ("public", public_id)]:
        chosen = [m for m in models if m.visibility == visibility]
        total = sum(f.stat().st_size for m in chosen for f in m.files)
        print(f"{visibility.upper()} {repo_id}: {len(chosen)} models, {total / 1024**3:.2f} GiB")
        for model in chosen:
            print(f"  {model.id}: {model.source_relative}")
    if not args.execute:
        print("Offline preview only. No network calls, staging or uploads. Add --execute to run.")
        return 0
    from huggingface_hub import HfApi, get_token

    token = get_token()
    if not token:
        raise ArchiveError("Authenticate with `hf auth login` or set HF_TOKEN before --execute")
    api = HfApi(token=token)
    os.umask(0o077)
    work = args.work_dir / args.namespace / (args.private_repo + "__" + args.public_repo)
    work.mkdir(parents=True, exist_ok=True, mode=0o700)
    plan = prepare(models, work, token, private_id, public_id)
    # Validate both targets before uploading either. Never change visibility.
    for visibility, repo_id in [("private", private_id), ("public", public_id)]:
        ensure_repo(api, repo_id, visibility == "private", plan["expected"][visibility])
    for visibility, repo_id in [("private", private_id), ("public", public_id)]:
        expected = plan["expected"][visibility]
        stage = plan["stages"][visibility]
        # Recheck immediately before each upload, including the main private repo.
        ensure_repo(api, repo_id, visibility == "private", expected)
        print(f"Uploading {visibility}: {repo_id}", flush=True)
        api.upload_large_folder(repo_id=repo_id, repo_type="model", folder_path=stage,
                                private=visibility == "private", num_workers=args.workers,
                                allow_patterns=sorted(expected), print_report_every=60)
        verify_remote(api, repo_id, expected, stage, visibility == "private")
    write_json(work / "verified.local.json", {"private_repo": private_id, "public_repo": public_id,
                                             "status": "verified", "models": len(models)})
    print("Both archives verified. Original files have not been modified or deleted.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ArchiveError, OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
