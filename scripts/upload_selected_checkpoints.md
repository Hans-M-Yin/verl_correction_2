# Upload selected checkpoints

This script uploads one main checkpoint to a **private model repository** and
the other selected checkpoints to a **public model repository**, under neutral
`model-001` … directory names. The public collection identifies each base model
and includes its upstream license; it contains no experiment names or results.

The local selection is `.checkpoint_archive/manifest.private.json`. This file is
ignored by Git and contains the original experiment names. The public/private
mapping is also uploaded as `archive_manifest.private.json` **only in the private
repository**, so it remains recoverable after losing access to this machine.
Keep that repository private. Neutral public names do not provide anonymity or
restrict downloading and redistribution.

The current selection has 13 models: one private and twelve public. The excluded
GPT-5.4 step-360 model is absent; the GPT-5.4-mini step-240 model remains selected.
The manifest can be edited before running. Exactly one model must be private.

Run from the project root (replace `YOUR_HF_USERNAME`):

```bash
# Offline preview: checks the local manifest/files, makes no network calls.
python scripts/upload_selected_checkpoints.py --namespace YOUR_HF_USERNAME

# Authenticate using a token that can create/write both repositories.
hf auth login

# Prepare upload copies, compute checksums, upload private first, then public.
python scripts/upload_selected_checkpoints.py \
  --namespace YOUR_HF_USERNAME \
  --private-repo checkpoint-main-private \
  --public-repo vl-checkpoint-archive \
  --execute
```

Uses the existing `huggingface_hub` installation (checked with 0.36.2 and
`hf_xet` 1.4.2). If running elsewhere, install `huggingface_hub>=0.36.2,<1` and
`hf_xet` first. Tokens are read from `HF_TOKEN` or the Hugging Face login cache;
they are never written to the upload directories or model cards.

Licenses are copied from each upstream model repository when provided. Some
Qwen2.5-VL and Qwen3-VL repositories declare Apache-2.0 only in their model cards.
For these explicitly supported models, the script verifies that declaration and
uses a checksum-verified copy from the official Qwen code repository, bundled in
`scripts/checkpoint_archive_licenses/`. `UPSTREAM_LICENSE.md` records the source.
Keep that directory alongside the script when copying it to another machine.

Staging, checksum cache and upload resume state default to
`/tmp/hf-checkpoint-archive/`, on local disk rather than HDFS. Use
`--work-dir /path/on/local/disk` to change this. Weights are symlinked, so staging does
not require another 183 GiB of disk. JSON loading configurations are copied and
cleaned of local path, experiment and credential fields. Source files and tensor
bytes remain unchanged; tokenizer vocabularies and chat templates are preserved.
Only whitelisted HF model artifacts are uploaded: no evaluation JSON, logs,
datasets, source code, distributed shards or optimizer state.

Initial hashing reads all selected weights and can take some time. Checksums
are cached; interrupted transfers can be resumed with the **same command and
work-dir**. The Hub uploader may retry transient failures for a long time;
interrupt it with Ctrl-C if needed and rerun later. Do not delete the work-dir
or original model files until verification completes.

Existing repositories must have the requested visibility and contain only the
expected archive files. The script refuses visibility mismatches or unrelated
content instead of changing visibility/deleting remote files. Public and private
repo names must differ. Upload completion is verified against remote file sizes
and LFS SHA-256 / Git blob hashes. Success is recorded in `verified.local.json`
inside the work directory. This is a transport verification, not a GPU inference
or model-quality test.

The private mapping, staging state, terminal logs and this local manifest should
not be included in any public code release. Original local files are never
deleted by this script.
