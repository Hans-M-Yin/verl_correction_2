"""Legacy bulk uploader disabled to prevent accidental public dataset uploads.

Use ``scripts/upload_private_datasets.py`` for the reviewed private dataset
archive. That script enforces an explicit file allowlist and private visibility.
"""

if __name__ == "__main__":
    raise SystemExit(
        "This legacy recursive uploader is disabled. "
        "Use scripts/upload_private_datasets.py with an explicit private repo."
    )
