"""Legacy checkpoint uploader disabled pending explicit repository review.

Use ``scripts/upload_selected_checkpoints.py`` for the reviewed checkpoint
archive. It checks visibility and uploads only the selected model files.
"""

if __name__ == "__main__":
    raise SystemExit(
        "This legacy uploader is disabled. "
        "Use scripts/upload_selected_checkpoints.py with explicit repositories."
    )
