"""Auto-download WeDetect checkpoints from the HuggingFace Hub.

All WeDetect weights live in a single public repo (``fushh7/WeDetect``).
The filenames on the Hub match the local names used by the app:

    wedetect_tiny.pth
    wedetect_base.pth
    wedetect_large.pth
    wedetect_base_uni.pth
    wedetect_large_uni.pth

Usage (from Streamlit pages)::

    from src.checkpoint_manager import ensure_checkpoint
    ensure_checkpoint(checkpoint_path)   # downloads if missing, raises on error
"""

from __future__ import annotations

from pathlib import Path

HF_REPO_ID = "fushh7/WeDetect"


def ensure_checkpoint(
    checkpoint_path: Path,
    repo_id: str = HF_REPO_ID,
) -> Path:
    """Ensure *checkpoint_path* exists, downloading it from HuggingFace if needed.

    Parameters
    ----------
    checkpoint_path:
        Absolute path where the checkpoint should live (e.g.
        ``WEDETECT_DIR / "checkpoints/wedetect_base.pth"``).
    repo_id:
        HuggingFace repo that hosts the file.  The filename on the Hub must
        match ``checkpoint_path.name``.

    Returns
    -------
    Path
        The same *checkpoint_path* (guaranteed to exist on success).

    Raises
    ------
    ImportError
        If ``huggingface_hub`` is not installed.
    Exception
        Any network or permission error raised by ``hf_hub_download``.
    """
    if checkpoint_path.exists():
        return checkpoint_path

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required for automatic checkpoint download. "
            "Install it with: pip install huggingface_hub"
        ) from exc

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    hf_hub_download(
        repo_id=repo_id,
        filename=checkpoint_path.name,
        local_dir=str(checkpoint_path.parent),
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Download appeared to succeed but {checkpoint_path} is still missing. "
            "Check HuggingFace Hub connectivity and disk space."
        )

    return checkpoint_path
