import torch


def resolve_device(requested="auto"):
    """Resolve and validate a training/inference device.

    auto prefers CUDA, then Apple MPS, then CPU. Explicit accelerator
    requests fail loudly rather than silently falling back to CPU.
    """
    requested = str(requested).strip().lower()

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
            and torch.backends.mps.is_built()
        ):
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cpu":
        return torch.device("cpu")

    if requested == "mps":
        if not (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
            and torch.backends.mps.is_built()
        ):
            raise RuntimeError(
                "MPS was requested but is not available in this PyTorch environment."
            )
        return torch.device("mps")

    if requested == "cuda" or requested.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but torch.cuda.is_available() is False."
            )
        try:
            device = torch.device(requested)
        except (RuntimeError, ValueError) as exc:
            raise ValueError(f"Invalid device specification: {requested}") from exc
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device index {device.index} requested, but only "
                f"{torch.cuda.device_count()} CUDA device(s) are available."
            )
        return device

    raise ValueError(
        f"Unsupported device '{requested}'. Choose auto, cpu, mps, cuda, or cuda:N."
    )


def load_model_state_compatibly(model, state_dict):
    """Load current checkpoints and migrate pre-shifting Transformer checkpoints.

    Before shifting was added, token ID 8 represented CLS. In the current
    vocabulary token ID 8 is shifting and token ID 9 is CLS. Preserve the
    old CLS embedding at its new index and leave the new shifting row at the
    model's fresh initialization.
    """
    state_dict = dict(state_dict)
    old_embedding = state_dict.get("type_embedding.weight")
    current_embedding = model.state_dict()["type_embedding.weight"]

    if (
        old_embedding is not None
        and old_embedding.ndim == 2
        and current_embedding.ndim == 2
        and old_embedding.shape[0] + 1 == current_embedding.shape[0]
        and old_embedding.shape[1] == current_embedding.shape[1]
    ):
        migrated = current_embedding.clone()
        migrated[: old_embedding.shape[0] - 1] = old_embedding[:-1]
        migrated[-1] = old_embedding[-1]
        state_dict["type_embedding.weight"] = migrated

    model.load_state_dict(state_dict)
    return model
