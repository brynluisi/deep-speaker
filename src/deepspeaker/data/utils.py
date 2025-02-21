from pathlib import Path


def get_input_target_from_path(data_dir: str) -> tuple[Path, Path]:
    p = Path(data_dir)

    if len(list(p.glob(f"*-input.wav"))) == 0:
        raise ValueError(f"No input files found in {data_dir}")

    if len(list(p.glob(f"*-target.wav"))) == 0:
        raise ValueError(f"No target files found in {data_dir}")

    if len(list(p.glob(f"*-input.wav"))) > 1:
        raise ValueError(f"Multiple input files found in {data_dir}")

    if len(list(p.glob(f"*-target.wav"))) > 1:
        raise ValueError(f"Multiple target files found in {data_dir}")

    input_path = list(p.glob(f"*-input.wav"))[0]
    target_path = list(p.glob(f"*-target.wav"))[0]

    return input_path, target_path


def get_path(data_dir: str, input_or_target: str) -> Path:
    """

    Args:
        data_dir:
        input_or_target:

    Returns:
        path:

    """
    p = Path(data_dir)

    if len(list(p.glob(f"*-{input_or_target}.wav"))) == 0:
        raise ValueError(f"No {input_or_target} files found in {data_dir}")

    if len(list(p.glob(f"*-{input_or_target}.wav"))) > 1:
        raise ValueError(f"Multiple {input_or_target} files found in {data_dir}")

    path = list(p.glob(f"*-{input_or_target}.wav"))[0]

    return path
