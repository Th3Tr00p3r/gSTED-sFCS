from pathlib import Path

from PyQt5.QtGui import QIcon

ICONS_PATH = Path("./gui/icons")


def gen_icon_paths_dict(dir_path: Path = ICONS_PATH, filetype="png"):
    """Doc."""

    icon_paths_dict = {}
    icon_paths = dir_path.glob(f"*.{filetype}")
    for icon_path in icon_paths:
        *_, icon_fname = Path(icon_path).parts
        icon_fname_no_extension = Path(icon_fname).stem
        icon_paths_dict[icon_fname_no_extension] = f"{dir_path}/{icon_fname}"
    return icon_paths_dict


def get_icon_paths():
    """Doc."""

    return {key: QIcon(val) for key, val in gen_icon_paths_dict().items()}
