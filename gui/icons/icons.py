import glob
import os

from PyQt5.QtGui import QIcon


def gen_icon_paths_dict(dir_path="./gui/icons", filetype="png"):
    """Doc."""

    icon_paths_dict = {}
    icon_paths = glob.glob(f"{dir_path}/*.{filetype}")
    for icon_path in icon_paths:
        _, icon_fname = os.path.split(icon_path)
        icon_fname_notype = os.path.splitext(icon_fname)[0]
        icon_paths_dict[icon_fname_notype] = f"{dir_path}/{icon_fname}"
    return icon_paths_dict


def get_icon_paths():
    """Doc."""

    return {key: QIcon(val) for key, val in gen_icon_paths_dict().items()}
