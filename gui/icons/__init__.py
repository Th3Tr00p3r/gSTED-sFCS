"""Icons Initialization"""

import glob
import os
import subprocess
from contextlib import suppress

from utilities.helper import write_list_to_file


def gen_icons_resource_file():
    """documentation."""
    header = "<!DOCTYPE RCC>" + "\n" '<RCC version="1.0">' + "\n" "<qresource>"
    footer = "</qresource>" + "\n" + "</RCC>"
    body = [
        f"<file>{os.path.split(icon_path)[1]}</file>"
        for icon_path in glob.glob("./gui/icons/*.png")
    ]
    body.insert(0, header)
    body.append(footer)
    write_list_to_file("./gui/icons/icons.qrc", body)

    # call pyrcc5 to compile Qt .qrc files (icons) before loading anything
    icon_dir_path = "./gui/icons/"
    binary_path = os.path.join(icon_dir_path, "pyrcc5")
    source_path = os.path.join(icon_dir_path, "icons.qrc")
    destination_path = os.path.join(icon_dir_path, "icons_rc.py")
    #    pyrcc5_command_list = [binary_path, '-o', destination_path, source_path]
    pyrcc5_command = f'"{binary_path}" -o "{destination_path}" "{source_path}"'
    try:
        subprocess.run(pyrcc5_command, check=True, shell=True)
    except subprocess.CalledProcessError:
        print(f"Unable to generate '{destination_path}' - older file will be used.", end=" ")


#    os.system(pyrcc5_command)


def gen_icon_paths_dict(dir_path="./gui/icons", filetype="png"):
    """Doc."""

    icon_paths_dict = {}
    icon_paths = glob.glob(f"{dir_path}/*.{filetype}")
    for icon_path in icon_paths:
        _, icon_fname = os.path.split(icon_path)
        icon_fname_notype = os.path.splitext(icon_fname)[0]
        icon_paths_dict[icon_fname_notype] = f"{dir_path}/{icon_fname}"
    return icon_paths_dict


# Create resource (.qrc) file
with suppress(FileNotFoundError):
    gen_icons_resource_file()
icon_paths_dict = gen_icon_paths_dict()
