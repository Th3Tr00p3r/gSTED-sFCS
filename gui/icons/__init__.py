"""Icons Initialization"""

import glob
import logging
import os
import subprocess


def gen_icons_resource_file():
    """documentation."""
    header = "<!DOCTYPE RCC>" + "\n" '<RCC version="1.0">' + "\n" "<qresource>" + "\n"
    footer = "</qresource>" + "\n" + "</RCC>" + "\n"
    icon_paths = glob.glob("./gui/icons/*.png")

    with open("./gui/icons/icons.qrc", "w") as f:
        f.write(header)
        for icon_path in icon_paths:
            _, icon_fname = os.path.split(icon_path)
            f.write(f"<file>{icon_fname}</file>" + "\n")
        f.write(footer)

    # call pyrcc5 to compile Qt .qrc files (icons) before loading anything
    icon_dir_path = os.getcwd() + "\\gui\\icons"
    binary_path = icon_dir_path + "\\pyrcc5"
    source_path = icon_dir_path + "\\icons.qrc"
    destination_path = icon_dir_path + "\\icons_rc.py"
    #    pyrcc5_command_list = [binary_path, '-o', destination_path, source_path]
    pyrcc5_command = f'"{binary_path}" -o "{destination_path}" "{source_path}"'
    try:
        subprocess.run(pyrcc5_command, check=True, shell=True)
    except subprocess.CalledProcessError:
        logging.warning(f"Unable to generate {destination_path}. older file will be used.")


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
gen_icons_resource_file()
ICON_PATHS_DICT = gen_icon_paths_dict()
