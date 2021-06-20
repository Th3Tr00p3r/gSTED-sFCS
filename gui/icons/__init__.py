"""Icons Initialization"""

import glob
import os


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
    current_dir_path = os.getcwd()
    icon_dir_path = current_dir_path + "\\gui\\icons"
    if " " in current_dir_path:
        pyrcc5_command = f'"{icon_dir_path}\\pyrcc5" -o "{icon_dir_path}\\icons_rc.py" "{icon_dir_path}\\icons.qrc"'
    else:
        pyrcc5_command = (
            f"{icon_dir_path}\\pyrcc5 -o {icon_dir_path}\\icons_rc.py {icon_dir_path}\\icons.qrc"
        )
    os.system(pyrcc5_command)


def gen_icon_paths_dict():
    """Doc."""
    # TODO: make this a more general fucntion - accept args for filetype (png), path to look in etc.

    icon_paths_dict = {}
    icon_paths = glob.glob("./gui/icons/*.png")
    for icon_path in icon_paths:
        _, icon_fname = os.path.split(icon_path)
        icon_fname_notype = os.path.splitext(icon_fname)[0]
        icon_paths_dict[icon_fname_notype] = f"./gui/icons/{icon_fname}"
    return icon_paths_dict


# Create resource (.qrc) file
gen_icons_resource_file()
ICON_PATHS_DICT = gen_icon_paths_dict()
