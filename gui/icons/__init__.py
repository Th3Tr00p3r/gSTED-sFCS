"""GUI Initialization"""

import subprocess
from pathlib import Path

from utilities.helper import write_list_to_file

ICON_DIR_PATH = Path("./gui/icons/")


def generate_icons_resource_file(icon_dir_path: Path = ICON_DIR_PATH):
    """Doc."""

    header = "<!DOCTYPE RCC>" + "\n" '<RCC version="1.0">' + "\n" "<qresource>"
    footer = "</qresource>" + "\n" + "</RCC>"
    body = [
        f"<file>{Path(icon_path).parts[-1]}</file>" for icon_path in icon_dir_path.glob("*.png")
    ]
    body.insert(0, header)
    body.append(footer)
    write_list_to_file(icon_dir_path / "icons.qrc", body)

    # call pyrcc5 to compile Qt .qrc files (icons) before loading anything
    binary_path = icon_dir_path / "pyrcc5"
    source_path = icon_dir_path / "icons.qrc"
    destination_path = icon_dir_path / "icons_rc.py"
    #    pyrcc5_command_list = [binary_path, '-o', destination_path, source_path]
    pyrcc5_command = f'"{binary_path}" -o "{destination_path}" "{source_path}"'
    try:
        subprocess.run(pyrcc5_command, check=True, shell=True)
    except subprocess.CalledProcessError:
        print(f"Unable to generate '{destination_path}' - existing file will be used.", end=" ")


generate_icons_resource_file()  # Create resource (.qrc) file
