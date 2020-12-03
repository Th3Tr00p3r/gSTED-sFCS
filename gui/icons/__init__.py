# -*- coding: utf-8 -*-
""" documentation."""

import glob
import os


def gen_icons_resource_file():
    """documentation."""
    header = (
        "<!DOCTYPE RCC>"
        + "\n"
        + '<RCC version="1.0">'
        + "\n"
        + "<qresource>"
        + "\n"
    )
    footer = "</qresource>" + "\n" + "</RCC>" + "\n"

    icon_paths = glob.glob("./gui/icons/*.png")

    with open("./gui/icons/icons.qrc", "w") as f:
        f.write(header)
        for icon_path in icon_paths:
            _, icon_fname = os.path.split(icon_path)
            f.write("<file>" + icon_fname + "</file>" + "\n")
        f.write(footer)


def gen_icon_paths_module():
    header = (
        "# -*- coding: utf-8 -*-\n"
        + '"""\n'
        + "Icon path constants\n"
        + "(This is an automatically generated file.\n"
        + "if needed, changes should be made in"
        + "'./icons/__init__.py)\n\n'"
        + '"""\n\n'
    )
    icon_paths = glob.glob("./gui/icons/*.png")
    with open("./gui/icons/icon_paths.py", "w") as f:
        f.write(header)
        for icon_path in icon_paths:
            _, icon_fname = os.path.split(icon_path)
            icon_fname_notype = os.path.splitext(icon_fname)[0]
            f.write(
                icon_fname_notype.upper()
                + " = '"
                + "./gui/icons/"
                + icon_fname
                + "'"
                + "\n"
            )


gen_icons_resource_file()
gen_icon_paths_module()
