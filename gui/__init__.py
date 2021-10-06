"""GUI Initialization"""

from gui.icons import gen_icon_paths_dict, gen_icons_resource_file

gen_icons_resource_file()  # Create resource (.qrc) file
ICON_PATHS = gen_icon_paths_dict()
