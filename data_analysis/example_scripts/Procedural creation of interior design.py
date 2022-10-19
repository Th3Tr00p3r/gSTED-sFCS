# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# Imports and general function definitions

# %%
import numpy as np

from dataclasses import dataclass, field
from random import random, randint, choice, uniform
from copy import copy, deepcopy
from typing import Union, Tuple
from shapely.geometry import Point, Polygon, LineString, MultiPoint, MultiLineString, MultiPolygon
from shapely.ops import unary_union, nearest_points, split
from polyskel_master import polyskel
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb
from descartes.patch import PolygonPatch
from types import SimpleNamespace
from pathlib import Path
from winsound import Beep
import os

###############################################
# Move to project root to easily import modules
###############################################

try:  # avoid changes if already set
    print("Working from: ", PROJECT_ROOT)
except NameError:
    try:  # running from Spyder
        PROJECT_ROOT = Path(__file__).resolve()
    except NameError:  # running as Jupyter Notebook
        PROJECT_ROOT = Path(os.getcwd()).resolve().parent / "swat"
    os.chdir(PROJECT_ROOT)
    print("Working from: ", PROJECT_ROOT)

from map_generation.geometry_helper import (
    display_polygons,
)
from map_generation.map_generation import (
    get_random_floor,
    create_halls_polygon,
    create_rooms,
)

# %% [markdown]
# Until now we dealt with creating a random floor and hall coordinates:

# %%
MAP_SIZE = 50  # N x N tiles

floor = get_random_floor(
    (MAP_SIZE, MAP_SIZE),
    rect_dim_factors=(0.05, 0.5),
    min_clearance=5,
    min_area_intersection_ratio=0.001,
)
display(floor)
display_polygons(floor)

# %%
halls = create_halls_polygon(floor, hall_width=2, dist_factor=0.01, min_wall_dist=3)

# TODO: hallways should take up, on average, a larger area out of the total floor - in short, there's not enough hallway space.
# Figure out how this can be acheived (adding more halls up to a certain area ratio?).
# This is especially important for square maps, where the straight skelleton becomes too simple, or even fails!

# %% [markdown]
# Next, dividing into rooms. After some thoughts, I decided to try the following rough algorithm:
# 1) get rooms_area = floor - walls - halls (MultiPolygon, subscriptable into polygons - "rooms")
#
# 2) keep polygons under certain size as rooms (with some randomness - sometimes leaving bigger rooms, sometimes reaching minimum if possible)
#
# 3) divide each "big" polygon by extending a wall (1 tile thick polygon) from one of its vertices, with some limits on the polygons it creates (choose other wall if chunks are unusable
#
# 4) back to (2)

# %%
rooms_area = floor - halls
room_list = create_rooms(
    rooms_area,
    floor,
    div_prob=0.1,
    max_area_factor=0.1,
    hallway_door_prob=0.9,
    neighbor_door_prob=0.2,
    min_clearance=2,
    max_ecc=0.75,
    window_wall_ratio=0.3,
)

# print room list # TODO: perhaps room list should be a class laster on - could be useful to perform actions/get attributes on all rooms easily
print(room_list)

# TODO: hallways should be rooms as well (of type "hall") - this way the rooms list is the whole map?
# TODO: If there are no halls, don't look for hallway connections when setting up doors! (infinite loop)

# %% [markdown]
# Plot what we have so far

# %%
display_polygons(
    [
        floor,
        halls,
        [room.poly for room in room_list],
    ],
    [
        dict(facecolor="gray", alpha=0.3),
        dict(facecolor="red", alpha=0.5),
        dict(facecolor="blue", alpha=0.5),
    ],
)

# %% [markdown]
# Draw geometries on array. This will be the final description of a map - each array element represents a single tile.

# %% [markdown]
# Drawing the rooms (which already include the halls in their borders)

# %%
# TODO: halls that border with the outer walls cause 1-tile halls - why does this happen?

from map_generation.map_generation import (
    initiate_blank_map_array,
    draw_polygon_on_array,
    draw_poly_boundary_on_array,
    print_map_array,
    draw_rooms_and_halls_on_array,
)


map_arr = initiate_blank_map_array((MAP_SIZE, MAP_SIZE))
floor_drawn = draw_poly_boundary_on_array(floor, map_arr, "W", should_erode=False)
rooms_drawn = draw_rooms_and_halls_on_array(room_list, floor_drawn, "R")
print_map_array(rooms_drawn)

# %% [markdown]
# Fixing missing corners, usually at hall ends - can this be avoided in the previous stage (poly to arr)?

# %%
from map_generation.map_generation import fill_walls, CHAR_INT_DICT

walls_filled = fill_walls(rooms_drawn, wall_int=CHAR_INT_DICT["R"], max_non_NESW_conns=1)
print_map_array(walls_filled)

# %% [markdown]
# Beep

# %%
Beep(4000, 500)
