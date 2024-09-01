from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import numpy as np

from minigrid.core.constants import (
    COLOR_TO_IDX,
    COLORS,
    IDX_TO_COLOR,
)  # type: ignore
from minigrid.utils.rendering import (
    fill_coords,
    point_in_circle,
    point_in_line,
    point_in_rect,
)  # type: ignore

# Map of object type to integers
OBJECT_TO_IDX = {
    "unseen": 0,
    "empty": 1,
    "wall": 2,
    "floor": 3,
    "door": 4,
    "key": 5,
    "ball": 6,
    "box": 7,
    "goal": 8,
    "lava": 9,
    "agent": 10,
    "teleporter": 11,
}

IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))

from minigrid.core.world_object import Door, Wall, Floor, Ball, Key, Box, Goal, Lava  # type: ignore

if TYPE_CHECKING:
    from minigrid.minigrid_env import MiniGridEnv  # type: ignore

Point = Tuple[int, int]


class WorldObj:
    """
    Base class for grid world objects
    """

    def __init__(self, type: str, color: str):
        assert type in OBJECT_TO_IDX, type
        assert color in COLOR_TO_IDX, color
        self.type = type
        self.color = color
        self.contains = None

        # Initial position of the object
        self.init_pos: Point | None = None

        # Current position of the object
        self.cur_pos: Point | None = None

    def can_overlap(self) -> bool:
        """Can the agent overlap with this?"""
        return False

    def can_pickup(self) -> bool:
        """Can the agent pick this up?"""
        return False

    def can_contain(self) -> bool:
        """Can this contain another object?"""
        return False

    def see_behind(self) -> bool:
        """Can the agent see behind this object?"""
        return True

    def toggle(self, env: MiniGridEnv, pos: tuple[int, int]) -> bool:
        """Method to trigger/toggle an action this object performs"""
        return False

    def encode(self) -> tuple[int, int, int]:
        """Encode the a description of this object as a 3-tuple of integers"""
        return (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], 0)

    @staticmethod
    def decode(type_idx: int, color_idx: int, state: int) -> WorldObj | None:
        """Create an object from a 3-tuple state description"""

        obj_type = IDX_TO_OBJECT[type_idx]
        color = IDX_TO_COLOR[color_idx]

        if obj_type == "empty" or obj_type == "unseen":
            return None

        # State, 0: open, 1: closed, 2: locked
        is_open = state == 0
        is_locked = state == 2

        if obj_type == "wall":
            v = Wall(color)
        elif obj_type == "floor":
            v = Floor(color)
        elif obj_type == "ball":
            v = Ball(color)
        elif obj_type == "key":
            v = Key(color)
        elif obj_type == "box":
            v = Box(color)
        elif obj_type == "door":
            v = Door(color, is_open, is_locked)
        elif obj_type == "goal":
            v = Goal()
        elif obj_type == "lava":
            v = Lava()
        elif obj_type == "teleporter":
            v = Teleporter()
        else:
            assert False, "unknown object type in decode '%s'" % obj_type

        return v

    def render(self, r: np.ndarray) -> np.ndarray:
        """Draw this object with the given renderer"""
        raise NotImplementedError


class Teleporter(WorldObj):
    """Object that teleports the agent to a set of locations with a given
    probability.
    """

    def __init__(
        self,
        active: bool = True,
        active_colour: str = "blue",
        inactive_colour: str = "grey",
    ):
        self.is_active = active
        self.active_colour = active_colour
        self.inactive_colour = inactive_colour
        colour = active_colour if active else inactive_colour
        super().__init__("teleporter", colour)

    def render(self, img: np.ndarray):
        active_colour = COLORS[self.active_colour]
        inactive_colour = COLORS[self.inactive_colour]
        if self.is_active:
            # Draw outer ring of the teleporter
            fill_coords(img, point_in_circle(cx=0.5, cy=0.5, r=0.48), active_colour)
            fill_coords(img, point_in_circle(cx=0.5, cy=0.5, r=0.40), (0, 0, 0))

            # Draw inner glowing circle
            fill_coords(img, point_in_circle(cx=0.5, cy=0.5, r=0.20), active_colour)

            # Add concentric circles for effect
            fill_coords(img, point_in_circle(cx=0.5, cy=0.5, r=0.16), (0, 0, 0))
            fill_coords(img, point_in_circle(cx=0.5, cy=0.5, r=0.12), active_colour)
        else:
            # Inactive teleporter (blue)
            fill_coords(
                img,
                point_in_circle(cx=0.5, cy=0.5, r=0.48),
                0.5 * np.array(inactive_colour),
            )
            fill_coords(img, point_in_circle(cx=0.5, cy=0.5, r=0.40), (0, 0, 0))

            # Draw inner circle
            fill_coords(
                img,
                point_in_circle(cx=0.5, cy=0.5, r=0.20),
                0.5 * np.array(inactive_colour),
            )

            # Add concentric circles for effect
            fill_coords(img, point_in_circle(cx=0.5, cy=0.5, r=0.16), (0, 0, 0))
            fill_coords(
                img,
                point_in_circle(cx=0.5, cy=0.5, r=0.12),
                0.5 * np.array(inactive_colour),
            )
