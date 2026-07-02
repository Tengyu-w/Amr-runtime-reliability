"""2D warehouse grid environment for AMR runtime reliability experiments."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.utils import GridPosition, SimulationConfig, manhattan


@dataclass
class WarehouseEnvironment:
    """A lightweight warehouse grid with shelves, aisles, and moving obstacles."""

    config: SimulationConfig = field(default_factory=SimulationConfig)
    start: GridPosition = (1, 1)
    target: GridPosition = (16, 10)
    alternate_targets: list[GridPosition] = field(
        default_factory=lambda: [(2, 10), (15, 1), (12, 8)]
    )
    shelves: set[GridPosition] = field(default_factory=set)
    static_obstacles: set[GridPosition] = field(default_factory=set)
    dynamic_obstacles: set[GridPosition] = field(default_factory=set)

    def __post_init__(self) -> None:
        """Populate a deterministic warehouse layout if no layout was provided."""

        if not self.shelves and not self.static_obstacles:
            self._build_default_layout()

    @property
    def width(self) -> int:
        """Return grid width."""

        return self.config.width

    @property
    def height(self) -> int:
        """Return grid height."""

        return self.config.height

    def _build_default_layout(self) -> None:
        """Create shelves and static obstacles while leaving navigable aisles."""

        shelves: set[GridPosition] = set()
        for x in [4, 5, 9, 10, 14]:
            for y in range(2, self.height - 2):
                if y in {5, 8}:
                    continue
                shelves.add((x, y))

        walls = {(0, y) for y in range(self.height)}
        walls |= {(self.width - 1, y) for y in range(self.height)}
        walls |= {(x, 0) for x in range(self.width)}
        walls |= {(x, self.height - 1) for x in range(self.width)}

        self.shelves = shelves
        self.static_obstacles = walls | shelves | {(7, 3), (7, 4), (12, 6)}

    def in_bounds(self, position: GridPosition) -> bool:
        """Return True if a position is inside the grid."""

        x, y = position
        return 0 <= x < self.width and 0 <= y < self.height

    def is_static_blocked(self, position: GridPosition) -> bool:
        """Return True if a cell is occupied by a wall, shelf, or fixed obstacle."""

        return position in self.static_obstacles

    def is_blocked(self, position: GridPosition, include_dynamic: bool = True) -> bool:
        """Return True if a cell is blocked for navigation."""

        if not self.in_bounds(position) or self.is_static_blocked(position):
            return True
        return include_dynamic and position in self.dynamic_obstacles

    def neighbors(self, position: GridPosition, include_dynamic: bool = True) -> list[GridPosition]:
        """Return navigable 4-connected neighbors."""

        x, y = position
        candidates = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        return [cell for cell in candidates if not self.is_blocked(cell, include_dynamic)]

    def set_dynamic_obstacles(self, obstacles: set[GridPosition]) -> None:
        """Replace dynamic obstacles with valid, non-static cells."""

        self.dynamic_obstacles = {
            cell
            for cell in obstacles
            if self.in_bounds(cell) and not self.is_static_blocked(cell)
        }

    def add_dynamic_obstacle(self, position: GridPosition) -> bool:
        """Add a dynamic obstacle if the cell can hold one."""

        if self.in_bounds(position) and not self.is_static_blocked(position):
            self.dynamic_obstacles.add(position)
            return True
        return False

    def clear_dynamic_obstacles(self) -> None:
        """Remove all dynamic obstacles."""

        self.dynamic_obstacles.clear()

    def change_target(self, target: GridPosition) -> None:
        """Move the active task target to a new unblocked cell."""

        if self.is_blocked(target, include_dynamic=False):
            raise ValueError(f"Target {target} is not navigable.")
        self.target = target

    def obstacle_distance(self, position: GridPosition) -> int:
        """Return Manhattan distance to the closest static or dynamic obstacle."""

        obstacles = self.static_obstacles | self.dynamic_obstacles
        return min(manhattan(position, obstacle) for obstacle in obstacles)
