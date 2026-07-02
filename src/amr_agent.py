"""AMR agent state and motion model."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.environment import WarehouseEnvironment
from src.utils import GridPosition, manhattan


@dataclass
class AMRAgent:
    """Autonomous mobile robot state used by the reliability demo."""

    position: GridPosition
    target: GridPosition
    path: list[GridPosition] = field(default_factory=list)
    speed_state: str = "normal"
    localization_uncertainty: float = 0.05
    sensor_confidence: float = 0.98
    replanning_failure_count: int = 0
    stagnant_steps: int = 0
    completed: bool = False
    deviated_from_path: bool = False
    needs_replan: bool = False
    last_distance_to_target: int | None = None
    blocked_move_count: int = 0

    def set_path(self, path: list[GridPosition]) -> None:
        """Assign a new path and clear path-related failure state."""

        self.path = path
        self.needs_replan = False
        self.deviated_from_path = self.position not in set(path) if path else True

    def update_target(self, target: GridPosition) -> None:
        """Update the active goal and mark the current path as stale."""

        self.target = target
        self.needs_replan = True

    def next_path_cell(self) -> GridPosition | None:
        """Return the next cell on the current path, if available."""

        if not self.path:
            return None
        try:
            idx = self.path.index(self.position)
        except ValueError:
            return None
        if idx + 1 >= len(self.path):
            return None
        return self.path[idx + 1]

    def move_along_path(self, environment: WarehouseEnvironment, allow_move: bool = True) -> bool:
        """Move one grid step along the planned path if the next cell is free."""

        if self.completed or not allow_move:
            self._update_progress()
            return False

        next_cell = self.next_path_cell()
        if next_cell is None or environment.is_blocked(next_cell):
            self.blocked_move_count += 1
            self._update_progress()
            return False

        self.position = next_cell
        self.blocked_move_count = 0
        self.deviated_from_path = False
        self.completed = self.position == self.target
        self._update_progress()
        return True

    def force_deviation(self, environment: WarehouseEnvironment) -> bool:
        """Move the robot to a nearby free cell outside the current path."""

        path_cells = set(self.path)
        x, y = self.position
        candidates = [(x, y + 1), (x + 1, y), (x, y - 1), (x - 1, y)]
        for cell in candidates:
            if not environment.is_blocked(cell) and cell not in path_cells:
                self.position = cell
                self.deviated_from_path = True
                self.needs_replan = True
                self._update_progress()
                return True
        return False

    def relocalize(self) -> None:
        """Reduce localization uncertainty after a simulated relocalization action."""

        self.localization_uncertainty = max(0.05, self.localization_uncertainty * 0.25)
        self.deviated_from_path = self.position not in set(self.path)

    def _update_progress(self) -> None:
        """Update stagnation based on distance-to-goal improvement."""

        distance = manhattan(self.position, self.target)
        if self.last_distance_to_target is None or distance < self.last_distance_to_target:
            self.stagnant_steps = 0
        else:
            self.stagnant_steps += 1
        self.last_distance_to_target = distance
