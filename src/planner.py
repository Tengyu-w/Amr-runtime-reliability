"""Baseline A* planner used by the AMR demo."""

from __future__ import annotations

import heapq
from dataclasses import dataclass

from src.environment import WarehouseEnvironment
from src.utils import GridPosition, manhattan


@dataclass
class AStarPlanner:
    """A minimal A* grid planner.

    This planner is intentionally simple. The project focus is runtime
    reliability supervision, not novel path planning.
    """

    environment: WarehouseEnvironment

    def plan(
        self,
        start: GridPosition,
        goal: GridPosition,
        include_dynamic: bool = True,
    ) -> list[GridPosition]:
        """Plan a path from start to goal, returning an empty list on failure."""

        if self.environment.is_blocked(start, include_dynamic=False):
            return []
        if self.environment.is_blocked(goal, include_dynamic=include_dynamic):
            return []

        frontier: list[tuple[int, int, GridPosition]] = []
        heapq.heappush(frontier, (0, 0, start))
        came_from: dict[GridPosition, GridPosition | None] = {start: None}
        cost_so_far: dict[GridPosition, int] = {start: 0}
        sequence = 0

        while frontier:
            _, _, current = heapq.heappop(frontier)
            if current == goal:
                return self._reconstruct(came_from, current)

            for neighbor in self.environment.neighbors(current, include_dynamic):
                new_cost = cost_so_far[current] + 1
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + manhattan(neighbor, goal)
                    sequence += 1
                    heapq.heappush(frontier, (priority, sequence, neighbor))
                    came_from[neighbor] = current

        return []

    @staticmethod
    def _reconstruct(
        came_from: dict[GridPosition, GridPosition | None],
        current: GridPosition,
    ) -> list[GridPosition]:
        """Reconstruct a path from an A* predecessor map."""

        path = [current]
        while came_from[current] is not None:
            current = came_from[current]  # type: ignore[assignment]
            path.append(current)
        path.reverse()
        return path
