"""Tests for the baseline A* planner."""

from src.environment import WarehouseEnvironment
from src.planner import AStarPlanner


def test_astar_finds_path_in_default_warehouse():
    """A* should find a valid path between the default start and target."""

    env = WarehouseEnvironment()
    planner = AStarPlanner(env)
    path = planner.plan(env.start, env.target)

    assert path[0] == env.start
    assert path[-1] == env.target
    assert all(not env.is_blocked(cell) for cell in path)


def test_astar_returns_empty_when_goal_blocked_by_dynamic_obstacle():
    """A* should report failure when the goal is dynamically blocked."""

    env = WarehouseEnvironment()
    env.add_dynamic_obstacle(env.target)
    planner = AStarPlanner(env)

    assert planner.plan(env.start, env.target) == []
