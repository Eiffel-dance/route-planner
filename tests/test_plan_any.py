import itertools
import random
import unittest

import app
from app import plan, plan_any, replay


def brute_force_static(width, height, blocked, start, goals, costs):
    """Every simple route from ``start`` to any goal, exactly ranked."""
    def step_cost(point):
        if costs is None:
            return 1
        return costs[point[1]][point[0]]

    best = None  # (cost, goal x, goal y, path)
    goal_set = set(goals)

    def visit(path, seen, cost):
        nonlocal best
        current = path[-1]
        if current in goal_set:
            key = (cost, current[0], current[1], tuple(path))
            if best is None or key < best:
                best = key
            return  # routes end at a goal
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (current[0] + dx, current[1] + dy)
            if not (0 <= nxt[0] < width and 0 <= nxt[1] < height):
                continue
            if nxt in blocked or nxt in seen:
                continue
            new_cost = cost + step_cost(nxt)
            if best is not None and new_cost > best[0]:
                continue
            path.append(nxt)
            seen.add(nxt)
            visit(path, seen, new_cost)
            seen.discard(nxt)
            path.pop()

    if start in goal_set:
        best = (0, start[0], start[1], (start,))
    else:
        visit([start], {start}, 0)
    return best


def brute_force_dynamic(width, height, blocked, frames, start, goals, costs):
    """Every feasible route in the time-expanded grid, exactly ranked."""
    def step_cost(point):
        if costs is None:
            return 1
        return costs[point[1]][point[0]]

    def frame_cells(t):
        return frames[t] if t < len(frames) else frames[-1]

    best = None  # (cost, goal x, goal y, path)
    goal_set = set(goals)

    def visit(path, seen, cost):
        nonlocal best
        current = path[-1]
        t = len(path) - 1
        if current in goal_set:
            key = (cost, current[0], current[1], tuple(path))
            if best is None or key < best:
                best = key
            return  # routes end at a goal
        frame = frame_cells(t + 1)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (current[0] + dx, current[1] + dy)
            if not (0 <= nxt[0] < width and 0 <= nxt[1] < height):
                continue
            if nxt in blocked or nxt in frame or nxt in seen:
                continue
            new_cost = cost + step_cost(nxt)
            if best is not None and new_cost > best[0]:
                continue
            path.append(nxt)
            seen.add(nxt)
            visit(path, seen, new_cost)
            seen.discard(nxt)
            path.pop()

    if start in goal_set:
        best = (0, start[0], start[1], (start,))
    else:
        visit([start], {start}, 0)
    return best


class PlanAnyBasicTest(unittest.TestCase):
    def test_single_goal_matches_plan(self):
        blocked = {(2, 0), (2, 1), (2, 2)}
        expected = plan(6, 4, blocked, (0, 0), (5, 3))
        result = plan_any(6, 4, blocked, (0, 0), [(5, 3)])
        self.assertEqual(result["cost"], expected["cost"])
        self.assertEqual(result["path"][0], (0, 0))
        self.assertEqual(result["path"][-1], (5, 3))
        self.assertEqual(set(result), {"path", "cost", "expanded"})
        check = replay(6, 4, blocked, (0, 0), (5, 3), result["path"])
        self.assertEqual(check, {"valid": True, "cost": expected["cost"],
                                 "steps": len(result["path"]) - 1})

    def test_nearest_goal_wins(self):
        result = plan_any(5, 5, set(), (0, 0), [(4, 4), (2, 0), (3, 3)])
        self.assertEqual(result["path"], [(0, 0), (1, 0), (2, 0)])
        self.assertEqual(result["cost"], 2)

    def test_duplicate_goals_merged(self):
        single = plan_any(4, 4, {(1, 1)}, (0, 0), [(3, 3)])
        duplicated = plan_any(4, 4, {(1, 1)}, (0, 0),
                              [(3, 3), (3, 3), [3, 3]])
        self.assertEqual(single, duplicated)

    def test_goal_permutation_invariance(self):
        goals = [(4, 0), (0, 4), (2, 2), (4, 4)]
        blocked = [(1, 1), (3, 3), (0, 2)]
        reference = plan_any(5, 5, blocked, (0, 0), goals, trace=True)
        for perm in itertools.permutations(goals):
            result = plan_any(5, 5, blocked, (0, 0), list(perm), trace=True)
            self.assertEqual(result, reference)

    def test_blocked_and_frame_iteration_order_invariance(self):
        cells = [(x, 1) for x in range(5) if x != 2]
        frames = [[(0, 2), (3, 2)], [(4, 0)]]
        reference = plan_any(5, 3, cells, (0, 0), [(4, 2), (2, 0)],
                             dynamic_blocked=frames, trace=True)
        shuffled = plan_any(5, 3, list(reversed(cells)), (0, 0),
                            [(2, 0), (4, 2)],
                            dynamic_blocked=[[(3, 2), (0, 2)], [(4, 0)]],
                            trace=True)
        self.assertEqual(reference, shuffled)

    def test_cost_tie_broken_by_goal_coordinates(self):
        # Both goals at distance 2; (0, 2) < (2, 0) lexicographically.
        result = plan_any(3, 3, set(), (0, 0), [(2, 0), (0, 2)])
        self.assertEqual(result["path"], [(0, 0), (0, 1), (0, 2)])
        self.assertEqual(result["cost"], 2)

    def test_path_tie_broken_lexicographically(self):
        # Two equal-cost routes to (1, 1); the one via (0, 1) wins.
        result = plan_any(2, 2, set(), (0, 0), [(1, 1)])
        self.assertEqual(result["path"], [(0, 0), (0, 1), (1, 1)])

    def test_goal_coordinates_beat_path_order(self):
        # (0, 2) and (1, 1) are both reachable at cost 6. The route to
        # (1, 1) is lexicographically smaller, but the goal coordinate
        # tie-break runs first, so (0, 2) wins.
        costs = (
            (1, 1, 1),
            (1, 5, 1),
            (1, 1, 1),
        )
        result = plan_any(3, 3, {(0, 1)}, (0, 0), [(1, 1), (0, 2)],
                          costs=costs)
        self.assertEqual(result["cost"], 6)
        self.assertEqual(result["path"],
                         [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2),
                          (1, 2), (0, 2)])
        # Sanity: the competing route to (1, 1) alone costs the same and
        # compares lexicographically smaller.
        other = plan_any(3, 3, {(0, 1)}, (0, 0), [(1, 1)], costs=costs)
        self.assertEqual(other["cost"], 6)
        self.assertLess(tuple(other["path"]), tuple(result["path"]))

    def test_costs_matrix_respected(self):
        # Direct route enters an expensive cell; the detour is cheaper.
        costs = (
            (1, 9, 1),
            (1, 1, 1),
        )
        result = plan_any(3, 2, set(), (0, 0), [(2, 0)], costs=costs)
        self.assertEqual(result["path"],
                         [(0, 0), (0, 1), (1, 1), (2, 1), (2, 0)])
        self.assertEqual(result["cost"], 4)

    def test_start_in_goals_single_point(self):
        result = plan_any(3, 3, set(), (1, 1), [(2, 2), (1, 1)])
        self.assertEqual(result, {"path": [(1, 1)], "cost": 0,
                                  "expanded": 1})

    def test_unreachable(self):
        blocked = {(1, y) for y in range(3)}
        result = plan_any(3, 3, blocked, (0, 0), [(2, 2), (2, 0)])
        self.assertEqual(result, {"path": None, "cost": None,
                                  "expanded": 3})

    def test_replay_round_trip(self):
        blocked = {(2, 0), (2, 1), (2, 2)}
        costs = ((1, 2, 1, 4, 1, 1),
                 (3, 1, 1, 1, 2, 1),
                 (1, 1, 5, 1, 1, 1),
                 (1, 2, 1, 1, 1, 2))
        result = plan_any(6, 4, blocked, (0, 0), [(5, 3), (5, 0)],
                          costs=costs)
        check = replay(6, 4, blocked, (0, 0), result["path"][-1],
                       result["path"], costs=costs)
        self.assertEqual(check, {"valid": True, "cost": result["cost"],
                                 "steps": len(result["path"]) - 1})


class PlanAnyValidationTest(unittest.TestCase):
    def test_outer_type_errors(self):
        for bad in ("ab", b"ab", None, 5, 3.5, {(1, 1)}, True):
            with self.assertRaises(TypeError, msg=f"goals={bad!r}"):
                plan_any(3, 3, set(), (0, 0), bad)

    def test_empty_goals_value_error(self):
        with self.assertRaises(ValueError):
            plan_any(3, 3, set(), (0, 0), [])

    def test_element_type_errors(self):
        for bad in ("ab", (1,), (1, 1, 1), (1.0, 1), (True, 1), None, 5):
            with self.assertRaises(TypeError, msg=f"goal={bad!r}"):
                plan_any(3, 3, set(), (0, 0), [(2, 2), bad])

    def test_out_of_bounds_value_error(self):
        for bad in ((3, 0), (0, 3), (-1, 0), (0, -1)):
            with self.assertRaises(ValueError, msg=f"goal={bad!r}"):
                plan_any(3, 3, set(), (0, 0), [(1, 1), bad])

    def test_goal_on_obstacle_value_error(self):
        with self.assertRaises(ValueError):
            plan_any(3, 3, {(1, 1)}, (0, 0), [(2, 2), (1, 1)])

    def test_goal_on_dynamic_frame_is_not_an_input_error(self):
        # A candidate blocked at frame 0 (or any frame) does not raise;
        # only ``start`` is checked against frame 0. Frame 1 persists, so
        # (2, 2) is simply reached at t = 4 under an empty last frame.
        result = plan_any(3, 3, set(), (0, 0), [(2, 2)],
                          dynamic_blocked=[[(2, 2)], []])
        self.assertEqual(result["cost"], 4)
        self.assertEqual(result["path"][-1], (2, 2))

    def test_start_on_dynamic_frame_zero_raises(self):
        with self.assertRaises(ValueError):
            plan_any(3, 3, set(), (0, 0), [(2, 2)],
                     dynamic_blocked=[[(0, 0)]])

    def test_validation_order_mirrors_plan(self):
        # Grid checks precede goals checks.
        with self.assertRaises(ValueError):  # width, not goals TypeError
            plan_any(0, 3, set(), (0, 0), "nope")
        # Goals checks precede costs checks.
        with self.assertRaises(ValueError):  # empty goals, not costs
            plan_any(3, 3, set(), (0, 0), [], costs="bad")
        # Obstacle checks precede the trace check.
        with self.assertRaises(ValueError):
            plan_any(3, 3, {(1, 1)}, (0, 0), [(1, 1)], trace=1)
        # The trace check precedes the budget check.
        with self.assertRaisesRegex(TypeError, "trace"):
            plan_any(3, 3, set(), (0, 0), [(1, 1)], trace=1,
                     max_expanded="x")
        # The dynamic frame-0 check precedes the budget check.
        with self.assertRaises(ValueError):
            plan_any(3, 3, set(), (0, 0), [(1, 1)],
                     dynamic_blocked=[[(0, 0)]], max_expanded=-1)

    def test_budget_validation(self):
        for bad in ("3", 3.0, 1.5, True, [3], (2,)):
            with self.assertRaises(TypeError, msg=f"max_expanded={bad!r}"):
                plan_any(3, 3, set(), (0, 0), [(1, 1)], max_expanded=bad)
        with self.assertRaises(ValueError):
            plan_any(3, 3, set(), (0, 0), [(1, 1)], max_expanded=-1)


class PlanAnyBudgetTest(unittest.TestCase):
    def test_budget_found(self):
        result = plan_any(4, 1, set(), (0, 0), [(3, 0)], max_expanded=4)
        self.assertEqual(result["status"], "found")
        self.assertEqual(result["cost"], 3)
        self.assertEqual(result["expanded"], 4)

    def test_budget_exhausted(self):
        result = plan_any(4, 1, set(), (0, 0), [(3, 0)], max_expanded=2)
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertIsNone(result["path"])
        self.assertIsNone(result["cost"])
        self.assertEqual(result["expanded"], 2)

    def test_budget_unreachable(self):
        blocked = {(1, y) for y in range(3)}
        result = plan_any(3, 3, blocked, (0, 0), [(2, 2)], max_expanded=10)
        self.assertEqual(result["status"], "unreachable")
        self.assertEqual(result["expanded"], 3)

    def test_zero_budget_with_start_in_goals(self):
        result = plan_any(3, 3, set(), (1, 1), [(1, 1)], max_expanded=0,
                          trace=True)
        self.assertEqual(result, {"path": None, "cost": None,
                                  "expanded": 0,
                                  "status": "budget_exhausted",
                                  "expanded_nodes": []})

    def test_single_point_needs_one_expansion(self):
        result = plan_any(3, 3, set(), (1, 1), [(1, 1)], max_expanded=1)
        self.assertEqual(result["status"], "found")
        self.assertEqual(result["path"], [(1, 1)])
        self.assertEqual(result["cost"], 0)

    def test_budget_keeps_already_found_route(self):
        # The goal is closed before the budget runs out; the already
        # determined route is returned, never replaced or dropped.
        result = plan_any(3, 3, set(), (0, 0), [(1, 0), (2, 2)],
                          max_expanded=2)
        self.assertEqual(result["status"], "found")
        self.assertEqual(result["path"], [(0, 0), (1, 0)])
        self.assertEqual(result["cost"], 1)

    def test_budgeted_trace_is_prefix(self):
        goals = [(4, 4), (4, 0), (0, 4)]
        full = plan_any(5, 5, {(2, 2)}, (0, 0), goals, trace=True)
        for budget in range(0, full["expanded"] + 1):
            limited = plan_any(5, 5, {(2, 2)}, (0, 0), goals, trace=True,
                               max_expanded=budget)
            self.assertEqual(
                limited["expanded_nodes"],
                full["expanded_nodes"][:limited["expanded"]],
            )
            self.assertEqual(limited["expanded"],
                             min(budget, full["expanded"]))


class PlanAnyTraceTest(unittest.TestCase):
    def test_trace_static(self):
        result = plan_any(3, 3, set(), (0, 0), [(2, 2)], trace=True)
        self.assertEqual(result["expanded"], len(result["expanded_nodes"]))
        self.assertEqual(result["expanded_nodes"][0], (0, 0))
        self.assertIn((2, 2), result["expanded_nodes"])
        self.assertEqual(len(set(result["expanded_nodes"])),
                         len(result["expanded_nodes"]))

    def test_trace_omitted_by_default(self):
        result = plan_any(3, 3, set(), (0, 0), [(2, 2)])
        self.assertNotIn("expanded_nodes", result)

    def test_trace_dynamic_records_triples(self):
        result = plan_any(3, 3, set(), (0, 0), [(2, 2)],
                          dynamic_blocked=[[(1, 0)]], trace=True)
        self.assertEqual(result["expanded"], len(result["expanded_nodes"]))
        for entry in result["expanded_nodes"]:
            self.assertEqual(len(entry), 3)
        self.assertEqual(result["expanded_nodes"][0], (0, 0, 0))


class PlanAnyDynamicTest(unittest.TestCase):
    def test_goal_blocked_at_arrival_frame_unreachable(self):
        # 3x1 corridor: (2, 0) is reached exactly at t = 2, which the
        # persistent last frame blocks; no waiting or detour exists.
        result = plan_any(3, 1, set(), (0, 0), [(2, 0)],
                          dynamic_blocked=[[], [], [(2, 0)]])
        self.assertIsNone(result["path"])
        self.assertIsNone(result["cost"])

    def test_other_goal_still_competes(self):
        result = plan_any(3, 1, set(), (0, 0), [(2, 0), (1, 0)],
                          dynamic_blocked=[[], [], [(2, 0)]])
        self.assertEqual(result["path"], [(0, 0), (1, 0)])
        self.assertEqual(result["cost"], 1)

    def test_later_arrival_time_competes(self):
        # (1, 1) is blocked only at t = 2; a four-step detour arrives at
        # t = 4 and is feasible.
        frames = [[], [], [(1, 1)], [], []]
        result = plan_any(3, 3, set(), (0, 0), [(1, 1)],
                          dynamic_blocked=frames)
        self.assertEqual(result["cost"], 4)
        self.assertEqual(result["path"][-1], (1, 1))
        check = replay(3, 3, set(), (0, 0), (1, 1), result["path"],
                       dynamic_blocked=frames)
        self.assertEqual(check, {"valid": True, "cost": 4, "steps": 4})

    def test_dynamic_single_goal_matches_plan(self):
        frames = [[], [(2, 1)], [(1, 2)]]
        expected = plan(4, 4, {(1, 1)}, (0, 0), (3, 3),
                        dynamic_blocked=frames)
        result = plan_any(4, 4, {(1, 1)}, (0, 0), [(3, 3)],
                          dynamic_blocked=frames)
        self.assertEqual(result["path"], expected["path"])
        self.assertEqual(result["cost"], expected["cost"])

    def test_dynamic_start_in_goals(self):
        result = plan_any(3, 3, set(), (1, 1), [(1, 1), (2, 2)],
                          dynamic_blocked=[[], [(2, 2)]])
        self.assertEqual(result, {"path": [(1, 1)], "cost": 0,
                                  "expanded": 1})

    def test_dynamic_replay_round_trip(self):
        frames = [[(3, 3)], [], [(2, 0)]]
        result = plan_any(4, 4, {(1, 1)}, (0, 0), [(3, 3), (0, 3)],
                          dynamic_blocked=frames)
        self.assertIsNotNone(result["path"])
        check = replay(4, 4, {(1, 1)}, (0, 0), result["path"][-1],
                       result["path"], dynamic_blocked=frames)
        self.assertEqual(check, {"valid": True, "cost": result["cost"],
                                 "steps": len(result["path"]) - 1})


class PlanAnyBruteForceTest(unittest.TestCase):
    def test_static_random_grids(self):
        rng = random.Random(20261004)
        for trial in range(60):
            width = rng.randint(2, 4)
            height = rng.randint(2, 4)
            cells = [(x, y) for x in range(width) for y in range(height)]
            start = rng.choice(cells)
            rest = [c for c in cells if c != start]
            rng.shuffle(rest)
            blocked = set(rest[:rng.randint(0, len(rest) // 3)])
            free = [c for c in rest if c not in blocked]
            goals = [start] if rng.random() < 0.2 else \
                rng.sample(free, min(len(free), rng.randint(1, 3)))
            if rng.random() < 0.5:
                costs = None
            else:
                costs = tuple(
                    tuple(rng.randint(1, 3) for _ in range(width))
                    for _ in range(height)
                )
            expected = brute_force_static(width, height, blocked, start,
                                          goals, costs)
            result = plan_any(width, height, blocked, start, goals,
                              costs=costs, trace=rng.random() < 0.5)
            if expected is None:
                self.assertIsNone(result["path"], msg=(trial, goals))
                self.assertIsNone(result["cost"])
            else:
                cost, _, _, path = expected
                self.assertEqual(tuple(result["path"]), path,
                                 msg=(trial, goals, costs))
                self.assertEqual(result["cost"], cost)
                check = replay(width, height, blocked, start,
                               result["path"][-1], result["path"],
                               costs=costs)
                self.assertEqual(
                    check, {"valid": True, "cost": cost,
                            "steps": len(path) - 1}, msg=(trial, goals))
            self.assertEqual(result["expanded"],
                             len(result.get("expanded_nodes",
                                            range(result["expanded"]))))

    def test_dynamic_random_grids(self):
        rng = random.Random(20261005)
        for trial in range(40):
            width = rng.randint(2, 3)
            height = rng.randint(2, 3)
            cells = [(x, y) for x in range(width) for y in range(height)]
            start = rng.choice(cells)
            rest = [c for c in cells if c != start]
            rng.shuffle(rest)
            blocked = set(rest[:rng.randint(0, len(rest) // 4)])
            free = [c for c in rest if c not in blocked]
            goals = rng.sample(free, min(len(free), rng.randint(1, 2)))
            frame_count = rng.randint(1, 3)
            frames = []
            for t in range(frame_count):
                pool = cells if t > 0 else [c for c in cells if c != start]
                frames.append({c for c in pool if rng.random() < 0.25})
            costs = None if rng.random() < 0.5 else tuple(
                tuple(rng.randint(1, 3) for _ in range(width))
                for _ in range(height)
            )
            expected = brute_force_dynamic(width, height, blocked, frames,
                                           start, goals, costs)
            result = plan_any(width, height, blocked, start, goals,
                              costs=costs, dynamic_blocked=frames)
            if expected is None:
                self.assertIsNone(result["path"], msg=(trial, goals))
                self.assertIsNone(result["cost"])
            else:
                cost, _, _, path = expected
                self.assertEqual(tuple(result["path"]), path,
                                 msg=(trial, goals, frames))
                self.assertEqual(result["cost"], cost)
                check = replay(width, height, blocked, start,
                               result["path"][-1], result["path"],
                               costs=costs, dynamic_blocked=frames)
                self.assertEqual(
                    check, {"valid": True, "cost": cost,
                            "steps": len(path) - 1}, msg=(trial, goals))


if __name__ == "__main__":
    unittest.main()
