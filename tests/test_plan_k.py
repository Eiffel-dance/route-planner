import itertools
import unittest

from app import plan, plan_k, replay


class PlanKBasicTest(unittest.TestCase):
    def test_result_shape(self):
        result = plan_k(3, 3, set(), (0, 0), (2, 2), 3)
        self.assertEqual(set(result), {"paths", "costs", "expanded"})
        self.assertEqual(len(result["paths"]), len(result["costs"]))
        self.assertLessEqual(len(result["paths"]), 3)

    def test_open_grid_top_six_ordered_by_cost_then_sequence(self):
        # Empty 3x3: exactly six shortest routes cost 4, and they sort by
        # the complete coordinate sequence. Longer simple paths (cost 6
        # and 8) are distinct candidates too and follow them when a larger
        # k is requested.
        result = plan_k(3, 3, set(), (0, 0), (2, 2), 6)
        paths = result["paths"]
        self.assertEqual(result["costs"], [4] * 6)
        self.assertEqual(paths, sorted(paths, key=lambda p: (len(p), p)))
        self.assertEqual(
            paths[0],
            [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)],
        )
        # The first route is exactly what ``plan`` chooses.
        single = plan(3, 3, set(), (0, 0), (2, 2))
        self.assertEqual(paths[0], single["path"])
        self.assertEqual(result["costs"][0], single["cost"])
        # Asking for more keeps the cost-4 prefix and appends longer
        # routes in strict ascending (cost, sequence) order.
        larger = plan_k(3, 3, set(), (0, 0), (2, 2), 20)
        self.assertEqual(larger["paths"][:6], paths)
        keys = list(zip(larger["costs"],
                        [tuple(p) for p in larger["paths"]]))
        self.assertEqual(keys, sorted(keys))
        self.assertIn(6, larger["costs"])

    def test_k_limits_the_count(self):
        for k in (1, 2, 3, 6):
            result = plan_k(3, 3, set(), (0, 0), (2, 2), k)
            self.assertEqual(len(result["paths"]), k)

    def test_corridor_has_single_route_regardless_of_k(self):
        result = plan_k(5, 1, set(), (0, 0), (4, 0), 100)
        self.assertEqual(
            result,
            {"paths": [[(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]],
             "costs": [4], "expanded": 5},
        )

    def test_routes_are_unique_simple_four_connected_paths(self):
        result = plan_k(4, 3, {(1, 1)}, (0, 0), (3, 2), 20)
        sequences = [tuple(p) for p in result["paths"]]
        self.assertEqual(len(set(sequences)), len(sequences))
        for path, cost in zip(result["paths"], result["costs"]):
            self.assertEqual(path[0], (0, 0))
            self.assertEqual(path[-1], (3, 2))
            self.assertEqual(len(path), len(set(path)))
            for a, b in zip(path, path[1:]):
                self.assertEqual(abs(a[0] - b[0]) + abs(a[1] - b[1]), 1)
            self.assertEqual(cost, len(path) - 1)

    def test_weighted_costs_order_routes_by_entered_cell_sum(self):
        costs = [[1, 1], [3, 1]]
        result = plan_k(2, 2, set(), (0, 0), (1, 1), 4, costs=costs)
        self.assertEqual(
            result["paths"],
            [[(0, 0), (1, 0), (1, 1)],
             [(0, 0), (0, 1), (1, 1)]],
        )
        self.assertEqual(result["costs"], [2, 4])

    def test_start_equals_goal_is_single_point_route(self):
        self.assertEqual(
            plan_k(3, 3, set(), (1, 1), (1, 1), 5),
            {"paths": [[(1, 1)]], "costs": [0], "expanded": 1},
        )

    def test_unreachable_returns_empty_arrays_with_expanded(self):
        wall = {(1, y) for y in range(3)}
        result = plan_k(3, 3, wall, (0, 0), (2, 2), 5)
        self.assertEqual(result["paths"], [])
        self.assertEqual(result["costs"], [])
        # The three cells of the sealed left column are still closed.
        self.assertEqual(result["expanded"], 3)

    def test_expansion_stops_as_soon_as_k_goals_are_closed(self):
        first = plan_k(3, 3, set(), (0, 0), (2, 2), 1)
        second = plan_k(3, 3, set(), (0, 0), (2, 2), 2)
        exhausted = plan_k(3, 3, set(), (0, 0), (2, 2), 100)
        self.assertLess(first["expanded"], second["expanded"])
        self.assertLess(second["expanded"], exhausted["expanded"])
        self.assertEqual(first["paths"], exhausted["paths"][:1])

    def test_input_order_invariance(self):
        cells = [(x, 1) for x in range(5) if x != 2]
        orderings = (
            cells,
            list(reversed(cells)),
            [cells[2], cells[0], cells[3], cells[1]],
        )
        results = [plan_k(5, 3, blocked, (0, 0), (4, 2), 6)
                   for blocked in orderings]
        for other in results[1:]:
            self.assertEqual(other, results[0])

    def test_every_returned_route_replays(self):
        blocked = {(2, 0), (2, 1), (2, 2), (0, 2), (4, 1)}
        result = plan_k(6, 4, blocked, (0, 0), (5, 3), 8)
        for path, cost in zip(result["paths"], result["costs"]):
            checked = replay(6, 4, blocked, (0, 0), (5, 3), path)
            self.assertTrue(checked["valid"])
            self.assertEqual(checked["cost"], cost)
            self.assertEqual(checked["steps"], len(path) - 1)


class PlanKDynamicTest(unittest.TestCase):
    FRAMES = [
        [(0, 1), (0, 2)],
        [(0, 1), (1, 2), (2, 0), (2, 2)],
        [(2, 1), (2, 2)],
        [(0, 0), (1, 2), (2, 1), (2, 2)],
        [],
        [(0, 1)],
    ]

    def test_dynamic_routes_order_and_replay(self):
        result = plan_k(3, 3, set(), (1, 0), (2, 2), 6,
                        dynamic_blocked=self.FRAMES)
        keys = list(zip(result["costs"],
                        [tuple(p) for p in result["paths"]]))
        self.assertEqual(keys, sorted(keys))
        sequences = [tuple(p) for p in result["paths"]]
        self.assertEqual(len(set(sequences)), len(sequences))
        for path, cost in zip(result["paths"], result["costs"]):
            self.assertEqual(len(path), len(set(path)))
            checked = replay(3, 3, set(), (1, 0), (2, 2), path,
                             dynamic_blocked=self.FRAMES)
            self.assertTrue(checked["valid"])
            self.assertEqual(checked["cost"], cost)
            self.assertEqual(checked["steps"], len(path) - 1)
        # The best route is the one ``plan`` returns.
        best = plan(3, 3, set(), (1, 0), (2, 2),
                    dynamic_blocked=self.FRAMES)
        self.assertEqual(result["paths"][0], best["path"])
        self.assertEqual(result["costs"][0], best["cost"])

    def test_distinct_histories_to_same_cell_are_both_counted(self):
        # (2,0) is blocked at frame 1 but free afterward: the direct
        # arrival is infeasible, the detour reaches it later.
        frames = [[], [(2, 0)], []]
        result = plan_k(3, 2, set(), (0, 0), (2, 1), 10,
                        dynamic_blocked=frames)
        self.assertTrue(result["paths"])
        self.assertGreaterEqual(result["expanded"], 1)
        for path, cost in zip(result["paths"], result["costs"]):
            checked = replay(3, 2, set(), (0, 0), (2, 1), path,
                             dynamic_blocked=frames)
            self.assertTrue(checked["valid"], path)
            self.assertEqual(checked["cost"], cost)

    def test_persistent_last_frame_can_make_goal_unreachable(self):
        # (2,0) stays blocked from frame 1 on (the last frame persists);
        # the only move into it is infeasible at every arrival time.
        result = plan_k(3, 1, set(), (0, 0), (2, 0), 4,
                        dynamic_blocked=[[], [(2, 0)]])
        self.assertEqual(result["paths"], [])
        self.assertEqual(result["costs"], [])
        # start and the intermediate (1,0) still actually close.
        self.assertEqual(result["expanded"], 2)

    def test_frame_iteration_order_invariance(self):
        orderings = [
            self.FRAMES,
            [list(reversed(f)) for f in self.FRAMES],
            [set(f) for f in self.FRAMES],
            [tuple(f) + (f[0],) if f else () for f in self.FRAMES],
        ]
        results = [plan_k(3, 3, [(2, 0), (2, 0)], (1, 0), (2, 2), 6,
                          dynamic_blocked=f)
                   for f in orderings]
        for other in results[1:]:
            self.assertEqual(other, results[0])

    def test_empty_frames_equivalent_to_static(self):
        static = plan_k(3, 3, set(), (0, 0), (2, 2), 4)
        for frames in (None, [], ()):
            self.assertEqual(
                plan_k(3, 3, set(), (0, 0), (2, 2), 4,
                       dynamic_blocked=frames),
                static,
            )

    def test_start_equals_goal_with_frames(self):
        frames = [[], [(1, 1)], [(1, 1)]]
        self.assertEqual(
            plan_k(3, 3, set(), (1, 1), (1, 1), 3,
                   dynamic_blocked=frames),
            {"paths": [[(1, 1)]], "costs": [0], "expanded": 1},
        )


class PlanKValidationTest(unittest.TestCase):
    def test_k_type_errors(self):
        for bad in ("2", 1.5, True, False, [2], (2,), None, 2.0):
            with self.assertRaises(TypeError, msg=f"k={bad!r}"):
                plan_k(3, 3, set(), (0, 0), (2, 2), bad)

    def test_k_non_positive_is_value_error(self):
        for bad in (0, -1, -100):
            with self.assertRaises(ValueError, msg=f"k={bad!r}"):
                plan_k(3, 3, set(), (0, 0), (2, 2), bad)

    def test_shared_plan_checks_still_apply(self):
        with self.assertRaises(TypeError):
            plan_k("3", 3, set(), (0, 0), (2, 2), 1)
        with self.assertRaises(ValueError):
            plan_k(0, 3, set(), (0, 0), (2, 2), 1)
        with self.assertRaises(TypeError):
            plan_k(3, 3, 42, (0, 0), (2, 2), 1)
        with self.assertRaises(TypeError):
            plan_k(3, 3, set(), (True, 0), (2, 2), 1)
        with self.assertRaises(ValueError):
            plan_k(3, 3, set(), (0, 0), (9, 9), 1)
        with self.assertRaises(ValueError):
            plan_k(3, 3, {(0, 0)}, (0, 0), (2, 2), 1)
        with self.assertRaises(ValueError):
            plan_k(3, 3, set(), (0, 0), (2, 2), 1, costs=[[1]])
        with self.assertRaises(TypeError):
            plan_k(3, 3, set(), (0, 0), (2, 2), 1, costs="x")
        with self.assertRaises(TypeError):
            plan_k(3, 3, set(), (0, 0), (2, 2), 1,
                   costs=[[1, 1, True], [1, 1, 1], [1, 1, 1]])
        with self.assertRaises(ValueError):
            plan_k(3, 3, set(), (0, 0), (2, 2), 1,
                   costs=[[0, 1, 1], [1, 1, 1], [1, 1, 1]])
        with self.assertRaises(TypeError):
            plan_k(3, 3, set(), (0, 0), (2, 2), 1, dynamic_blocked=42)
        with self.assertRaises(ValueError):
            plan_k(3, 3, set(), (0, 0), (2, 2), 1,
                   dynamic_blocked=[[(0, 0)]])

    def test_shared_checks_precede_the_k_check(self):
        # A bad k must not mask an earlier shared-check failure.
        with self.assertRaises(ValueError):  # width, not k's TypeError
            plan_k(0, 3, set(), (0, 0), (2, 2), 1.5)
        with self.assertRaises(TypeError):  # blocked structure
            plan_k(3, 3, 42, (0, 0), (2, 2), 1.5)
        with self.assertRaises(ValueError):  # goal out of bounds
            plan_k(3, 3, set(), (0, 0), (9, 9), 1.5)
        with self.assertRaises(ValueError):  # start on an obstacle
            plan_k(3, 3, {(0, 0)}, (0, 0), (2, 2), 1.5)
        with self.assertRaises(ValueError):  # costs shape
            plan_k(3, 3, set(), (0, 0), (2, 2), 0, costs=[[1]])
        with self.assertRaises(ValueError):  # frame-0 start block
            plan_k(3, 3, set(), (0, 0), (2, 2), 0,
                   dynamic_blocked=[[(0, 0)]])
        # Once the shared checks pass, the k error surfaces.
        with self.assertRaises(ValueError):
            plan_k(3, 3, set(), (0, 0), (2, 2), 0)
        with self.assertRaises(TypeError):
            plan_k(3, 3, set(), (0, 0), (2, 2), True)

    def test_existing_entry_points_unchanged(self):
        self.assertEqual(
            set(plan(3, 3, set(), (0, 0), (2, 2))),
            {"path", "cost", "expanded"},
        )
        # plan_k accepts no trace/budget/snapshot/max_cost options.
        for kwargs in ({"trace": True}, {"max_expanded": 1},
                       {"snapshot": True}, {"max_cost": 1}):
            with self.assertRaises(TypeError, msg=kwargs):
                plan_k(3, 3, set(), (0, 0), (2, 2), 1, **kwargs)


if __name__ == '__main__':
    unittest.main()
