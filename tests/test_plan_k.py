import unittest

import app
from app import plan_k, replay


class PlanKResultTest(unittest.TestCase):
    def test_result_keys(self):
        result = plan_k(3, 3, set(), (0, 0), (2, 2), 2)
        self.assertEqual(set(result), {"paths", "costs", "expanded"})

    def test_ranked_by_cost_then_lexicographic(self):
        # Open 3x3 corner to corner: six 4-step monotone routes, ranked
        # by the complete coordinate sequence.
        result = plan_k(3, 3, set(), (0, 0), (2, 2), 3)
        self.assertEqual(result["costs"], [4, 4, 4])
        self.assertEqual(result["paths"], [
            [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)],
            [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)],
            [(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)],
        ])

    def test_k_larger_than_route_count(self):
        # Only six simple routes exist between opposite corners.
        result = plan_k(3, 3, set(), (0, 0), (2, 2), 10)
        self.assertEqual(len(result["paths"]), 10)
        self.assertEqual(result["costs"],
                         [4, 4, 4, 4, 4, 4, 6, 6, 6, 6])
        pairs = list(zip(result["costs"], result["paths"]))
        self.assertEqual(pairs, sorted(pairs))

    def test_k_caps_the_result(self):
        result = plan_k(3, 3, set(), (0, 0), (2, 2), 1)
        self.assertEqual(result["paths"],
                         [[(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]])
        self.assertEqual(result["costs"], [4])

    def test_paths_are_unique(self):
        result = plan_k(3, 3, set(), (0, 0), (2, 2), 10)
        self.assertEqual(len({tuple(p) for p in result["paths"]}),
                         len(result["paths"]))

    def test_costs_match_paths_itemwise(self):
        result = plan_k(3, 3, set(), (0, 0), (2, 2), 10)
        self.assertEqual(len(result["paths"]), len(result["costs"]))
        for path, cost in zip(result["paths"], result["costs"]):
            self.assertEqual(cost, len(path) - 1)  # unit costs

    def test_unreachable(self):
        blocked = {(1, y) for y in range(3)}
        result = plan_k(3, 3, blocked, (0, 0), (2, 2), 4)
        self.assertEqual(result["paths"], [])
        self.assertEqual(result["costs"], [])
        self.assertEqual(result["expanded"], 3)  # left column closed

    def test_start_equals_goal(self):
        result = plan_k(3, 3, set(), (1, 1), (1, 1), 5)
        self.assertEqual(result["paths"], [[(1, 1)]])
        self.assertEqual(result["costs"], [0])
        self.assertEqual(result["expanded"], 1)

    def test_expanded_counts_closed_candidates(self):
        # A larger k never closes fewer candidates: determining more
        # routes requires closing at least as many route candidates.
        one = plan_k(3, 3, set(), (0, 0), (2, 2), 1)
        ten = plan_k(3, 3, set(), (0, 0), (2, 2), 10)
        self.assertEqual(one["expanded"], 19)
        self.assertGreaterEqual(ten["expanded"], one["expanded"])

    def test_no_repeated_coordinates(self):
        result = plan_k(4, 4, {(1, 1)}, (0, 0), (3, 3), 8)
        for path in result["paths"]:
            self.assertEqual(len(set(path)), len(path))

    def test_costs_matrix(self):
        costs = [[1, 9, 1], [1, 9, 1], [1, 1, 1]]
        result = plan_k(3, 3, set(), (0, 0), (2, 0), 3, costs=costs)
        self.assertEqual(result["costs"], [6, 10, 12])
        self.assertEqual(result["paths"][0],
                         [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2),
                          (2, 1), (2, 0)])

    def test_dynamic_blocked(self):
        # Frame 1 seals the direct corridor; the cheapest route detours.
        frames = [[(1, 0)]]
        result = plan_k(3, 3, set(), (0, 0), (2, 0), 3,
                        dynamic_blocked=frames)
        self.assertEqual(result["costs"], [4, 6, 6])
        self.assertEqual(result["paths"][0],
                         [(0, 0), (0, 1), (1, 1), (2, 1), (2, 0)])

    def test_dynamic_last_frame_persists(self):
        # The single frame keeps blocking (1, 0) at every later index.
        frames = [[(1, 0)]]
        result = plan_k(3, 1, set(), (0, 0), (2, 0), 2,
                        dynamic_blocked=frames)
        self.assertEqual(result["paths"], [])
        self.assertEqual(result["costs"], [])

    def test_replay_verifies_every_route(self):
        costs = [[1, 2, 1], [3, 1, 2], [1, 1, 4]]
        frames = [[(1, 1)], [(2, 1)]]
        for kwargs in ({}, {"costs": costs},
                       {"dynamic_blocked": frames},
                       {"costs": costs, "dynamic_blocked": frames}):
            result = plan_k(3, 3, {(0, 2)}, (0, 0), (2, 2), 4, **kwargs)
            for path, cost in zip(result["paths"], result["costs"]):
                rep = replay(3, 3, {(0, 2)}, (0, 0), (2, 2), path,
                             **kwargs)
                self.assertEqual(
                    rep,
                    {"valid": True, "cost": cost, "steps": len(path) - 1},
                )

    def test_result_independent_of_input_order(self):
        cells = [(x, 1) for x in range(5) if x != 2]
        forward = plan_k(5, 3, cells, (0, 0), (4, 2), 4)
        reverse = plan_k(5, 3, list(reversed(cells)), (0, 0), (4, 2), 4)
        shuffled = plan_k(5, 3, [cells[2], cells[0], cells[3], cells[1]],
                          (0, 0), (4, 2), 4)
        self.assertEqual(forward, reverse)
        self.assertEqual(forward, shuffled)
        frames_a = [[(1, 0), (0, 1)], [(2, 2)]]
        frames_b = [[(0, 1), (1, 0)], [(2, 2)]]
        self.assertEqual(
            plan_k(3, 3, set(), (0, 0), (2, 2), 3, dynamic_blocked=frames_a),
            plan_k(3, 3, set(), (0, 0), (2, 2), 3, dynamic_blocked=frames_b),
        )

    def test_determinism(self):
        first = plan_k(4, 4, {(1, 1), (2, 2)}, (0, 0), (3, 3), 5)
        for _ in range(10):
            self.assertEqual(
                plan_k(4, 4, {(1, 1), (2, 2)}, (0, 0), (3, 3), 5), first)


class PlanKValidationTest(unittest.TestCase):
    def test_k_type_errors(self):
        for bad in ("2", 2.0, None, True, [2], (2,)):
            with self.assertRaises(TypeError, msg=f"k={bad!r}"):
                plan_k(3, 3, set(), (0, 0), (2, 2), bad)

    def test_k_value_errors(self):
        for bad in (0, -1, -100):
            with self.assertRaises(ValueError, msg=f"k={bad!r}"):
                plan_k(3, 3, set(), (0, 0), (2, 2), bad)

    def test_shared_checks_run_before_k(self):
        # A grid violation wins over a malformed k (shared checks first).
        with self.assertRaises(ValueError):  # non-positive width
            plan_k(0, 3, set(), (0, 0), (2, 2), "bad")
        with self.assertRaises(TypeError):  # bad start coordinate
            plan_k(3, 3, set(), "x", (2, 2), 0)
        with self.assertRaises(ValueError):  # goal on an obstacle
            plan_k(3, 3, {(2, 2)}, (0, 0), (2, 2), 0)
        with self.assertRaises(ValueError):  # bad costs shape
            plan_k(3, 3, set(), (0, 0), (2, 2), 0, costs=[[1, 1]])
        with self.assertRaises(ValueError):  # start blocked at frame 0
            plan_k(3, 3, set(), (0, 0), (2, 2), 0,
                   dynamic_blocked=[[(0, 0)]])

    def test_grid_validation_matches_plan(self):
        with self.assertRaises(TypeError):
            plan_k(3, 3, set(), (0, 0), (2, 2), 1, costs="nope")
        with self.assertRaises(TypeError):
            plan_k(3, 3, set(), (0, 0), (2, 2), 1,
                   dynamic_blocked=[[(0, 0, 0)]])
        with self.assertRaises(ValueError):
            plan_k(3, 3, {(9, 9)}, (0, 0), (2, 2), 1)


if __name__ == "__main__":
    unittest.main()
