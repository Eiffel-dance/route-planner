import json
import unittest

import app
from app import distance_field, plan, replay


class DistanceFieldUnitCostTest(unittest.TestCase):
    def test_unit_costs_match_manhattan(self):
        result = distance_field(4, 3, set(), (3, 2))
        self.assertEqual(set(result), {"distances", "expanded"})
        expected = [
            [5, 4, 3, 2],
            [4, 3, 2, 1],
            [3, 2, 1, 0],
        ]
        self.assertEqual(result["distances"], expected)
        self.assertEqual(result["expanded"], 12)

    def test_goal_is_zero_and_counted(self):
        result = distance_field(3, 3, set(), (1, 1))
        self.assertEqual(result["distances"][1][1], 0)
        # Every cell is reachable, so every cell is closed once.
        self.assertEqual(result["expanded"], 9)

    def test_obstacle_cells_are_none(self):
        blocked = {(1, 0)}
        result = distance_field(3, 2, blocked, (2, 1))
        self.assertIsNone(result["distances"][0][1])
        self.assertEqual(result["distances"][1][2], 0)
        # (0, 0) detours around the obstacle through the bottom row.
        self.assertEqual(result["distances"][0][0], 3)
        self.assertEqual(result["distances"][1][0], 2)

    def test_isolated_region_is_none(self):
        # Wall at x=1 seals the left column off from the goal.
        blocked = {(1, y) for y in range(3)}
        result = distance_field(3, 3, blocked, (2, 2))
        for y in range(3):
            self.assertIsNone(result["distances"][y][0])
            self.assertIsNone(result["distances"][y][1])
        self.assertEqual(result["distances"][2][2], 0)
        self.assertEqual(result["distances"][0][2], 2)
        self.assertEqual(result["distances"][1][2], 1)
        # Only the right column is reachable: 3 closed cells.
        self.assertEqual(result["expanded"], 3)

    def test_goal_only_reachable_cell(self):
        # The goal is sealed in by obstacles on every side.
        blocked = {(0, 1), (1, 0), (1, 2), (2, 1)}
        result = distance_field(3, 3, blocked, (1, 1))
        self.assertEqual(result["distances"][1][1], 0)
        self.assertEqual(result["expanded"], 1)
        reachable = [cell for row in result["distances"] for cell in row
                     if cell is not None]
        self.assertEqual(reachable, [0])


class DistanceFieldWeightedTest(unittest.TestCase):
    def test_weighted_costs_entering_cell_semantics(self):
        # Entering (1, 0) costs 9, so the cheapest route from (0, 0) to
        # (2, 0) detours through the bottom row: 1 + 1 + 1 + 1 = 4.
        costs = [[1, 9, 1], [1, 1, 1]]
        result = distance_field(3, 2, set(), (2, 0), costs=costs)
        self.assertEqual(result["distances"][0][0], 4)
        self.assertEqual(result["distances"][0][1], 1)  # enters the goal
        self.assertEqual(result["distances"][0][2], 0)
        self.assertEqual(result["distances"][1][2], 1)
        self.assertEqual(result["distances"][1][1], 2)
        self.assertEqual(result["distances"][1][0], 3)

    def test_goal_cost_not_counted_for_itself(self):
        costs = [[5, 5], [5, 7]]
        result = distance_field(2, 2, set(), (1, 1), costs=costs)
        self.assertEqual(result["distances"][1][1], 0)
        # Neighbors pay exactly the goal's entering cost.
        self.assertEqual(result["distances"][1][0], 7)
        self.assertEqual(result["distances"][0][1], 7)
        self.assertEqual(result["distances"][0][0], 12)

    def test_matches_plan_cost_for_every_reachable_start(self):
        costs = [[3, 1, 4, 1], [5, 9, 2, 6], [5, 3, 5, 8]]
        blocked = {(1, 1)}
        goal = (3, 2)
        field = distance_field(4, 3, blocked, goal, costs=costs)
        for y in range(3):
            for x in range(4):
                if (x, y) in blocked:
                    continue
                result = plan(4, 3, blocked, (x, y), goal, costs=costs)
                self.assertEqual(field["distances"][y][x], result["cost"],
                                 f"start {(x, y)}")

    def test_cross_check_with_plan_and_replay(self):
        costs = [[2, 7, 6, 1], [4, 3, 8, 2], [1, 9, 5, 4]]
        blocked = {(2, 1)}
        goal = (0, 2)
        field = distance_field(4, 3, blocked, goal, costs=costs)
        for y in range(3):
            for x in range(4):
                start = (x, y)
                if start in blocked:
                    continue
                result = plan(4, 3, blocked, start, goal, costs=costs)
                self.assertEqual(field["distances"][y][x], result["cost"])
                check = replay(4, 3, blocked, start, goal, result["path"],
                               costs=costs)
                self.assertTrue(check["valid"])
                self.assertEqual(check["cost"], field["distances"][y][x])
                self.assertEqual(check["steps"], len(result["path"]) - 1)


class DistanceFieldDeterminismTest(unittest.TestCase):
    def test_duplicate_and_shuffled_blocked(self):
        cells = [(1, 1), (2, 2), (0, 2)]
        base = distance_field(4, 3, cells, (3, 0), trace=True)
        duplicated = distance_field(
            4, 3, cells + cells[::-1], (3, 0), trace=True)
        shuffled = distance_field(
            4, 3, [cells[2], cells[0], cells[1]], (3, 0), trace=True)
        self.assertEqual(base, duplicated)
        self.assertEqual(base, shuffled)

    def test_costs_input_order_irrelevant(self):
        rows = [[3, 1, 4], [1, 5, 9], [2, 6, 5]]
        base = distance_field(3, 3, set(), (2, 2), costs=rows, trace=True)
        as_tuples = distance_field(
            3, 3, set(), (2, 2),
            costs=tuple(tuple(row) for row in rows), trace=True)
        again = distance_field(3, 3, set(), (2, 2),
                               costs=[list(row) for row in rows], trace=True)
        self.assertEqual(base, as_tuples)
        self.assertEqual(base, again)

    def test_trace_stability(self):
        blocked = {(1, 0)}
        first = distance_field(4, 3, blocked, (3, 2), trace=True)
        second = distance_field(4, 3, blocked, (3, 2), trace=True)
        self.assertEqual(first, second)
        nodes = first["expanded_nodes"]
        self.assertEqual(first["expanded"], len(nodes))
        self.assertEqual(nodes[0], (3, 2))  # the goal closes first
        self.assertEqual(len(set(nodes)), len(nodes))  # closed once each
        # Closing order: non-decreasing distance, (x, y) lexicographic
        # tie-break.
        keys = [(first["distances"][y][x], x, y) for x, y in nodes]
        self.assertEqual(keys, sorted(keys))

    def test_trace_omitted_by_default(self):
        self.assertNotIn("expanded_nodes", distance_field(2, 2, set(), (0, 0)))
        self.assertNotIn("expanded_nodes",
                         distance_field(2, 2, set(), (0, 0), trace=False))

    def test_trace_records_coordinate_pairs(self):
        result = distance_field(3, 2, set(), (2, 1), trace=True)
        for node in result["expanded_nodes"]:
            self.assertIsInstance(node, tuple)
            self.assertEqual(len(node), 2)

    def test_result_is_json_serializable(self):
        result = distance_field(3, 3, {(1, 1)}, (2, 2), trace=True)
        encoded = json.dumps(result)
        decoded = json.loads(encoded)
        self.assertEqual(decoded["distances"], result["distances"])
        self.assertEqual(decoded["expanded"], result["expanded"])

    def test_matrix_shape_and_non_negative(self):
        result = distance_field(5, 4, {(2, 2)}, (0, 0))
        self.assertEqual(len(result["distances"]), 4)
        for row in result["distances"]:
            self.assertEqual(len(row), 5)
            for cell in row:
                if cell is not None:
                    self.assertGreaterEqual(cell, 0)


class DistanceFieldValidationTest(unittest.TestCase):
    def test_dimension_errors(self):
        with self.assertRaises(TypeError):
            distance_field("4", 3, set(), (0, 0))
        with self.assertRaises(TypeError):
            distance_field(True, 3, set(), (0, 0))
        with self.assertRaises(ValueError):
            distance_field(0, 3, set(), (0, 0))
        with self.assertRaises(ValueError):
            distance_field(4, -1, set(), (0, 0))

    def test_goal_structure_errors(self):
        with self.assertRaises(TypeError):
            distance_field(3, 3, set(), "00")
        with self.assertRaises(TypeError):
            distance_field(3, 3, set(), (0, 0, 0))
        with self.assertRaises(TypeError):
            distance_field(3, 3, set(), (0, 0.5))

    def test_goal_out_of_bounds(self):
        with self.assertRaises(ValueError):
            distance_field(3, 3, set(), (3, 0))
        with self.assertRaises(ValueError):
            distance_field(3, 3, set(), (0, -1))

    def test_goal_on_obstacle(self):
        with self.assertRaises(ValueError):
            distance_field(3, 3, {(1, 1)}, (1, 1))

    def test_blocked_errors(self):
        with self.assertRaises(TypeError):
            distance_field(3, 3, "not-cells", (0, 0))
        with self.assertRaises(TypeError):
            distance_field(3, 3, [(0, 0, 0)], (0, 0))
        with self.assertRaises(ValueError):
            distance_field(3, 3, [(5, 0)], (0, 0))

    def test_costs_errors(self):
        with self.assertRaises(TypeError):
            distance_field(2, 2, set(), (0, 0), costs="costs")
        with self.assertRaises(ValueError):
            distance_field(2, 2, set(), (0, 0), costs=[[1, 1]])
        with self.assertRaises(ValueError):
            distance_field(2, 2, set(), (0, 0), costs=[[1], [1, 1]])
        with self.assertRaises(TypeError):
            distance_field(2, 2, set(), (0, 0), costs=[[1, True], [1, 1]])
        with self.assertRaises(ValueError):
            distance_field(2, 2, set(), (0, 0), costs=[[1, 0], [1, 1]])

    def test_trace_must_be_bool(self):
        with self.assertRaises(TypeError):
            distance_field(2, 2, set(), (0, 0), trace=1)
        with self.assertRaises(TypeError):
            distance_field(2, 2, set(), (0, 0), trace="yes")

    def test_no_extra_parameters(self):
        # The offline entry point accepts no start, dynamic obstacles,
        # budget, snapshot or cost-limit arguments.
        with self.assertRaises(TypeError):
            distance_field(2, 2, set(), (0, 0), max_expanded=1)
        with self.assertRaises(TypeError):
            distance_field(2, 2, set(), (0, 0), dynamic_blocked=[])


class DistanceFieldPublicSurfaceTest(unittest.TestCase):
    def test_exported(self):
        self.assertIn("distance_field", app.__all__)
        self.assertIs(app.distance_field, distance_field)

    def test_existing_entry_points_untouched(self):
        # The pre-existing public behavior is unchanged.
        result = plan(3, 3, set(), (0, 0), (2, 2))
        self.assertEqual(result, {"path": [(0, 0), (0, 1), (0, 2),
                                           (1, 2), (2, 2)],
                                  "cost": 4, "expanded": 5})


if __name__ == "__main__":
    unittest.main()
