import unittest

from app import plan, plan_any, replay


class PlanAnyBasicTest(unittest.TestCase):
    def test_picks_reachable_candidate(self):
        blocked = {(2, 0), (2, 1), (2, 2)}
        result = plan_any(6, 4, blocked, (0, 0), [(5, 3), (5, 2)],
                          trace=True)
        path = result["path"]
        self.assertEqual(path[0], (0, 0))
        self.assertIn(path[-1], {(5, 3), (5, 2)})
        self.assertEqual(result["cost"], len(path) - 1)
        self.assertEqual(len(path), len(set(path)))  # no repeated cells
        self.assertNotIn("status", result)
        self.assertEqual(set(result),
                         {"path", "cost", "expanded", "expanded_nodes"})
        nodes = result["expanded_nodes"]
        self.assertTrue(all(isinstance(p, tuple) and len(p) == 2 for p in nodes))

    def test_minimum_cost_wins_over_distance(self):
        # (1, 0) is one unit away but costs 9 to enter; the farther goal
        # (2, 0) is reached through cheap cells.
        costs = [[1, 9, 1], [1, 1, 1]]
        result = plan_any(3, 2, set(), (0, 0), [(1, 0), (2, 0)],
                          costs=costs)
        self.assertEqual(result["path"][-1], (2, 0))
        self.assertEqual(result["cost"],
                         sum(costs[y][x] for x, y in result["path"][1:]))

    def test_equal_cost_prefers_smaller_goal_coordinate(self):
        # Open 3x3, unit costs: both candidates are two steps away at cost 2.
        result = plan_any(3, 3, set(), (0, 0), [(2, 0), (0, 2)])
        self.assertEqual(result["path"][-1], (0, 2))  # (0, 2) < (2, 0)
        self.assertEqual(result["path"], [(0, 0), (0, 1), (0, 2)])
        # Goal order in the input changes nothing.
        reversed_result = plan_any(3, 3, set(), (0, 0), [(0, 2), (2, 0)])
        self.assertEqual(reversed_result, result)

    def test_equal_cost_path_lexicographic_tie_break(self):
        # Single candidate on the open grid: all shortest routes cost 4;
        # the lexicographically smallest coordinate sequence wins.
        result = plan_any(3, 3, set(), (0, 0), [(2, 2)])
        self.assertEqual(
            result["path"],
            [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)],
        )

    def test_duplicate_candidates_merge(self):
        once = plan_any(5, 3, set(), (0, 0), [(4, 0), (4, 2)])
        repeated = plan_any(5, 3, set(), (0, 0),
                            [(4, 0), (4, 2), (4, 0), (4, 2), (4, 0)],
                            trace=True)
        self.assertEqual(repeated["path"][-1], once["path"][-1])
        self.assertEqual(repeated["cost"], once["cost"])

    def test_input_permutation_and_blocked_order_invariance(self):
        cells = [(x, 1) for x in range(5) if x != 2]
        goals = [(4, 2), (4, 0), (0, 2)]
        orderings = (
            (cells, goals),
            (list(reversed(cells)), list(reversed(goals))),
            ([cells[2], cells[0], cells[3], cells[1]],
             [goals[2], goals[0], goals[1], goals[0]]),
        )
        results = [plan_any(5, 3, blocked, (0, 0), gl, trace=True)
                   for blocked, gl in orderings]
        for other in results[1:]:
            self.assertEqual(other, results[0])

    def test_start_in_goals_is_single_point_route(self):
        result = plan_any(3, 3, set(), (1, 1),
                          [(2, 2), (1, 1), (0, 0)], trace=True)
        self.assertEqual(result, {"path": [(1, 1)], "cost": 0,
                                  "expanded": 1,
                                  "expanded_nodes": [(1, 1)]})

    def test_unreachable_when_every_goal_sealed(self):
        wall = {(1, y) for y in range(3)}
        result = plan_any(3, 3, wall, (0, 0), [(2, 0), (2, 2)], trace=True)
        self.assertIsNone(result["path"])
        self.assertIsNone(result["cost"])
        self.assertEqual(result["expanded"], 3)
        self.assertEqual(result["expanded_nodes"],
                         [(0, 0), (0, 1), (0, 2)])
        self.assertNotIn("status", result)

    def test_replay_verifies_result_using_path_endpoint(self):
        blocked = {(2, 0), (2, 1), (2, 2), (0, 2), (4, 1)}
        for goals in ([(5, 3)], [(5, 2), (5, 3), (1, 3)],
                      [(3, 1), (5, 3)]):
            planned = plan_any(6, 4, blocked, (0, 0), goals)
            goal = planned["path"][-1]
            checked = replay(6, 4, blocked, (0, 0), goal, planned["path"])
            self.assertTrue(checked["valid"])
            self.assertEqual(checked["cost"], planned["cost"])
            self.assertEqual(checked["steps"], len(planned["path"]) - 1)


class PlanAnyValidationTest(unittest.TestCase):
    def test_outer_non_sequence_is_type_error(self):
        for bad in (None, 42, 1.5, "ab", b"ab", {1, 2}, {"x": 1}, True):
            with self.assertRaises(TypeError, msg=f"goals={bad!r}"):
                plan_any(3, 3, set(), (0, 0), bad)

    def test_empty_sequence_is_value_error(self):
        for bad in ((), [], tuple()):
            with self.assertRaises(ValueError, msg=f"goals={bad!r}"):
                plan_any(3, 3, set(), (0, 0), bad)

    def test_element_structure_errors(self):
        for bad in (1, "ab", (1,), (1, 2, 3), (1.5, 2), (1, None),
                    (True, 0), None, {"x": 1, "y": 2}):
            with self.assertRaises(TypeError, msg=f"goal={bad!r}"):
                plan_any(3, 3, set(), (0, 0), [(2, 2), bad])

    def test_coordinate_bounds_errors(self):
        for bad in ((-1, 0), (0, -1), (3, 0), (0, 3), (10, 10)):
            with self.assertRaises(ValueError, msg=f"goal={bad!r}"):
                plan_any(3, 3, set(), (0, 0), [(2, 2), bad])

    def test_goal_on_static_obstacle_is_value_error(self):
        with self.assertRaises(ValueError):
            plan_any(3, 3, {(2, 2)}, (0, 0), [(1, 1), (2, 2)])
        # Goals appearing in dynamic frames are not statically invalid.
        result = plan_any(3, 3, set(), (0, 0), [(2, 2)],
                          dynamic_blocked=[[], [(2, 2)]])
        self.assertIsNone(result["path"])

    def test_validation_order_mirrors_plan(self):
        # width ValueError is reported before anything goals-specific.
        with self.assertRaises(ValueError):
            plan_any(0, 3, set(), (0, 0), "bad")
        # Bad goals structure precedes blocked normalization, as goal does.
        with self.assertRaises(TypeError):
            plan_any(3, 3, 42, (0, 0), [1])
        # Goal bounds precede the start-on-obstacle / costs / trace checks.
        with self.assertRaises(ValueError):
            plan_any(3, 3, {(0, 0)}, (0, 0), [(9, 9)])
        with self.assertRaises(ValueError):
            plan_any(3, 3, set(), (0, 0), [(9, 9)],
                     costs=[[1, 1], [1, 1], [1, 1]])
        with self.assertRaises(ValueError):
            plan_any(3, 3, set(), (0, 0), [(9, 9)], trace=1)
        # A goal-on-obstacle ValueError precedes the trace TypeError, just
        # like plan's goal check.
        with self.assertRaises(ValueError):
            plan_any(3, 3, {(2, 2)}, (0, 0), [(2, 2)], trace=1)
        # trace TypeError precedes the budget type error.
        with self.assertRaises(TypeError):
            plan_any(3, 3, set(), (0, 0), [(2, 2)], trace=1,
                     max_expanded="x")
        # Goal checks precede the budget validation.
        with self.assertRaises(ValueError):
            plan_any(3, 3, set(), (0, 0), [(9, 9)], max_expanded="x")
        with self.assertRaises(TypeError):
            plan_any(3, 3, set(), (0, 0), [1], max_expanded="x")
        # The frame-0 ValueError precedes the budget TypeError.
        with self.assertRaises(ValueError):
            plan_any(3, 3, set(), (0, 0), [(2, 2)],
                     dynamic_blocked=[[(0, 0)]], max_expanded="x")
        # Dynamic structure TypeError precedes the budget ValueError.
        with self.assertRaises(TypeError):
            plan_any(3, 3, set(), (0, 0), [(2, 2)],
                     dynamic_blocked=42, max_expanded=-1)

    def test_start_still_frame_zero_checked(self):
        with self.assertRaises(ValueError):
            plan_any(3, 3, set(), (0, 0), [(2, 2)],
                     dynamic_blocked=[[(0, 0)]])

    def test_existing_plan_and_replay_entry_points_unchanged(self):
        # plan keeps its exact positional signature: passing goals to it
        # is simply not supported; its documented result shape is intact.
        self.assertEqual(
            set(plan(3, 3, set(), (0, 0), (2, 2))),
            {"path", "cost", "expanded"},
        )


class PlanAnyDynamicTest(unittest.TestCase):
    FRAMES = [
        [(0, 1), (0, 2)],
        [(0, 1), (1, 2), (2, 0), (2, 2)],
        [(2, 1), (2, 2)],
        [(0, 0), (1, 2), (2, 1), (2, 2)],
        [],
        [(0, 1)],
    ]

    def test_dynamic_goal_blocked_at_early_frame_still_reachable_later(self):
        # (1, 0) is blocked at frame 1 but free from frame 2 on; the
        # one-step arrival is infeasible, the detour arrives at frame 3.
        frames = [[], [(1, 0)], []]
        result = plan_any(3, 2, set(), (0, 0), [(1, 0), (2, 0)],
                          dynamic_blocked=frames, trace=True)
        self.assertEqual(result["path"],
                         [(0, 0), (0, 1), (1, 1), (1, 0)])
        self.assertEqual(result["cost"], 3)
        self.assertTrue(all(len(s) == 3 for s in result["expanded_nodes"]))
        checked = replay(3, 2, set(), (0, 0), (1, 0), result["path"],
                         dynamic_blocked=frames)
        self.assertTrue(checked["valid"])
        self.assertEqual(checked["cost"], 3)
        # A persistently blocked candidate is not rejected at validation
        # time; the search simply reports it unreachable.
        persistent = [[], [(2, 0)]]
        none = plan_any(3, 1, set(), (0, 0), [(2, 0)],
                        dynamic_blocked=persistent)
        self.assertIsNone(none["path"])
        self.assertIsNone(none["cost"])

    def test_dynamic_selection_and_trace_triples(self):
        result = plan_any(3, 3, set(), (1, 0), [(2, 2), (0, 2)],
                          trace=True, dynamic_blocked=self.FRAMES)
        self.assertEqual(result["path"],
                         [(1, 0), (0, 0), (0, 1), (0, 2)])
        self.assertEqual(result["cost"], 3)
        nodes = result["expanded_nodes"]
        self.assertTrue(all(isinstance(s, tuple) and len(s) == 3
                            for s in nodes))
        self.assertEqual(len(nodes), result["expanded"])
        self.assertEqual(nodes[0], (1, 0, 0))

    def test_single_goal_dynamic_matches_plan_bit_for_bit(self):
        # With exactly one candidate the multi-goal dynamic search is the
        # same traversal as ``plan``: every key, including the trace and
        # expansion count, agrees.
        for kwargs in ({}, {"trace": True}, {"max_expanded": 4},
                       {"trace": True, "max_expanded": 100},
                       {"trace": True, "costs": [
                           [1, 1, 1], [1, 1, 100], [1, 1, 1]]}):
            planned = plan(3, 3, set(), (1, 0), (2, 2),
                           dynamic_blocked=self.FRAMES, **kwargs)
            anyed = plan_any(3, 3, set(), (1, 0), [(2, 2)],
                             dynamic_blocked=self.FRAMES, **kwargs)
            self.assertEqual(anyed, planned, kwargs)

    def test_frame_iteration_order_invariance(self):
        goals = [(2, 2), (0, 2)]
        orderings = [
            self.FRAMES,
            [list(reversed(f)) for f in self.FRAMES],
            [set(f) for f in self.FRAMES],
            [tuple(f) + (f[0],) if f else () for f in self.FRAMES],
        ]
        results = [plan_any(3, 3, [(2, 0), (2, 0)], (1, 0), goals,
                            trace=True, dynamic_blocked=f)
                   for f in orderings]
        for other in results[1:]:
            self.assertEqual(other, results[0])

    def test_dynamic_result_replays(self):
        planned = plan_any(3, 3, set(), (1, 0), [(2, 2), (0, 2)],
                           dynamic_blocked=self.FRAMES)
        goal = planned["path"][-1]
        checked = replay(3, 3, set(), (1, 0), goal, planned["path"],
                         dynamic_blocked=self.FRAMES)
        self.assertEqual(checked, {
            "valid": True,
            "cost": planned["cost"],
            "steps": len(planned["path"]) - 1,
        })

    def test_start_in_goals_dynamic_budget_rules(self):
        frames = [[], [(0, 0)]]
        stopped = plan_any(3, 3, set(), (1, 1), [(2, 2), (1, 1)],
                           trace=True, dynamic_blocked=frames,
                           max_expanded=0)
        self.assertEqual(stopped, {"path": None, "cost": None, "expanded": 0,
                                   "status": "budget_exhausted",
                                   "expanded_nodes": []})
        found = plan_any(3, 3, set(), (1, 1), [(2, 2), (1, 1)],
                         trace=True, dynamic_blocked=frames, max_expanded=1)
        self.assertEqual(found, {"path": [(1, 1)], "cost": 0,
                                 "expanded": 1, "status": "found",
                                 "expanded_nodes": [(1, 1, 0)]})


class PlanAnyBudgetTest(unittest.TestCase):
    WALL = {(1, y) for y in range(3)}

    def test_generous_budget_matches_baseline_plus_status(self):
        blocked = {(2, 0), (2, 1), (2, 2)}
        baseline = plan_any(6, 4, blocked, (0, 0), [(5, 3), (5, 2)],
                            trace=True)
        budgeted = plan_any(6, 4, blocked, (0, 0), [(5, 3), (5, 2)],
                            trace=True, max_expanded=1000)
        self.assertEqual(budgeted, {**baseline, "status": "found"})

    def test_budget_exhausted_static(self):
        result = plan_any(3, 3, self.WALL, (0, 0), [(2, 2), (2, 0)],
                          max_expanded=2)
        self.assertEqual(result, {"path": None, "cost": None, "expanded": 2,
                                  "status": "budget_exhausted"})
        traced = plan_any(3, 3, self.WALL, (0, 0), [(2, 2), (2, 0)],
                          trace=True, max_expanded=2)
        self.assertEqual(traced["expanded_nodes"], [(0, 0), (0, 1)])

    def test_unreachable_vs_budget_exhausted(self):
        for budget in (3, 10):
            result = plan_any(3, 3, self.WALL, (0, 0), [(2, 2)],
                              max_expanded=budget)
            self.assertEqual(result["status"], "unreachable")
            self.assertEqual(result["expanded"], 3)
        stopped = plan_any(3, 3, self.WALL, (0, 0), [(2, 2)],
                           max_expanded=0, trace=True)
        self.assertEqual(stopped["status"], "budget_exhausted")
        self.assertEqual(stopped["expanded"], 0)
        self.assertEqual(stopped["expanded_nodes"], [])

    def test_goal_closed_at_budget_is_found_with_optimal_cost(self):
        result = plan_any(2, 1, set(), (0, 0), [(1, 0)], max_expanded=2)
        self.assertEqual(result, {"path": [(0, 0), (1, 0)], "cost": 1,
                                  "expanded": 2, "status": "found"})

    def test_budgeted_trace_is_prefix_and_found_stays_optimal(self):
        blocked = {(2, 0), (2, 1), (2, 2)}
        goals = [(5, 3), (5, 2)]
        baseline = plan_any(6, 4, blocked, (0, 0), goals, trace=True)
        for budget in range(0, baseline["expanded"] + 1):
            budgeted = plan_any(6, 4, blocked, (0, 0), goals, trace=True,
                                max_expanded=budget)
            self.assertEqual(
                budgeted["expanded_nodes"],
                baseline["expanded_nodes"][:budgeted["expanded"]],
            )
            self.assertLessEqual(budgeted["expanded"], budget)
            if budgeted["status"] == "found":
                # A budget never returns a strictly more expensive route
                # than the unbudgeted optimum.
                self.assertEqual(budgeted["cost"], baseline["cost"])
                checked = replay(6, 4, blocked, (0, 0),
                                 budgeted["path"][-1], budgeted["path"])
                self.assertTrue(checked["valid"])
                self.assertEqual(checked["cost"], budgeted["cost"])

    def test_budget_type_and_value_errors(self):
        for bad in ("5", 1.5, True, [5], (5,)):
            with self.assertRaises(TypeError, msg=f"max_expanded={bad!r}"):
                plan_any(3, 3, set(), (0, 0), [(2, 2)], max_expanded=bad)
        for bad in (-1, -100):
            with self.assertRaises(ValueError, msg=f"max_expanded={bad!r}"):
                plan_any(3, 3, set(), (0, 0), [(2, 2)], max_expanded=bad)


if __name__ == '__main__':
    unittest.main()
