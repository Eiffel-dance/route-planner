import json
import unittest

from app import plan, plan_multi_start, replay, resume


class MultiStartBasicTest(unittest.TestCase):
    def test_nearest_start_wins(self):
        result = plan_multi_start(4, 3, set(), [(0, 0), (3, 2)], (3, 0),
                                  trace=True)
        self.assertEqual(result, {
            "path": [(3, 2), (3, 1), (3, 0)],
            "cost": 2,
            "expanded": 3,
            "expanded_nodes": [(3, 2), (3, 1), (3, 0)],
        })

    def test_result_shape_without_trace(self):
        result = plan_multi_start(4, 3, set(), [(0, 0), (3, 2)], (3, 0))
        self.assertEqual(set(result), {"path", "cost", "expanded"})
        self.assertNotIn("status", result)
        self.assertNotIn("expanded_nodes", result)

    def test_cheaper_start_beats_nearer_one(self):
        # Entering (3, 0) costs 9 either way; the route from (2, 0) is one
        # step, the route from (0, 0) is three steps, so (2, 0) wins.
        costs = [[1, 1, 1, 9], [1, 1, 1, 1], [1, 1, 1, 1]]
        result = plan_multi_start(4, 3, set(), [(0, 0), (2, 0)], (3, 0),
                                  costs=costs)
        self.assertEqual(result["path"], [(2, 0), (3, 0)])
        self.assertEqual(result["cost"], 9)

    def test_equal_cost_picks_lexicographically_smallest_path(self):
        # Open 3x3: both starts are two steps from the goal; the route
        # from (0, 2) is the lexicographically smaller coordinate sequence.
        result = plan_multi_start(3, 3, set(), [(2, 0), (0, 2)], (0, 0))
        self.assertEqual(result["path"], [(0, 2), (0, 1), (0, 0)])
        self.assertEqual(result["cost"], 2)
        reversed_result = plan_multi_start(3, 3, set(), [(0, 2), (2, 0)],
                                           (0, 0))
        self.assertEqual(reversed_result, result)

    def test_duplicate_starts_merge(self):
        once = plan_multi_start(4, 3, set(), [(0, 0), (3, 2)], (3, 0),
                                trace=True)
        repeated = plan_multi_start(
            4, 3, set(), [(3, 2), (0, 0), (3, 2), (0, 0)], (3, 0), trace=True)
        self.assertEqual(repeated, once)

    def test_input_permutation_and_blocked_order_invariance(self):
        cells = [(1, y) for y in range(3)] + [(3, 1)]
        starts = [(0, 0), (4, 2), (2, 2)]
        orderings = (
            (cells, starts),
            (list(reversed(cells)), list(reversed(starts))),
            ([cells[2], cells[0], cells[3], cells[1]],
             [starts[1], starts[0], starts[2], starts[0]]),
        )
        results = [plan_multi_start(5, 3, blocked, ss, (4, 0), trace=True)
                   for blocked, ss in orderings]
        for other in results[1:]:
            self.assertEqual(other, results[0])

    def test_start_equal_goal_single_point(self):
        result = plan_multi_start(3, 3, set(), [(1, 1), (0, 0)], (1, 1),
                                  trace=True)
        self.assertEqual(result, {"path": [(1, 1)], "cost": 0, "expanded": 1,
                                  "expanded_nodes": [(1, 1)]})

    def test_unreachable_when_every_start_sealed(self):
        wall = {(1, y) for y in range(3)}
        result = plan_multi_start(3, 3, wall, [(0, 0), (0, 2)], (2, 1),
                                  trace=True)
        self.assertEqual(result["path"], None)
        self.assertEqual(result["cost"], None)
        self.assertEqual(result["expanded"], 3)
        self.assertEqual(result["expanded_nodes"], [(0, 0), (0, 1), (0, 2)])

    def test_trace_records_unique_coordinates_in_static_mode(self):
        result = plan_multi_start(5, 3, {(2, 0), (2, 1)}, [(0, 0), (4, 2)],
                                  (4, 0), trace=True)
        nodes = result["expanded_nodes"]
        self.assertEqual(len(nodes), len(set(nodes)))
        self.assertTrue(all(isinstance(p, tuple) and len(p) == 2
                            for p in nodes))
        self.assertEqual(result["expanded"], len(nodes))

    def test_successful_path_verifies_in_replay(self):
        blocked = {(2, 0), (2, 1)}
        costs = [[1, 2, 1, 3, 1], [2, 1, 4, 1, 2], [1, 3, 1, 2, 1]]
        result = plan_multi_start(5, 3, blocked, [(0, 0), (4, 2), (0, 2)],
                                  (4, 0), costs=costs)
        rep = replay(5, 3, blocked, result["path"][0], (4, 0),
                     result["path"], costs=costs)
        self.assertEqual(rep, {"valid": True, "cost": result["cost"],
                               "steps": len(result["path"]) - 1})

    def test_single_start_matches_plan(self):
        blocked = {(1, 0), (2, 2)}
        costs = [[2, 1, 3, 1], [1, 4, 1, 2], [3, 1, 2, 1]]
        expected = plan(4, 3, blocked, (0, 0), (3, 2), costs=costs,
                        trace=True)
        result = plan_multi_start(4, 3, blocked, [(0, 0)], (3, 2),
                                  costs=costs, trace=True)
        self.assertEqual(result, expected)


class MultiStartValidationTest(unittest.TestCase):
    def test_starts_outer_type(self):
        for bad in (None, "ab", b"ab", 5, 3.5, {(0, 0)}):
            with self.assertRaises(TypeError, msg=repr(bad)):
                plan_multi_start(3, 3, set(), bad, (0, 0))

    def test_starts_empty(self):
        with self.assertRaises(ValueError):
            plan_multi_start(3, 3, set(), [], (0, 0))
        with self.assertRaises(ValueError):
            plan_multi_start(3, 3, set(), (), (0, 0))

    def test_start_element_structure(self):
        for bad in ([(0, 0, 0)], [(0,)], [(0, "a")], [(True, 0)], ["ab"],
                    [None], [5]):
            with self.assertRaises(TypeError, msg=repr(bad)):
                plan_multi_start(3, 3, set(), bad, (0, 0))

    def test_start_out_of_bounds(self):
        for bad in ([(3, 0)], [(0, -1)], [(-1, 0)], [(0, 3)]):
            with self.assertRaises(ValueError, msg=repr(bad)):
                plan_multi_start(3, 3, set(), bad, (0, 0))

    def test_start_on_obstacle(self):
        with self.assertRaises(ValueError):
            plan_multi_start(3, 3, {(1, 1)}, [(0, 0), (1, 1)], (2, 2))

    def test_goal_checks_still_apply(self):
        with self.assertRaises(ValueError):
            plan_multi_start(3, 3, set(), [(0, 0)], (3, 0))
        with self.assertRaises(ValueError):
            plan_multi_start(3, 3, {(2, 2)}, [(0, 0)], (2, 2))
        with self.assertRaises(TypeError):
            plan_multi_start(3, 3, set(), [(0, 0)], "x")

    def test_frame_zero_blocks_any_start(self):
        with self.assertRaises(ValueError):
            plan_multi_start(3, 3, set(), [(0, 0), (1, 1)], (2, 2),
                             dynamic_blocked=[[(1, 1)]])

    def test_validation_order_follows_plan(self):
        # Dimensions are checked before starts.
        with self.assertRaises(ValueError):
            plan_multi_start(0, 3, set(), None, (0, 0))
        # starts take start's position: before goal.
        with self.assertRaises(TypeError):
            plan_multi_start(3, 3, set(), None, "x")
        # goal is checked before blocked.
        with self.assertRaises(TypeError):
            plan_multi_start(3, 3, "x", [(0, 0)], None)
        # Grid checks run before the budget check.
        with self.assertRaises(ValueError):
            plan_multi_start(3, 3, set(), [(9, 9)], (0, 0), max_expanded=-1)
        # The budget check runs before the snapshot flag check.
        with self.assertRaises(ValueError):
            plan_multi_start(3, 3, set(), [(0, 0)], (0, 0), max_expanded=-1,
                             snapshot=1)

    def test_optional_argument_validation(self):
        with self.assertRaises(TypeError):
            plan_multi_start(3, 3, set(), [(0, 0)], (0, 0), trace=1)
        with self.assertRaises(TypeError):
            plan_multi_start(3, 3, set(), [(0, 0)], (0, 0), max_expanded="x")
        with self.assertRaises(TypeError):
            plan_multi_start(3, 3, set(), [(0, 0)], (0, 0), max_expanded=True)
        with self.assertRaises(ValueError):
            plan_multi_start(3, 3, set(), [(0, 0)], (0, 0), max_expanded=-1)
        with self.assertRaises(TypeError):
            plan_multi_start(3, 3, set(), [(0, 0)], (0, 0), snapshot=1)
        with self.assertRaises(TypeError):
            plan_multi_start(3, 3, set(), [(0, 0)], (0, 0), max_cost="x")
        with self.assertRaises(ValueError):
            plan_multi_start(3, 3, set(), [(0, 0)], (0, 0), max_cost=-1)


class MultiStartDynamicTest(unittest.TestCase):
    def test_dynamic_frames_and_persistent_last_frame(self):
        result = plan_multi_start(4, 3, set(), [(0, 0), (3, 2)], (3, 0),
                                  dynamic_blocked=[[(3, 1)], [(3, 2)]],
                                  trace=True)
        self.assertEqual(result["path"], [(3, 2), (3, 1), (3, 0)])
        self.assertEqual(result["expanded_nodes"],
                         [(3, 2, 0), (3, 1, 1), (3, 0, 2)])

    def test_dynamic_trace_records_triples(self):
        result = plan_multi_start(3, 2, set(), [(0, 0), (2, 1)], (2, 0),
                                  dynamic_blocked=[[(1, 1)]], trace=True)
        self.assertEqual(result["path"], [(2, 1), (2, 0)])
        self.assertEqual(result["expanded_nodes"], [(2, 1, 0), (2, 0, 1)])
        self.assertEqual(result["expanded"],
                         len(result["expanded_nodes"]))

    def test_dynamic_route_verifies_in_replay(self):
        frames = [[(1, 1)], [(2, 2)], []]
        costs = [[1, 2, 1, 3], [2, 1, 4, 1], [1, 1, 1, 2], [3, 1, 2, 1]]
        result = plan_multi_start(4, 4, {(0, 3)}, [(0, 0), (3, 3), (1, 3)],
                                  (3, 0), costs=costs,
                                  dynamic_blocked=frames)
        self.assertEqual(result["path"], [(0, 0), (1, 0), (2, 0), (3, 0)])
        rep = replay(4, 4, {(0, 3)}, (0, 0), (3, 0), result["path"],
                     costs=costs, dynamic_blocked=frames)
        self.assertEqual(rep, {"valid": True, "cost": result["cost"],
                               "steps": len(result["path"]) - 1})

    def test_dynamic_unreachable(self):
        # (2, 0) is sealed by the persistent last frame; (0, 0) is walled.
        frames = [[], [(2, 0)]]
        result = plan_multi_start(3, 1, {(1, 0)}, [(0, 0)], (2, 0),
                                  dynamic_blocked=frames)
        self.assertEqual(result["path"], None)
        self.assertEqual(result["cost"], None)


class MultiStartLimitsTest(unittest.TestCase):
    def test_budget_statuses(self):
        result = plan_multi_start(4, 3, set(), [(0, 0), (3, 2)], (3, 0),
                                  max_expanded=2, trace=True)
        self.assertEqual(result, {
            "path": None, "cost": None, "expanded": 2,
            "status": "budget_exhausted",
            "expanded_nodes": [(3, 2), (3, 1)],
        })
        found = plan_multi_start(4, 3, set(), [(0, 0), (3, 2)], (3, 0),
                                 max_expanded=3)
        self.assertEqual(found["status"], "found")
        self.assertEqual(found["path"], [(3, 2), (3, 1), (3, 0)])
        sealed = plan_multi_start(3, 3, {(1, y) for y in range(3)},
                                  [(0, 0)], (2, 1), max_expanded=10)
        self.assertEqual(sealed["status"], "unreachable")

    def test_zero_budget_is_budget_exhausted(self):
        result = plan_multi_start(3, 3, set(), [(1, 1), (0, 0)], (1, 1),
                                  max_expanded=0, trace=True)
        self.assertEqual(result, {"path": None, "cost": None, "expanded": 0,
                                  "status": "budget_exhausted",
                                  "expanded_nodes": []})

    def test_budgeted_trace_is_prefix_of_unbudgeted(self):
        kwargs = dict(costs=[[2, 1, 3, 1], [1, 4, 1, 2], [3, 1, 2, 1]],
                      trace=True)
        full = plan_multi_start(4, 3, {(1, 1)}, [(0, 0), (3, 2)], (3, 0),
                                **kwargs)
        part = plan_multi_start(4, 3, {(1, 1)}, [(0, 0), (3, 2)], (3, 0),
                                max_expanded=2, **kwargs)
        self.assertEqual(part["expanded_nodes"],
                         full["expanded_nodes"][:part["expanded"]])

    def test_max_cost_statuses(self):
        capped = plan_multi_start(4, 3, set(), [(0, 0), (3, 2)], (3, 0),
                                  max_cost=1, trace=True)
        self.assertEqual(capped["status"], "cost_exhausted")
        self.assertEqual(capped["path"], None)
        exact = plan_multi_start(4, 3, set(), [(0, 0), (3, 2)], (3, 0),
                                 max_cost=2)
        self.assertEqual(exact["status"], "found")
        self.assertEqual(exact["cost"], 2)
        sealed = plan_multi_start(3, 3, {(1, y) for y in range(3)},
                                  [(0, 0)], (2, 1), max_cost=100)
        self.assertEqual(sealed["status"], "unreachable")

    def test_budget_stop_wins_over_cost_limit(self):
        result = plan_multi_start(4, 3, set(), [(0, 0), (3, 2)], (3, 0),
                                  max_expanded=2, max_cost=1)
        self.assertEqual(result["status"], "budget_exhausted")


class MultiStartSnapshotTest(unittest.TestCase):
    ARGS = (4, 4, {(0, 3)}, [(0, 0), (3, 3), (1, 3)], (3, 0))
    COSTS = [[1, 2, 1, 3], [2, 1, 4, 1], [1, 1, 1, 2], [3, 1, 2, 1]]
    FRAMES = [[(1, 1)], [(2, 2)], []]

    def test_static_checkpoint_roundtrip(self):
        full = plan_multi_start(4, 4, set(), [(0, 0), (3, 3)], (3, 0),
                                trace=True)
        part = plan_multi_start(4, 4, set(), [(0, 0), (3, 3)], (3, 0),
                                max_expanded=1, snapshot=True, trace=True)
        self.assertEqual(part["status"], "budget_exhausted")
        checkpoint = part["checkpoint"]
        self.assertEqual(checkpoint["planner"], "plan_multi_start")
        self.assertEqual(checkpoint["starts"], [[0, 0], [3, 3]])
        self.assertEqual(checkpoint["goal"], [3, 0])
        self.assertEqual(checkpoint["closed"], part["expanded"])
        self.assertEqual(json.loads(json.dumps(checkpoint)), checkpoint)
        out = resume(checkpoint, max_expanded=full["expanded"] + 10)
        self.assertEqual(out["path"], full["path"])
        self.assertEqual(out["cost"], full["cost"])
        self.assertEqual(out["expanded"], full["expanded"])
        self.assertEqual(out["expanded_nodes"], full["expanded_nodes"])
        self.assertNotIn("checkpoint", out)

    def test_dynamic_checkpoint_roundtrip(self):
        w, h, blocked, starts, goal = self.ARGS
        full = plan_multi_start(w, h, blocked, starts, goal,
                                costs=self.COSTS, dynamic_blocked=self.FRAMES,
                                trace=True)
        part = plan_multi_start(w, h, blocked, starts, goal,
                                costs=self.COSTS, dynamic_blocked=self.FRAMES,
                                max_expanded=3, snapshot=True, trace=True)
        self.assertEqual(part["status"], "budget_exhausted")
        checkpoint = part["checkpoint"]
        self.assertEqual(checkpoint["planner"], "plan_multi_start")
        self.assertEqual(checkpoint["starts"],
                         sorted([list(s) for s in starts]))
        self.assertEqual(json.loads(json.dumps(checkpoint)), checkpoint)
        out = resume(checkpoint, max_expanded=full["expanded"] + 20)
        self.assertEqual(out["path"], full["path"])
        self.assertEqual(out["expanded"], full["expanded"])
        self.assertEqual(out["expanded_nodes"], full["expanded_nodes"])

    def test_repeated_resume_with_cumulative_budget(self):
        w, h, blocked, starts, goal = self.ARGS
        kwargs = dict(costs=self.COSTS, dynamic_blocked=self.FRAMES,
                      trace=True)
        full = plan_multi_start(w, h, blocked, starts, goal, **kwargs)
        first = plan_multi_start(w, h, blocked, starts, goal,
                                 max_expanded=2, snapshot=True, **kwargs)
        self.assertEqual(first["status"], "budget_exhausted")
        # A cumulative budget not beyond the closed count closes nothing.
        again = resume(first["checkpoint"], max_expanded=2)
        self.assertEqual(again["status"], "budget_exhausted")
        self.assertEqual(again["expanded"], 2)
        self.assertIn("checkpoint", again)
        done = resume(again["checkpoint"],
                      max_expanded=full["expanded"] + 20)
        self.assertEqual(done["path"], full["path"])
        self.assertEqual(done["expanded"], full["expanded"])
        self.assertNotIn("checkpoint", done)

    def test_checkpoint_records_max_cost(self):
        w, h, blocked, starts, goal = self.ARGS
        kwargs = dict(costs=self.COSTS, max_cost=5, trace=True)
        full = plan_multi_start(w, h, blocked, starts, goal, **kwargs)
        self.assertEqual(full["status"], "cost_exhausted")
        part = plan_multi_start(w, h, blocked, starts, goal, max_expanded=2,
                                snapshot=True, **kwargs)
        self.assertEqual(part["status"], "budget_exhausted")
        self.assertEqual(part["checkpoint"]["max_cost"], 5)
        out = resume(part["checkpoint"], max_expanded=1000)
        self.assertEqual(out["status"], "cost_exhausted")
        self.assertEqual(out["expanded"], full["expanded"])

    def test_found_and_unreachable_results_carry_no_checkpoint(self):
        found = plan_multi_start(3, 3, set(), [(0, 0)], (2, 2),
                                 max_expanded=100, snapshot=True)
        self.assertNotIn("checkpoint", found)
        sealed = plan_multi_start(3, 3, {(1, y) for y in range(3)},
                                  [(0, 0)], (2, 2), max_expanded=100,
                                  snapshot=True)
        self.assertNotIn("checkpoint", sealed)

    def test_resume_rejects_invalid_checkpoints(self):
        good = plan_multi_start(4, 4, set(), [(0, 0), (3, 3)], (3, 0),
                                max_expanded=1, snapshot=True)["checkpoint"]
        with self.assertRaises(TypeError):
            resume(None)
        with self.assertRaises(TypeError):
            resume([1, 2, 3])
        for key in ("version", "planner", "starts", "goal", "state"):
            broken = dict(good)
            del broken[key]
            with self.assertRaises(TypeError, msg=key):
                resume(broken)
        broken = dict(good, planner="plan_multi", start=[0, 0])
        with self.assertRaises(ValueError):
            resume(broken)
        broken = dict(good, version=99)
        with self.assertRaises(ValueError):
            resume(broken)
        broken = dict(good, starts=[])
        with self.assertRaises(ValueError):
            resume(broken)
        broken = dict(good, starts=[[9, 9]])
        with self.assertRaises(ValueError):
            resume(broken)
        broken = dict(good, closed=-1)
        with self.assertRaises(ValueError):
            resume(broken)
        broken = dict(good, trace=[[0, 0], [0, 0]])
        with self.assertRaises(ValueError):
            resume(broken)
        broken = dict(good, state=dict(good["state"], open=[]))
        with self.assertRaises(ValueError):
            resume(broken)


if __name__ == "__main__":
    unittest.main()
