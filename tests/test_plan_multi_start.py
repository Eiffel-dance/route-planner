import json
import unittest

from app import plan_multi_start, replay, resume


FRAMES = [
    [(0, 1), (0, 2)],
    [(0, 1), (1, 2), (2, 0), (2, 2)],
    [(2, 1), (2, 2)],
    [(0, 0), (1, 2), (2, 1), (2, 2)],
    [],
    [(0, 1)],
]
BLOCKED = {(2, 0), (2, 1), (2, 2)}


def without_checkpoint(result):
    return {key: value for key, value in result.items()
            if key != "checkpoint"}


class StartsValidationTest(unittest.TestCase):
    def test_outer_type_errors(self):
        for bad in (None, "starts", b"starts", 42, 1.5, {(0, 0)}, True):
            with self.assertRaises(TypeError, msg=f"starts={bad!r}"):
                plan_multi_start(3, 3, set(), bad, (2, 2))

    def test_empty_starts_is_value_error(self):
        for empty in ([], ()):
            with self.assertRaises(ValueError, msg=f"starts={empty!r}"):
                plan_multi_start(3, 3, set(), empty, (2, 2))

    def test_element_type_errors(self):
        for bad in ([(0, 0, 0)], [(0,)], ["ab"], [(0, "1")], [(True, 0)],
                    [None], [[0, 0], "x"]):
            with self.assertRaises(TypeError, msg=f"starts={bad!r}"):
                plan_multi_start(3, 3, set(), bad, (2, 2))

    def test_bounds_and_obstacle_are_value_errors(self):
        for bad in ([(-1, 0)], [(3, 0)], [(0, -1)], [(0, 3)]):
            with self.assertRaises(ValueError, msg=f"starts={bad!r}"):
                plan_multi_start(3, 3, set(), bad, (2, 2))
        with self.assertRaises(ValueError):
            plan_multi_start(3, 3, {(1, 1)}, [(1, 1), (0, 0)], (2, 2))

    def test_goal_on_obstacle_is_value_error(self):
        with self.assertRaises(ValueError):
            plan_multi_start(3, 3, {(2, 2)}, [(0, 0)], (2, 2))

    def test_validation_order_follows_plan(self):
        # Dimensions precede the starts checks.
        with self.assertRaises(TypeError):
            plan_multi_start("3", 3, set(), None, (2, 2))
        with self.assertRaises(ValueError):
            plan_multi_start(0, 3, set(), None, (2, 2))
        # starts structure precedes the goal structure.
        with self.assertRaises(TypeError):
            plan_multi_start(3, 3, set(), None, "goal")
        # starts bounds precede goal bounds.
        with self.assertRaises(ValueError):
            plan_multi_start(3, 3, set(), [(9, 9)], (9, 9))
        # costs precede trace, trace precedes dynamic frames.
        with self.assertRaises(TypeError):
            plan_multi_start(3, 3, set(), [(0, 0)], (2, 2), costs="x",
                             trace=1)
        with self.assertRaises(TypeError):
            plan_multi_start(3, 3, set(), [(0, 0)], (2, 2), trace=1,
                             dynamic_blocked="x")
        # The frame-0 check precedes the budget checks.
        with self.assertRaises(ValueError):
            plan_multi_start(3, 3, set(), [(0, 0)], (2, 2),
                             dynamic_blocked=[[(0, 0)]], max_expanded="x")
        # The budget check precedes the snapshot flag check.
        with self.assertRaises(TypeError):
            plan_multi_start(3, 3, set(), [(0, 0)], (2, 2),
                             max_expanded="x", snapshot=1)
        # The snapshot flag check precedes the max_cost check.
        with self.assertRaises(TypeError):
            plan_multi_start(3, 3, set(), [(0, 0)], (2, 2), snapshot=1,
                             max_cost="x")

    def test_frame_zero_blocking_any_start_is_value_error(self):
        with self.assertRaises(ValueError):
            plan_multi_start(3, 3, set(), [(0, 0), (2, 2)], (1, 1),
                             dynamic_blocked=[[(2, 2)]])
        with self.assertRaises(ValueError):
            plan_multi_start(3, 3, set(), [(0, 0), (2, 2)], (1, 1),
                             dynamic_blocked=[[(0, 0)]])

    def test_optional_argument_type_errors(self):
        for bad in ("5", 1.5, True, [5]):
            with self.assertRaises(TypeError, msg=f"max_expanded={bad!r}"):
                plan_multi_start(3, 3, set(), [(0, 0)], (2, 2),
                                 max_expanded=bad)
            with self.assertRaises(TypeError, msg=f"max_cost={bad!r}"):
                plan_multi_start(3, 3, set(), [(0, 0)], (2, 2), max_cost=bad)
        with self.assertRaises(ValueError):
            plan_multi_start(3, 3, set(), [(0, 0)], (2, 2), max_expanded=-1)
        with self.assertRaises(ValueError):
            plan_multi_start(3, 3, set(), [(0, 0)], (2, 2), max_cost=-1)
        with self.assertRaises(TypeError):
            plan_multi_start(3, 3, set(), [(0, 0)], (2, 2), snapshot=1)

    def test_duplicates_merge_and_order_is_irrelevant(self):
        grid = (6, 4, BLOCKED)
        base = plan_multi_start(*grid, [(0, 0), (5, 3), (0, 3)], (5, 0),
                                trace=True)
        permuted = plan_multi_start(*grid, [(0, 3), (0, 0), (5, 3)], (5, 0),
                                    trace=True)
        duplicated = plan_multi_start(
            *grid, [(5, 3), (0, 0), (0, 3), (0, 0), (5, 3)], (5, 0),
            trace=True)
        self.assertEqual(base, permuted)
        self.assertEqual(base, duplicated)


class RoutingTest(unittest.TestCase):
    def test_nearest_start_wins(self):
        result = plan_multi_start(3, 3, set(), [(0, 0), (2, 2)], (2, 0))
        self.assertEqual(result["path"], [(0, 0), (1, 0), (2, 0)])
        self.assertEqual(result["cost"], 2)

    def test_equal_cost_tie_uses_full_path_order(self):
        # Both starts reach the goal in one step; the lexicographically
        # smaller complete path wins regardless of the input order.
        expected = [(0, 0), (1, 0)]
        self.assertEqual(
            plan_multi_start(3, 1, set(), [(0, 0), (2, 0)], (1, 0))["path"],
            expected)
        self.assertEqual(
            plan_multi_start(3, 1, set(), [(2, 0), (0, 0)], (1, 0))["path"],
            expected)

    def test_costs_tie_uses_full_path_order(self):
        # Two routes of cost 5 leave (1, 0): straight down and around the
        # left edge. The left-edge route is lexicographically smaller and
        # must win even though the straight one develops first.
        costs = [[1, 2, 1], [2, 4, 1], [1, 1, 4]]
        result = plan_multi_start(3, 3, set(), [(1, 0)], (1, 2), costs=costs)
        self.assertEqual(result["path"],
                         [(1, 0), (0, 0), (0, 1), (0, 2), (1, 2)])
        self.assertEqual(result["cost"], 5)

    def test_start_equals_goal_is_single_point(self):
        result = plan_multi_start(3, 3, set(), [(1, 1), (0, 0)], (1, 1))
        self.assertEqual(result, {"path": [(1, 1)], "cost": 0,
                                  "expanded": 1})

    def test_unreachable(self):
        wall = {(1, y) for y in range(3)}
        result = plan_multi_start(3, 3, wall, [(0, 0), (0, 2)], (2, 2))
        self.assertEqual(result["path"], None)
        self.assertEqual(result["cost"], None)
        self.assertEqual(result["expanded"], 3)

    def test_trace_static_records_unique_coordinates(self):
        result = plan_multi_start(6, 4, BLOCKED, [(0, 0), (5, 3)], (5, 0),
                                  trace=True)
        self.assertEqual(result["expanded"], len(result["expanded_nodes"]))
        self.assertEqual(len(set(result["expanded_nodes"])),
                         len(result["expanded_nodes"]))
        self.assertIn(result["expanded_nodes"][0],
                      [(0, 0), (5, 3)])
        self.assertEqual(result["expanded_nodes"][-1], (5, 0))

    def test_trace_omitted_by_default(self):
        result = plan_multi_start(3, 3, set(), [(0, 0)], (2, 2))
        self.assertNotIn("expanded_nodes", result)

    def test_dynamic_basic_and_persistent_last_frame(self):
        # The cell (1, 0) is blocked from frame 1 on, so the route must
        # go around through row 1 whichever start it comes from.
        frames = [[], [(1, 0)]]
        result = plan_multi_start(3, 2, set(), [(0, 0), (0, 1)], (2, 0),
                                  dynamic_blocked=frames)
        self.assertEqual(result["path"], [(0, 1), (1, 1), (2, 1), (2, 0)])
        self.assertEqual(result["cost"], 3)

    def test_dynamic_trace_records_triples(self):
        result = plan_multi_start(3, 3, set(), [(1, 0), (0, 0)], (2, 2),
                                  dynamic_blocked=FRAMES, trace=True)
        self.assertEqual(result["expanded"], len(result["expanded_nodes"]))
        self.assertTrue(all(len(entry) == 3
                            for entry in result["expanded_nodes"]))
        self.assertEqual(result["expanded_nodes"][0][2], 0)

    def test_dynamic_input_order_is_irrelevant(self):
        frames_a = [[(0, 1), (0, 2)], [(2, 0)]]
        frames_b = [[(0, 2), (0, 1)], [(2, 0)]]
        base = plan_multi_start(3, 3, set(), [(1, 0), (0, 0)], (2, 2),
                                dynamic_blocked=frames_a, trace=True)
        self.assertEqual(
            base,
            plan_multi_start(3, 3, set(), [(0, 0), (1, 0)], (2, 2),
                             dynamic_blocked=frames_a, trace=True))
        self.assertEqual(
            base,
            plan_multi_start(3, 3, set(), [(1, 0), (0, 0)], (2, 2),
                             dynamic_blocked=frames_b, trace=True))

    def test_result_replays_offline(self):
        for frames in (None, FRAMES):
            result = plan_multi_start(3, 3, set(), [(1, 0), (0, 0)], (2, 2),
                                      dynamic_blocked=frames)
            self.assertIsNotNone(result["path"])
            checked = replay(3, 3, set(), result["path"][0], (2, 2),
                             result["path"], dynamic_blocked=frames)
            self.assertEqual(checked, {
                "valid": True,
                "cost": result["cost"],
                "steps": len(result["path"]) - 1,
            })


class LimitTest(unittest.TestCase):
    def test_budget_statuses(self):
        found = plan_multi_start(6, 4, BLOCKED, [(0, 0)], (5, 3),
                                 max_expanded=100)
        self.assertEqual(found["status"], "found")
        self.assertIsNotNone(found["path"])
        stopped = plan_multi_start(6, 4, BLOCKED, [(0, 0)], (5, 3),
                                   max_expanded=3)
        self.assertEqual(stopped["status"], "budget_exhausted")
        self.assertEqual(stopped["path"], None)
        self.assertEqual(stopped["cost"], None)
        self.assertEqual(stopped["expanded"], 3)
        wall = {(1, y) for y in range(3)}
        unreachable = plan_multi_start(3, 3, wall, [(0, 0)], (2, 2),
                                       max_expanded=10)
        self.assertEqual(unreachable["status"], "unreachable")

    def test_zero_budget_start_equals_goal(self):
        result = plan_multi_start(3, 3, set(), [(1, 1)], (1, 1),
                                  max_expanded=0, trace=True)
        self.assertEqual(result, {"path": None, "cost": None, "expanded": 0,
                                  "status": "budget_exhausted",
                                  "expanded_nodes": []})
        one = plan_multi_start(3, 3, set(), [(1, 1)], (1, 1), max_expanded=1)
        self.assertEqual(one["status"], "found")
        self.assertEqual(one["path"], [(1, 1)])

    def test_budgeted_trace_is_prefix_of_unbudgeted(self):
        full = plan_multi_start(6, 4, BLOCKED, [(0, 0), (5, 3)], (5, 0),
                                trace=True)
        for budget in range(0, full["expanded"] + 1):
            limited = plan_multi_start(6, 4, BLOCKED, [(0, 0), (5, 3)],
                                       (5, 0), trace=True,
                                       max_expanded=budget)
            self.assertEqual(
                limited["expanded_nodes"],
                full["expanded_nodes"][:limited["expanded"]],
                f"budget={budget}",
            )

    def test_max_cost_statuses(self):
        found = plan_multi_start(3, 3, set(), [(0, 0)], (2, 2), max_cost=4)
        self.assertEqual(found["status"], "found")
        self.assertEqual(found["cost"], 4)
        exhausted = plan_multi_start(3, 3, set(), [(0, 0)], (2, 2),
                                     max_cost=3)
        self.assertEqual(exhausted["status"], "cost_exhausted")
        self.assertEqual(exhausted["path"], None)
        wall = {(1, y) for y in range(3)}
        unreachable = plan_multi_start(3, 3, wall, [(0, 0)], (2, 2),
                                       max_cost=100)
        self.assertEqual(unreachable["status"], "unreachable")
        # A budget stop wins over the cost-limit status.
        both = plan_multi_start(6, 4, BLOCKED, [(0, 0)], (5, 3),
                                max_expanded=3, max_cost=100)
        self.assertEqual(both["status"], "budget_exhausted")

    def test_max_cost_never_changes_unlimited_route(self):
        full = plan_multi_start(6, 4, BLOCKED, [(0, 0), (5, 3)], (5, 0))
        limited = plan_multi_start(6, 4, BLOCKED, [(0, 0), (5, 3)], (5, 0),
                                   max_cost=full["cost"])
        self.assertEqual(limited["path"], full["path"])
        self.assertEqual(limited["cost"], full["cost"])
        self.assertEqual(limited["status"], "found")


class SnapshotResumeTest(unittest.TestCase):
    def assert_resume_matches(self, args, kwargs):
        full = plan_multi_start(*args, trace=True, **kwargs)
        n = full["expanded"]
        for b1 in range(0, n + 2):
            first = plan_multi_start(*args, trace=True, max_expanded=b1,
                                     snapshot=True, **kwargs)
            self.assertEqual(
                without_checkpoint(first),
                without_checkpoint(plan_multi_start(
                    *args, trace=True, max_expanded=b1, **kwargs)),
                f"b1={b1}",
            )
            if first["status"] != "budget_exhausted":
                self.assertNotIn("checkpoint", first)
                continue
            checkpoint = json.loads(json.dumps(first["checkpoint"]))
            self.assertEqual(checkpoint["planner"], "plan_multi_start")
            self.assertEqual(checkpoint["starts"],
                             sorted(checkpoint["starts"]))
            self.assertEqual(resume(checkpoint), full, f"b1={b1}")
            for b2 in (b1, b1 + 1, n + 5):
                resumed = resume(checkpoint, max_expanded=b2)
                direct = plan_multi_start(*args, trace=True,
                                          max_expanded=b2, snapshot=True,
                                          **kwargs)
                self.assertEqual(without_checkpoint(resumed),
                                 without_checkpoint(direct),
                                 f"b1={b1} b2={b2}")
                if resumed["status"] == "budget_exhausted":
                    self.assertIn("checkpoint", resumed)
                    self.assertEqual(resume(resumed["checkpoint"]), full)
                else:
                    self.assertNotIn("checkpoint", resumed)

    def test_static(self):
        self.assert_resume_matches((6, 4, BLOCKED, [(0, 0), (5, 3)], (5, 0)),
                                   {})

    def test_static_with_costs(self):
        costs = [[1, 10, 1], [1, 10, 1], [1, 1, 1]]
        self.assert_resume_matches((3, 3, set(), [(0, 0), (2, 2)], (2, 0)),
                                   {"costs": costs})

    def test_static_with_max_cost(self):
        self.assert_resume_matches(
            (6, 4, BLOCKED, [(0, 0), (5, 3)], (5, 0)), {"max_cost": 9})

    def test_dynamic(self):
        self.assert_resume_matches((3, 3, set(), [(1, 0), (0, 0)], (2, 2)),
                                   {"dynamic_blocked": FRAMES})

    def test_checkpoint_is_json_serializable(self):
        result = plan_multi_start(6, 4, BLOCKED, [(0, 0), (5, 3)], (5, 0),
                                  max_expanded=3, snapshot=True)
        checkpoint = result["checkpoint"]
        self.assertEqual(json.loads(json.dumps(checkpoint)), checkpoint)
        self.assertEqual(checkpoint["planner"], "plan_multi_start")
        self.assertEqual(checkpoint["starts"], [[0, 0], [5, 3]])
        self.assertEqual(checkpoint["goal"], [5, 0])
        self.assertEqual(checkpoint["closed"], result["expanded"])

    def test_found_and_unreachable_carry_no_checkpoint(self):
        found = plan_multi_start(6, 4, BLOCKED, [(0, 0)], (5, 3),
                                 max_expanded=1000, snapshot=True)
        self.assertNotIn("checkpoint", found)
        wall = {(1, y) for y in range(3)}
        unreachable = plan_multi_start(3, 3, wall, [(0, 0)], (2, 2),
                                       max_expanded=10, snapshot=True)
        self.assertNotIn("checkpoint", unreachable)

    def test_resumed_path_replays(self):
        first = plan_multi_start(3, 3, set(), [(1, 0), (0, 0)], (2, 2),
                                 dynamic_blocked=FRAMES, max_expanded=2,
                                 snapshot=True)
        resumed = resume(first["checkpoint"])
        self.assertIsNotNone(resumed["path"])
        checked = replay(3, 3, set(), resumed["path"][0], (2, 2),
                         resumed["path"], dynamic_blocked=FRAMES)
        self.assertEqual(checked, {
            "valid": True,
            "cost": resumed["cost"],
            "steps": len(resumed["path"]) - 1,
        })

    def test_resume_validation_boundaries(self):
        checkpoint = plan_multi_start(6, 4, BLOCKED, [(0, 0), (5, 3)],
                                      (5, 0), max_expanded=3,
                                      snapshot=True)["checkpoint"]

        def mutated(mutate):
            import copy
            cp = json.loads(json.dumps(checkpoint))
            mutate(cp)
            return cp

        for bad in (None, 42, "checkpoint", [1]):
            with self.assertRaises(TypeError, msg=f"checkpoint={bad!r}"):
                resume(bad)
        for key in ("version", "planner", "width", "height", "blocked",
                    "starts", "goal", "costs", "dynamic_blocked", "closed",
                    "trace", "state"):
            with self.assertRaises(TypeError, msg=f"missing {key}"):
                resume(mutated(lambda c, k=key: c.pop(k)))
        type_cases = [
            lambda c: c.update(version="1"),
            lambda c: c.update(planner=1),
            lambda c: c.update(starts="x"),
            lambda c: c.update(goal=[0]),
            lambda c: c.update(closed="3"),
            lambda c: c.update(trace="x"),
            lambda c: c.update(state=[]),
            lambda c: c["state"].update(open=None),
            lambda c: c["state"].update(best=None),
            lambda c: c["state"].update(closed=[[0]]),
        ]
        for mutate in type_cases:
            with self.assertRaises(TypeError, msg=mutate):
                resume(mutated(mutate))

        def unsupported_planner(c):
            # An unknown planner kind with an otherwise complete object.
            c["planner"] = "astar"
            c["start"] = c.pop("starts")

        value_cases = [
            lambda c: c.update(version=2),
            unsupported_planner,
            lambda c: c.update(starts=[]),         # no starts
            lambda c: c.update(starts=[[9, 9]]),
            lambda c: c.update(starts=[[2, 0]]),   # start on obstacle
            lambda c: c.update(goal=[2, 0]),       # goal on obstacle
            lambda c: c.update(closed=-1),
            lambda c: c.update(closed=2),          # trace length mismatch
            lambda c: c.update(trace=[]),
            lambda c: c["state"].update(open=[]),  # no pending candidates
            lambda c: c["state"].update(closed=[]),
        ]
        for mutate in value_cases:
            with self.assertRaises(ValueError, msg=mutate):
                resume(mutated(mutate))


if __name__ == '__main__':
    unittest.main()
