import json
import unittest

from app import plan, plan_any, replay, resume


# A 3x3 grid where the direct route to (2, 0) enters (1, 0) at t=1 and
# (2, 0) at t=2, both priced 9, while every later frame is uniformly
# cheap: the minimum-cost route takes a detour and arrives late.
ONES = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
FRAMES_T1_PRICEY = [[1, 9, 1], [1, 1, 1], [1, 1, 1]]
FRAMES_T2_PRICEY = [[1, 1, 9], [1, 1, 1], [1, 1, 1]]
COST_FRAMES = [ONES, FRAMES_T1_PRICEY, FRAMES_T2_PRICEY, ONES]


def without_checkpoint(result):
    return {key: value for key, value in result.items()
            if key != "checkpoint"}


class DynamicCostsValidationTest(unittest.TestCase):
    def test_non_sequence_outer_raises_type_error(self):
        for bad in (5, 1.5, "x", b"x", True, {1: 2}):
            with self.assertRaises(TypeError, msg=f"dynamic_costs={bad!r}"):
                plan(3, 3, set(), (0, 0), (2, 2), dynamic_costs=bad)
            with self.assertRaises(TypeError, msg=f"dynamic_costs={bad!r}"):
                plan_any(3, 3, set(), (0, 0), [(2, 2)], dynamic_costs=bad)
            with self.assertRaises(TypeError, msg=f"dynamic_costs={bad!r}"):
                replay(3, 3, set(), (0, 0), (2, 2), [(0, 0)],
                       dynamic_costs=bad)

    def test_bad_frame_or_row_raises_type_error(self):
        bad_frames = (
            [5],                # frame is not a sequence
            ["x"],              # frame is a string
            [[[1, 1, 1], 5, [1, 1, 1]]],   # row is not a sequence
            [[[1, 1, 1], "x", [1, 1, 1]]],  # row is a string
        )
        for bad in bad_frames:
            with self.assertRaises(TypeError, msg=f"dynamic_costs={bad!r}"):
                plan(3, 3, set(), (0, 0), (2, 2), dynamic_costs=bad)

    def test_bad_cell_type_raises_type_error(self):
        for cell in ("1", 1.5, None, True, False):
            frame = [[1, 1, 1], [1, cell, 1], [1, 1, 1]]
            with self.assertRaises(TypeError, msg=f"cell={cell!r}"):
                plan(3, 3, set(), (0, 0), (2, 2), dynamic_costs=[frame])
            with self.assertRaises(TypeError, msg=f"cell={cell!r}"):
                replay(3, 3, set(), (0, 0), (2, 2), [(0, 0)],
                       dynamic_costs=[frame])

    def test_wrong_shape_raises_value_error(self):
        bad_shapes = (
            [[[1, 1], [1, 1]]],                    # too few rows
            [[ONES, ONES, ONES, ONES]],            # too many rows
            [[[1, 1, 1], [1, 1], [1, 1, 1]]],      # short row
            [[[1, 1, 1], [1, 1, 1, 1], [1, 1, 1]]],  # long row
        )
        for bad in bad_shapes:
            with self.assertRaises(ValueError, msg=f"dynamic_costs={bad!r}"):
                plan(3, 3, set(), (0, 0), (2, 2), dynamic_costs=bad)
            with self.assertRaises(ValueError, msg=f"dynamic_costs={bad!r}"):
                plan_any(3, 3, set(), (0, 0), [(2, 2)], dynamic_costs=bad)

    def test_non_positive_cell_raises_value_error(self):
        for cell in (0, -3):
            frame = [[1, 1, 1], [1, cell, 1], [1, 1, 1]]
            with self.assertRaises(ValueError, msg=f"cell={cell!r}"):
                plan(3, 3, set(), (0, 0), (2, 2), dynamic_costs=[frame])

    def test_costs_conflict_raises_value_error(self):
        with self.assertRaises(ValueError):
            plan(3, 3, set(), (0, 0), (2, 2), costs=ONES,
                 dynamic_costs=COST_FRAMES)
        with self.assertRaises(ValueError):
            plan_any(3, 3, set(), (0, 0), [(2, 2)], costs=ONES,
                     dynamic_costs=COST_FRAMES)
        with self.assertRaises(ValueError):
            replay(3, 3, set(), (0, 0), (2, 2), [(0, 0)], costs=ONES,
                   dynamic_costs=COST_FRAMES)

    def test_none_and_empty_sequence_are_not_provided(self):
        args = (3, 3, set(), (0, 0), (2, 2))
        default = plan(*args)
        self.assertEqual(plan(*args, dynamic_costs=None), default)
        self.assertEqual(plan(*args, dynamic_costs=[]), default)
        self.assertEqual(plan(*args, dynamic_costs=()), default)
        # An empty sequence does not conflict with a static cost matrix.
        with_costs = plan(*args, costs=ONES)
        self.assertEqual(plan(*args, costs=ONES, dynamic_costs=[]),
                         with_costs)
        any_default = plan_any(3, 3, set(), (0, 0), [(2, 2), (0, 2)])
        self.assertEqual(
            plan_any(3, 3, set(), (0, 0), [(2, 2), (0, 2)],
                     dynamic_costs=[]),
            any_default,
        )
        replay_default = replay(3, 3, set(), (0, 0), (2, 2),
                                [(0, 0), (1, 1), (2, 2)])
        self.assertEqual(
            replay(3, 3, set(), (0, 0), (2, 2), [(0, 0), (1, 1), (2, 2)],
                   dynamic_costs=None),
            replay_default,
        )

    def test_validated_after_existing_checks(self):
        # A costs TypeError precedes any dynamic_costs error.
        with self.assertRaises(TypeError) as caught:
            plan(3, 3, set(), (0, 0), (2, 2), costs="x", dynamic_costs=5)
        self.assertIn("costs", str(caught.exception))
        # An allow_wait TypeError precedes any dynamic_costs error.
        with self.assertRaises(TypeError) as caught:
            plan(3, 3, set(), (0, 0), (2, 2), allow_wait=1,
                 dynamic_costs=5)
        self.assertIn("allow_wait", str(caught.exception))
        # A max_cost TypeError precedes any dynamic_costs error.
        with self.assertRaises(TypeError) as caught:
            plan(3, 3, set(), (0, 0), (2, 2), max_cost="x",
                 dynamic_costs=5)
        self.assertIn("max_cost", str(caught.exception))
        # In replay a diagnose TypeError precedes any dynamic_costs error.
        with self.assertRaises(TypeError) as caught:
            replay(3, 3, set(), (0, 0), (2, 2), [(0, 0)], diagnose=1,
                   dynamic_costs=5)
        self.assertIn("diagnose", str(caught.exception))

    def test_other_entry_points_unchanged(self):
        from app import (plan_k, plan_batch, plan_multi_start,
                         distance_field, distance_field_any)
        with self.assertRaises(TypeError):
            plan_k(3, 3, set(), (0, 0), (2, 2), 2, dynamic_costs=COST_FRAMES)
        with self.assertRaises(TypeError):
            plan_batch(3, 3, set(), [[(0, 0), (2, 2)]],
                       dynamic_costs=COST_FRAMES)
        with self.assertRaises(TypeError):
            plan_multi_start(3, 3, set(), [(0, 0)], (2, 2),
                             dynamic_costs=COST_FRAMES)
        with self.assertRaises(TypeError):
            distance_field(3, 3, set(), (2, 2), dynamic_costs=COST_FRAMES)
        with self.assertRaises(TypeError):
            distance_field_any(3, 3, set(), [(2, 2)],
                               dynamic_costs=COST_FRAMES)


class DynamicCostsSearchTest(unittest.TestCase):
    def test_detour_to_cheaper_frame(self):
        # The direct route costs 9 + 9; the cheapest detour arrives late
        # and pays only unit costs.
        result = plan(3, 3, set(), (0, 0), (2, 0), dynamic_costs=COST_FRAMES)
        self.assertEqual(result["cost"], 4)
        self.assertEqual(result["path"],
                         [(0, 0), (0, 1), (1, 1), (1, 0), (2, 0)])

    def test_start_cell_never_counted(self):
        # Frame 0 prices the start cell 100; the cost is still 0 + 1 + 1.
        frames = [[[100, 1, 1]]]
        result = plan(3, 1, set(), (0, 0), (2, 0), dynamic_costs=frames)
        self.assertEqual(result["cost"], 2)

    def test_last_frame_persists(self):
        # Only two frames; the second one prices (2, 0) at 7 and persists.
        frames = [[[1, 1, 1]], [[1, 1, 7]]]
        result = plan(3, 1, set(), (0, 0), (2, 0), dynamic_costs=frames)
        self.assertEqual(result["path"], [(0, 0), (1, 0), (2, 0)])
        self.assertEqual(result["cost"], 1 + 7)

    def test_no_wait_without_dynamic_blocked(self):
        # Cost frames alone never generate waits: arriving later would be
        # cheaper, but without dynamic_blocked frames no wait is legal.
        frames = [[[1, 1, 1]], [[1, 1, 1]], [[1, 1, 9]], [[1, 1, 1]]]
        result = plan(3, 1, set(), (0, 0), (2, 0), dynamic_costs=frames,
                      allow_wait=True)
        self.assertEqual(result["path"], [(0, 0), (1, 0), (2, 0)])
        self.assertEqual(result["cost"], 1 + 9)

    def test_wait_adds_no_cost_with_dynamic_blocked(self):
        # With dynamic frames and allow_wait a route may wait for a
        # cheaper frame; the wait itself is free.
        blocked_frames = [[], [], [], []]
        cost_frames = [ONES[0:1], ONES[0:1], [[1, 1, 9]], ONES[0:1]]
        result = plan(3, 1, set(), (0, 0), (2, 0),
                      dynamic_blocked=blocked_frames,
                      dynamic_costs=cost_frames, allow_wait=True)
        self.assertEqual(result["path"], [(0, 0), (0, 0), (1, 0), (2, 0)])
        self.assertEqual(result["cost"], 2)
        # Without waiting the direct route pays the pricey frame.
        no_wait = plan(3, 1, set(), (0, 0), (2, 0),
                       dynamic_blocked=blocked_frames,
                       dynamic_costs=cost_frames)
        self.assertEqual(no_wait["cost"], 1 + 9)

    def test_trace_records_space_time_triples(self):
        result = plan(3, 3, set(), (0, 0), (2, 0), trace=True,
                      dynamic_costs=COST_FRAMES)
        self.assertIn("expanded_nodes", result)
        self.assertTrue(all(len(node) == 3
                            for node in result["expanded_nodes"]))
        self.assertEqual(result["expanded"], len(result["expanded_nodes"]))
        self.assertEqual(result["expanded_nodes"][0], (0, 0, 0))

    def test_histories_are_not_merged(self):
        # (0,0)->(0,1)->(1,1) and (0,0)->(1,0)->(1,1) reach (1, 1) at
        # t=2 through different histories with equal cost; both
        # space-time states are closed and recorded, once each.
        result = plan(3, 3, set(), (0, 0), (2, 2), trace=True,
                      dynamic_costs=[ONES])
        arrivals = [node for node in result["expanded_nodes"]
                    if node == (1, 1, 2)]
        self.assertEqual(len(arrivals), 2)
        self.assertEqual(result["expanded"], len(result["expanded_nodes"]))

    def test_start_equals_goal(self):
        result = plan(3, 3, set(), (1, 1), (1, 1),
                      dynamic_costs=COST_FRAMES)
        self.assertEqual(result,
                         {"path": [(1, 1)], "cost": 0, "expanded": 1})

    def test_unreachable(self):
        result = plan(3, 3, {(1, 0), (1, 1), (1, 2)}, (0, 0), (2, 2),
                      dynamic_costs=COST_FRAMES)
        self.assertEqual(result["path"], None)
        self.assertEqual(result["cost"], None)
        self.assertGreater(result["expanded"], 0)

    def test_zero_budget(self):
        result = plan(3, 3, set(), (1, 1), (1, 1), max_expanded=0,
                      dynamic_costs=COST_FRAMES)
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(result["expanded"], 0)
        self.assertNotIn("expanded_nodes", result)

    def test_max_cost_caps_dynamic_cost(self):
        found = plan(3, 3, set(), (0, 0), (2, 0), max_cost=4,
                     dynamic_costs=COST_FRAMES)
        self.assertEqual(found["status"], "found")
        self.assertEqual(found["cost"], 4)
        exhausted = plan(3, 3, set(), (0, 0), (2, 0), max_cost=3,
                         dynamic_costs=COST_FRAMES)
        self.assertEqual(exhausted["status"], "cost_exhausted")
        self.assertIsNone(exhausted["path"])
        self.assertIsNone(exhausted["cost"])

    def test_budget_priority_over_cost_limit(self):
        result = plan(3, 3, set(), (0, 0), (2, 0), max_expanded=1,
                      max_cost=1, dynamic_costs=COST_FRAMES)
        self.assertEqual(result["status"], "budget_exhausted")


class DynamicCostsPlanAnyTest(unittest.TestCase):
    def test_winning_endpoint_by_cost(self):
        # (0, 2) is reachable cheaply; (2, 0) only through pricey frames.
        result = plan_any(3, 3, set(), (0, 0), [(2, 0), (0, 2)],
                          dynamic_costs=COST_FRAMES)
        self.assertEqual(result["path"], [(0, 0), (0, 1), (0, 2)])
        self.assertEqual(result["cost"], 2)

    def test_endpoint_tie_break(self):
        # Both endpoints cost 2; the lexicographically smaller wins.
        result = plan_any(3, 3, set(), (0, 0), [(2, 0), (0, 2)],
                          dynamic_costs=[ONES])
        self.assertEqual(result["path"], [(0, 0), (0, 1), (0, 2)])
        self.assertEqual(result["cost"], 2)

    def test_trace_shape_and_start_in_goals(self):
        result = plan_any(3, 3, set(), (0, 0), [(0, 0), (2, 2)],
                          dynamic_costs=COST_FRAMES, trace=True)
        self.assertEqual(result["path"], [(0, 0)])
        self.assertEqual(result["cost"], 0)
        self.assertEqual(result["expanded_nodes"], [(0, 0, 0)])

    def test_unreachable(self):
        result = plan_any(3, 3, {(1, 0), (1, 1), (1, 2)}, (0, 0),
                          [(2, 0), (2, 2)], dynamic_costs=COST_FRAMES)
        self.assertIsNone(result["path"])
        self.assertIsNone(result["cost"])

    def test_successful_path_replays(self):
        result = plan_any(3, 3, set(), (0, 0), [(2, 0), (0, 2)],
                          dynamic_costs=COST_FRAMES)
        checked = replay(3, 3, set(), (0, 0), result["path"][-1],
                         result["path"], dynamic_costs=COST_FRAMES)
        self.assertTrue(checked["valid"])
        self.assertEqual(checked["cost"], result["cost"])
        self.assertEqual(checked["steps"], len(result["path"]) - 1)


class DynamicCostsReplayTest(unittest.TestCase):
    PATH = [(0, 0), (0, 1), (1, 1), (1, 0), (2, 0)]

    def test_cost_recomputed_with_frames(self):
        result = replay(3, 3, set(), (0, 0), (2, 0), self.PATH,
                        dynamic_costs=COST_FRAMES)
        self.assertEqual(result, {"valid": True, "cost": 4, "steps": 4})

    def test_direct_path_pays_frame_prices(self):
        result = replay(3, 3, set(), (0, 0), (2, 0),
                        [(0, 0), (1, 0), (2, 0)],
                        dynamic_costs=COST_FRAMES)
        self.assertEqual(result, {"valid": True, "cost": 18, "steps": 2})

    def test_planned_path_verifies(self):
        planned = plan(3, 3, set(), (0, 0), (2, 0),
                       dynamic_costs=COST_FRAMES)
        checked = replay(3, 3, set(), (0, 0), (2, 0), planned["path"],
                         dynamic_costs=COST_FRAMES)
        self.assertEqual(checked, {"valid": True, "cost": planned["cost"],
                                   "steps": len(planned["path"]) - 1})

    def test_invalid_path_structure_unchanged(self):
        result = replay(3, 3, set(), (0, 0), (2, 2),
                        [(0, 0), (0, 0), (1, 1), (2, 2)],
                        dynamic_costs=COST_FRAMES)
        self.assertEqual(result,
                         {"valid": False, "cost": None, "steps": None})
        diagnosed = replay(3, 3, set(), (0, 0), (2, 2),
                           [(0, 0), (0, 0), (1, 1), (2, 2)],
                           dynamic_costs=COST_FRAMES, diagnose=True)
        self.assertEqual(diagnosed["error"], "repeated_coordinate")
        self.assertEqual(diagnosed["error_index"], 1)

    def test_diagnose_codes_unchanged(self):
        diagnosed = replay(3, 3, set(), (0, 0), (2, 0),
                           [(0, 0), (2, 0)], dynamic_costs=COST_FRAMES,
                           diagnose=True)
        self.assertEqual(diagnosed["error"], "non_adjacent")
        self.assertEqual(diagnosed["error_index"], 1)


class DynamicCostsSnapshotTest(unittest.TestCase):
    def assert_resume_matches(self, call, args, kwargs):
        full = call(*args, trace=True, **kwargs)
        n = full["expanded"]
        for b1 in range(0, n + 2):
            first = call(*args, trace=True, max_expanded=b1, snapshot=True,
                         **kwargs)
            self.assertEqual(
                without_checkpoint(first),
                call(*args, trace=True, max_expanded=b1, **kwargs),
                f"b1={b1}",
            )
            if first["status"] != "budget_exhausted":
                self.assertNotIn("checkpoint", first)
                continue
            checkpoint = json.loads(json.dumps(first["checkpoint"]))
            self.assertEqual(resume(checkpoint), full, f"b1={b1}")
            for b2 in (b1, b1 + 1, n + 5):
                resumed = resume(checkpoint, max_expanded=b2)
                direct = call(*args, trace=True, max_expanded=b2,
                              snapshot=True, **kwargs)
                self.assertEqual(without_checkpoint(resumed),
                                 without_checkpoint(direct),
                                 f"b1={b1} b2={b2}")
                if resumed["status"] == "budget_exhausted":
                    self.assertIn("checkpoint", resumed)
                    self.assertEqual(resume(resumed["checkpoint"]), full)
                else:
                    self.assertNotIn("checkpoint", resumed)

    def test_resume_equivalence(self):
        self.assert_resume_matches(
            plan, (3, 3, set(), (0, 0), (2, 0)),
            {"dynamic_costs": COST_FRAMES})
        self.assert_resume_matches(
            plan_any, (3, 3, set(), (0, 0), [(2, 0), (0, 2)]),
            {"dynamic_costs": COST_FRAMES})

    def test_resume_equivalence_with_frames_and_wait(self):
        self.assert_resume_matches(
            plan, (3, 1, set(), (0, 0), (2, 0)),
            {"dynamic_blocked": [[], [], [], []],
             "dynamic_costs": [ONES[0:1], ONES[0:1], [[1, 1, 9]], ONES[0:1]],
             "allow_wait": True})

    def test_resume_equivalence_with_max_cost(self):
        self.assert_resume_matches(
            plan, (3, 3, set(), (0, 0), (2, 0)),
            {"dynamic_costs": COST_FRAMES, "max_cost": 4})

    def test_checkpoint_records_dynamic_costs(self):
        checkpoint = plan(3, 3, set(), (0, 0), (2, 0),
                          dynamic_costs=COST_FRAMES, max_expanded=2,
                          snapshot=True)["checkpoint"]
        self.assertEqual(checkpoint["dynamic_costs"],
                         [list(map(list, frame)) for frame in COST_FRAMES])
        json.dumps(checkpoint)  # fully JSON-serializable
        plain = plan(3, 3, set(), (0, 0), (2, 0), max_expanded=2,
                     snapshot=True)["checkpoint"]
        self.assertNotIn("dynamic_costs", plain)

    def make_checkpoint(self):
        return plan(3, 3, set(), (0, 0), (2, 0),
                    dynamic_costs=COST_FRAMES, max_expanded=2,
                    snapshot=True)["checkpoint"]

    def test_corrupted_field_raises_type_error(self):
        checkpoint = self.make_checkpoint()
        for bad in ("x", 5, [5], [[[1, 1, 1], [1, "a", 1], [1, 1, 1]]]):
            broken = json.loads(json.dumps(checkpoint))
            broken["dynamic_costs"] = bad
            with self.assertRaises(TypeError, msg=f"bad={bad!r}"):
                resume(broken)

    def test_corrupted_field_raises_value_error(self):
        checkpoint = self.make_checkpoint()
        bad_values = (
            [[[1, 1], [1, 1]]],                            # wrong shape
            [[[1, 1, 1], [1, 0, 1], [1, 1, 1]]],           # non-positive
            [[[9, 9, 9], [9, 9, 9], [9, 9, 9]]] * 4,       # state mismatch
        )
        for bad in bad_values:
            broken = json.loads(json.dumps(checkpoint))
            broken["dynamic_costs"] = bad
            with self.assertRaises(ValueError, msg=f"bad={bad!r}"):
                resume(broken)

    def test_conflicting_costs_field_raises_value_error(self):
        broken = json.loads(json.dumps(self.make_checkpoint()))
        broken["costs"] = ONES
        with self.assertRaises(ValueError):
            resume(broken)

    def test_dropped_field_does_not_silently_degrade(self):
        # Removing the field makes the recorded dynamic trace/state
        # inconsistent, so resume must fail rather than plan without the
        # cost frames.
        broken = json.loads(json.dumps(self.make_checkpoint()))
        del broken["dynamic_costs"]
        with self.assertRaises((TypeError, ValueError)):
            resume(broken)


if __name__ == '__main__':
    unittest.main()
