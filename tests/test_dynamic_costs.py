"""Tests for the time-varying cost maps (``dynamic_costs``)."""

import json
import unittest

from app import (plan, plan_any, plan_batch, plan_k, plan_multi_start,
                 replay, resume)


def frames_all(w, h, value, count):
    return [[[value] * w for _ in range(h)] for _ in range(count)]


class LegacyBehaviorTests(unittest.TestCase):
    def test_omitted_none_and_empty_are_equivalent(self):
        base = plan(4, 3, [(1, 1)], (0, 0), (3, 2), trace=True)
        for dynamic_costs in (None, []):
            result = plan(4, 3, [(1, 1)], (0, 0), (3, 2), trace=True,
                          dynamic_costs=dynamic_costs)
            self.assertEqual(result, base)
            self.assertEqual(set(result), {"path", "cost", "expanded",
                                           "expanded_nodes"})

    def test_empty_sequence_does_not_conflict_with_costs(self):
        result = plan(2, 2, [], (0, 0), (1, 1),
                      costs=[[1, 1], [1, 1]], dynamic_costs=[])
        self.assertEqual(result["cost"], 2)

    def test_other_entry_points_have_no_dynamic_costs_parameter(self):
        with self.assertRaises(TypeError):
            plan_k(3, 2, [], (0, 0), (2, 0), 2,
                   dynamic_costs=frames_all(3, 2, 1, 1))
        with self.assertRaises(TypeError):
            plan_multi_start(3, 2, [], [(0, 0)], (2, 0),
                             dynamic_costs=frames_all(3, 2, 1, 1))
        with self.assertRaises(TypeError):
            plan_batch(3, 2, [], [[(0, 0), (2, 0)]],
                       dynamic_costs=frames_all(3, 2, 1, 1))


class ValidationTests(unittest.TestCase):
    def test_type_errors(self):
        bad = [
            "x",                                   # outer not a sequence
            7,                                     # outer not a sequence
            ["x"],                                 # frame not a sequence
            [[[1, True], [1, 1]]],                 # bool cell
            [[[1, 1.5], [1, 1]]],                  # non-integer cell
            [[["x", 1], [1, 1]]],                  # non-integer cell
            [[[1, 1], "x"]],                       # row not a sequence
        ]
        for dynamic_costs in bad:
            with self.subTest(dynamic_costs=dynamic_costs):
                with self.assertRaises(TypeError):
                    plan(2, 2, [], (0, 0), (1, 1),
                         dynamic_costs=dynamic_costs)

    def test_value_errors(self):
        bad = [
            [[[1, 1]]],                            # wrong row count
            [[[1], [1]]],                          # wrong column count
            [[[1, 0], [1, 1]]],                    # non-positive cost
            [[[1, -2], [1, 1]]],                   # negative cost
        ]
        for dynamic_costs in bad:
            with self.subTest(dynamic_costs=dynamic_costs):
                with self.assertRaises(ValueError):
                    plan(2, 2, [], (0, 0), (1, 1),
                         dynamic_costs=dynamic_costs)

    def test_costs_conflict_is_value_error(self):
        matrix = [[1, 1], [1, 1]]
        with self.assertRaises(ValueError):
            plan(2, 2, [], (0, 0), (1, 1), costs=matrix,
                 dynamic_costs=[matrix])
        with self.assertRaises(ValueError):
            plan_any(2, 2, [], (0, 0), [(1, 1)], costs=matrix,
                     dynamic_costs=[matrix])
        with self.assertRaises(ValueError):
            replay(2, 2, [], (0, 0), (1, 1), [(0, 0), (1, 1)],
                   costs=matrix, dynamic_costs=[matrix])

    def test_replay_validation(self):
        with self.assertRaises(TypeError):
            replay(2, 2, [], (0, 0), (1, 1), [(0, 0), (1, 1)],
                   dynamic_costs=7)
        with self.assertRaises(ValueError):
            replay(2, 2, [], (0, 0), (1, 1), [(0, 0), (1, 1)],
                   dynamic_costs=[[[1, 0], [1, 1]]])


class SearchSemanticsTests(unittest.TestCase):
    def test_frame_t_entering_cost(self):
        dynamic_costs = [
            [[1, 1, 1]],
            [[1, 5, 1]],
            [[1, 1, 7]],
        ]
        result = plan(3, 1, [], (0, 0), (2, 0), dynamic_costs=dynamic_costs)
        self.assertEqual(result["path"], [(0, 0), (1, 0), (2, 0)])
        self.assertEqual(result["cost"], 5 + 7)

    def test_last_frame_persists(self):
        dynamic_costs = [
            [[1, 1, 1], [1, 1, 1]],
            [[1, 1, 1], [1, 1, 1]],
            [[1, 1, 1], [1, 1, 1]],
            [[9, 9, 9], [9, 9, 9]],
        ]
        result = plan(3, 2, [], (0, 0), (2, 0), dynamic_costs=dynamic_costs)
        self.assertEqual(result["path"], [(0, 0), (1, 0), (2, 0)])
        self.assertEqual(result["cost"], 2)

    def test_timing_changes_the_optimal_route(self):
        # Entering (1, 0) at t == 1 and (2, 0) at t == 2 is expensive, so
        # the minimum-cost route detours through the bottom row.
        dynamic_costs = [
            [[1, 1, 1], [1, 1, 1]],
            [[1, 9, 1], [1, 1, 1]],
            [[1, 1, 9], [1, 1, 1]],
            [[1, 1, 1], [1, 1, 1]],
            [[1, 1, 1], [1, 1, 1]],
        ]
        result = plan(3, 2, [], (0, 0), (2, 0), dynamic_costs=dynamic_costs)
        self.assertEqual(result["cost"], 4)
        self.assertEqual(result["path"],
                         [(0, 0), (0, 1), (1, 1), (1, 0), (2, 0)])

    def test_trace_records_space_time_triples(self):
        dynamic_costs = frames_all(3, 2, 1, 3)
        result = plan(3, 2, [], (0, 0), (2, 0),
                      dynamic_costs=dynamic_costs, trace=True)
        self.assertTrue(all(len(node) == 3
                            for node in result["expanded_nodes"]))
        self.assertEqual(result["expanded"], len(result["expanded_nodes"]))
        self.assertEqual(result["expanded_nodes"][0], (0, 0, 0))

    def test_start_equals_goal(self):
        result = plan(2, 2, [], (1, 1), (1, 1),
                      dynamic_costs=frames_all(2, 2, 3, 2))
        self.assertEqual(result, {"path": [(1, 1)], "cost": 0,
                                  "expanded": 1})

    def test_unreachable(self):
        result = plan(3, 1, [(1, 0)], (0, 0), (2, 0),
                      dynamic_costs=frames_all(3, 1, 1, 1))
        self.assertEqual(result["path"], None)
        self.assertEqual(result["cost"], None)
        self.assertNotIn("status", result)

    def test_max_cost_caps_dynamic_accumulation(self):
        dynamic_costs = [
            [[1, 1, 1], [1, 1, 1]],
            [[1, 9, 1], [1, 1, 1]],
            [[1, 1, 9], [1, 1, 1]],
            [[1, 1, 1], [1, 1, 1]],
            [[1, 1, 1], [1, 1, 1]],
        ]
        result = plan(3, 2, [], (0, 0), (2, 0),
                      dynamic_costs=dynamic_costs, max_cost=4)
        self.assertEqual(result["status"], "found")
        self.assertEqual(result["cost"], 4)
        result = plan(3, 2, [], (0, 0), (2, 0),
                      dynamic_costs=dynamic_costs, max_cost=3)
        self.assertEqual(result["status"], "cost_exhausted")
        self.assertIsNone(result["path"])
        self.assertIsNone(result["cost"])

    def test_combined_with_dynamic_blocked(self):
        dynamic_blocked = [set(), {(1, 0)}, set(), set(), set()]
        dynamic_costs = frames_all(3, 2, 1, 5)
        result = plan(3, 2, [], (0, 0), (2, 0),
                      dynamic_blocked=dynamic_blocked,
                      dynamic_costs=dynamic_costs)
        self.assertEqual(result["path"],
                         [(0, 0), (0, 1), (1, 1), (1, 0), (2, 0)])
        self.assertEqual(result["cost"], 4)

    def test_waits_cost_nothing(self):
        dynamic_blocked = [set(), {(1, 0)}, set()]
        dynamic_costs = [
            [[1, 1, 1], [1, 1, 1]],
            [[1, 1, 1], [1, 1, 1]],
            [[5, 5, 5], [5, 5, 5]],
        ]
        result = plan(3, 2, [], (0, 0), (2, 0),
                      dynamic_blocked=dynamic_blocked,
                      dynamic_costs=dynamic_costs, allow_wait=True)
        self.assertIsNotNone(result["path"])
        checked = replay(3, 2, [], (0, 0), (2, 0), result["path"],
                         dynamic_blocked=dynamic_blocked,
                         dynamic_costs=dynamic_costs, allow_wait=True)
        self.assertTrue(checked["valid"])
        self.assertEqual(checked["cost"], result["cost"])
        self.assertEqual(checked["steps"], len(result["path"]) - 1)


class PlanAnyTests(unittest.TestCase):
    def test_winning_endpoint_follows_frame_costs(self):
        dynamic_costs = [
            [[1, 1, 1], [1, 1, 1]],
            [[1, 9, 1], [1, 1, 1]],
            [[1, 1, 9], [1, 1, 1]],
        ]
        result = plan_any(3, 2, [], (0, 0), [(2, 0), (0, 1)],
                          dynamic_costs=dynamic_costs)
        self.assertEqual(result["path"], [(0, 0), (0, 1)])
        self.assertEqual(result["cost"], 1)

    def test_start_in_goals(self):
        result = plan_any(2, 2, [], (1, 1), [(1, 1), (0, 0)],
                          dynamic_costs=frames_all(2, 2, 2, 1))
        self.assertEqual(result["path"], [(1, 1)])
        self.assertEqual(result["cost"], 0)

    def test_replay_verifies_winning_path(self):
        dynamic_costs = [
            [[1, 1, 1], [1, 1, 1]],
            [[1, 9, 1], [1, 1, 1]],
            [[1, 1, 9], [1, 1, 1]],
            [[1, 1, 1], [1, 1, 1]],
            [[1, 1, 1], [1, 1, 1]],
        ]
        result = plan_any(3, 2, [], (0, 0), [(2, 0), (2, 1)],
                          dynamic_costs=dynamic_costs)
        goal = result["path"][-1]
        checked = replay(3, 2, [], (0, 0), goal, result["path"],
                         dynamic_costs=dynamic_costs)
        self.assertEqual(checked, {"valid": True, "cost": result["cost"],
                                   "steps": len(result["path"]) - 1})


class ReplayTests(unittest.TestCase):
    def test_replay_recomputes_with_the_same_frames(self):
        dynamic_costs = [
            [[1, 1, 1]],
            [[1, 5, 1]],
            [[1, 1, 7]],
        ]
        path = [(0, 0), (1, 0), (2, 0)]
        result = replay(3, 1, [], (0, 0), (2, 0), path,
                        dynamic_costs=dynamic_costs)
        self.assertEqual(result, {"valid": True, "cost": 12, "steps": 2})

    def test_replay_diagnose_unchanged(self):
        dynamic_costs = frames_all(3, 1, 2, 3)
        result = replay(3, 1, [(1, 0)], (0, 0), (2, 0),
                        [(0, 0), (1, 0), (2, 0)],
                        dynamic_costs=dynamic_costs, diagnose=True)
        self.assertEqual(result["error"], "static_blocked")
        self.assertEqual(result["error_index"], 1)
        self.assertIsNone(result["cost"])
        self.assertIsNone(result["steps"])


class SnapshotResumeTests(unittest.TestCase):
    def setUp(self):
        self.dynamic_costs = [
            [[1 + ((x + y + t) % 3) for x in range(4)] for y in range(3)]
            for t in range(6)
        ]

    def test_checkpoint_records_dynamic_costs_json(self):
        result = plan(4, 3, [(1, 1)], (0, 0), (3, 2),
                      dynamic_costs=self.dynamic_costs,
                      max_expanded=3, snapshot=True)
        self.assertEqual(result["status"], "budget_exhausted")
        checkpoint = result["checkpoint"]
        json.dumps(checkpoint)  # must be JSON-serializable
        self.assertEqual(checkpoint["dynamic_costs"],
                         [[list(row) for row in frame]
                          for frame in self.dynamic_costs])

    def test_checkpoint_omits_field_without_dynamic_costs(self):
        result = plan(4, 3, [(1, 1)], (0, 0), (3, 2),
                      max_expanded=3, snapshot=True)
        self.assertNotIn("dynamic_costs", result["checkpoint"])

    def test_resume_matches_uninterrupted_call(self):
        full = plan(4, 3, [(1, 1)], (0, 0), (3, 2),
                    dynamic_costs=self.dynamic_costs,
                    trace=True, max_expanded=10 ** 9)
        part = plan(4, 3, [(1, 1)], (0, 0), (3, 2),
                    dynamic_costs=self.dynamic_costs,
                    max_expanded=3, snapshot=True)
        resumed = resume(part["checkpoint"], max_expanded=10 ** 9)
        for key in ("path", "cost", "expanded", "status",
                    "expanded_nodes"):
            self.assertEqual(resumed[key], full[key], key)

    def test_chained_resume_matches_same_cumulative_budget(self):
        checkpoint = plan(4, 3, [(1, 1)], (0, 0), (3, 2),
                          dynamic_costs=self.dynamic_costs,
                          max_expanded=2, snapshot=True)["checkpoint"]
        budget = 2
        while True:
            out = resume(checkpoint, max_expanded=budget + 2)
            budget += 2
            if "checkpoint" not in out:
                break
            checkpoint = out["checkpoint"]
        one_shot = plan(4, 3, [(1, 1)], (0, 0), (3, 2),
                        dynamic_costs=self.dynamic_costs,
                        max_expanded=budget, trace=True)
        for key in ("path", "cost", "expanded", "status",
                    "expanded_nodes"):
            self.assertEqual(out[key], one_shot[key], key)

    def test_plan_any_resume(self):
        full = plan_any(4, 3, [(1, 1)], (0, 0), [(3, 2), (3, 0)],
                        dynamic_costs=self.dynamic_costs,
                        trace=True, max_expanded=10 ** 9)
        part = plan_any(4, 3, [(1, 1)], (0, 0), [(3, 2), (3, 0)],
                        dynamic_costs=self.dynamic_costs,
                        max_expanded=2, snapshot=True)
        resumed = resume(part["checkpoint"], max_expanded=10 ** 9)
        for key in ("path", "cost", "expanded", "status",
                    "expanded_nodes"):
            self.assertEqual(resumed[key], full[key], key)

    def _checkpoint(self):
        result = plan(4, 3, [(1, 1)], (0, 0), (3, 2),
                      dynamic_costs=self.dynamic_costs,
                      max_expanded=3, snapshot=True)
        return json.loads(json.dumps(result["checkpoint"]))

    def test_corrupt_dynamic_costs_type(self):
        checkpoint = self._checkpoint()
        checkpoint["dynamic_costs"] = "nope"
        with self.assertRaises(TypeError):
            resume(checkpoint)

    def test_corrupt_dynamic_costs_shape(self):
        checkpoint = self._checkpoint()
        checkpoint["dynamic_costs"] = [[[1, 1, 1, 1]]]
        with self.assertRaises(ValueError):
            resume(checkpoint)

    def test_corrupt_dynamic_costs_nonpositive(self):
        checkpoint = self._checkpoint()
        checkpoint["dynamic_costs"][0][0][0] = 0
        with self.assertRaises(ValueError):
            resume(checkpoint)

    def test_corrupt_dynamic_costs_conflict(self):
        checkpoint = self._checkpoint()
        checkpoint["costs"] = [[1] * 4 for _ in range(3)]
        with self.assertRaises(ValueError):
            resume(checkpoint)

    def test_corrupt_dynamic_costs_inconsistent_state(self):
        checkpoint = self._checkpoint()
        checkpoint["dynamic_costs"][1] = [[99] * 4 for _ in range(3)]
        with self.assertRaises(ValueError):
            resume(checkpoint)

    def test_dynamic_costs_rejected_for_multi_start_checkpoint(self):
        result = plan_multi_start(4, 3, [(1, 1)], [(0, 0)], (3, 2),
                                  max_expanded=2, snapshot=True)
        if "checkpoint" not in result:
            self.skipTest("search finished before the budget stopped it")
        checkpoint = json.loads(json.dumps(result["checkpoint"]))
        checkpoint["dynamic_costs"] = self.dynamic_costs
        with self.assertRaises(ValueError):
            resume(checkpoint)


if __name__ == "__main__":
    unittest.main()
