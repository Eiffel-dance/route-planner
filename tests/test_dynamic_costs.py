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

    def test_other_entry_points_omitted_none_and_empty_are_equivalent(self):
        base_k = plan_k(3, 2, [], (0, 0), (2, 0), 2)
        base_multi = plan_multi_start(3, 2, [], [(0, 0)], (2, 0), trace=True)
        base_batch = plan_batch(3, 2, [], [[(0, 0), (2, 0)]], trace=True)
        for dynamic_costs in (None, []):
            self.assertEqual(
                plan_k(3, 2, [], (0, 0), (2, 0), 2,
                       dynamic_costs=dynamic_costs), base_k)
            self.assertEqual(
                plan_multi_start(3, 2, [], [(0, 0)], (2, 0), trace=True,
                                 dynamic_costs=dynamic_costs), base_multi)
            self.assertEqual(
                plan_batch(3, 2, [], [[(0, 0), (2, 0)]], trace=True,
                           dynamic_costs=dynamic_costs), base_batch)

    def test_empty_sequence_does_not_conflict_with_costs_anywhere(self):
        matrix = [[1, 1], [1, 1]]
        self.assertEqual(
            plan_k(2, 2, [], (0, 0), (1, 1), 1, costs=matrix,
                   dynamic_costs=[])["costs"], [2])
        self.assertEqual(
            plan_multi_start(2, 2, [], [(0, 0)], (1, 1), costs=matrix,
                             dynamic_costs=[])["cost"], 2)
        self.assertEqual(
            plan_batch(2, 2, [], [[(0, 0), (1, 1)]], costs=matrix,
                       dynamic_costs=[])["results"][0]["cost"], 2)


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

    def test_dynamic_costs_injected_into_static_multi_start_checkpoint(self):
        # A checkpoint recorded without time-varying costs cannot gain
        # them afterwards: the stored pair trace and unit-cost state can
        # never belong to a time-expanded search, so the restore rejects
        # the inconsistent snapshot before any searching.
        result = plan_multi_start(4, 3, [(1, 1)], [(0, 0)], (3, 2),
                                  max_expanded=2, snapshot=True)
        if "checkpoint" not in result:
            self.skipTest("search finished before the budget stopped it")
        checkpoint = json.loads(json.dumps(result["checkpoint"]))
        checkpoint["dynamic_costs"] = self.dynamic_costs
        with self.assertRaises((TypeError, ValueError)):
            resume(checkpoint)


class PlanBatchDynamicCostsTests(unittest.TestCase):
    def setUp(self):
        self.dynamic_costs = [
            [[1, 1, 1], [1, 1, 1]],
            [[1, 9, 1], [1, 1, 1]],
            [[1, 1, 9], [1, 1, 1]],
            [[1, 1, 1], [1, 1, 1]],
            [[1, 1, 1], [1, 1, 1]],
        ]

    def test_results_match_independent_plan_calls(self):
        requests = [[(0, 0), (2, 0)], [(2, 1), (0, 1)], [(0, 0), (0, 0)],
                    [(0, 0), (2, 0)]]
        batch = plan_batch(3, 2, [], requests,
                           dynamic_costs=self.dynamic_costs, trace=True)
        self.assertEqual(len(batch["results"]), len(requests))
        for result, (start, goal) in zip(batch["results"], requests):
            expected = plan(3, 2, [], start, goal,
                            dynamic_costs=self.dynamic_costs, trace=True)
            self.assertEqual(result, expected)

    def test_budget_and_cost_limit_apply_per_query(self):
        requests = [[(0, 0), (2, 0)], [(2, 1), (0, 1)]]
        for limits in ({"max_expanded": 4}, {"max_cost": 4},
                       {"max_expanded": 4, "max_cost": 4}):
            batch = plan_batch(3, 2, [], requests,
                               dynamic_costs=self.dynamic_costs, **limits)
            for result, (start, goal) in zip(batch["results"], requests):
                expected = plan(3, 2, [], start, goal,
                                dynamic_costs=self.dynamic_costs, **limits)
                self.assertEqual(result, expected)

    def test_validation_errors(self):
        with self.assertRaises(TypeError):
            plan_batch(2, 2, [], [[(0, 0), (1, 1)]], dynamic_costs="x")
        with self.assertRaises(TypeError):
            plan_batch(2, 2, [], [[(0, 0), (1, 1)]],
                       dynamic_costs=[[[1, True], [1, 1]]])
        with self.assertRaises(ValueError):
            plan_batch(2, 2, [], [[(0, 0), (1, 1)]],
                       dynamic_costs=[[[1, 0], [1, 1]]])
        with self.assertRaises(ValueError):
            plan_batch(2, 2, [], [[(0, 0), (1, 1)]],
                       dynamic_costs=[[[1, 1]]])
        with self.assertRaises(ValueError):
            plan_batch(2, 2, [], [[(0, 0), (1, 1)]],
                       costs=[[1, 1], [1, 1]],
                       dynamic_costs=[[[1, 1], [1, 1]]])

    def test_all_requests_validated_before_any_search(self):
        # The bad request is reported even though the first request is
        # fine; no partial results are ever returned.
        with self.assertRaises(ValueError):
            plan_batch(2, 2, [], [[(0, 0), (1, 1)], [(5, 5), (0, 0)]],
                       dynamic_costs=frames_all(2, 2, 1, 2))
        # The requests checks still precede the dynamic_costs checks.
        with self.assertRaises(ValueError):
            plan_batch(2, 2, [], [], dynamic_costs="x")

    def test_per_request_checkpoint_records_and_resumes(self):
        result = plan_batch(4, 3, [(1, 1)], [[(0, 0), (3, 2)]],
                            dynamic_costs=frames_all(4, 3, 2, 6),
                            max_expanded=3, snapshot=True)
        inner = result["results"][0]
        self.assertEqual(inner["status"], "budget_exhausted")
        checkpoint = inner["checkpoint"]
        json.dumps(checkpoint)  # must be JSON-serializable
        self.assertEqual(checkpoint["planner"], "plan")
        self.assertEqual(checkpoint["dynamic_costs"],
                         [[[2] * 4 for _ in range(3)] for _ in range(6)])
        full = plan(4, 3, [(1, 1)], (0, 0), (3, 2),
                    dynamic_costs=frames_all(4, 3, 2, 6),
                    trace=True, max_expanded=10 ** 9)
        resumed = resume(checkpoint, max_expanded=10 ** 9)
        for key in ("path", "cost", "expanded", "status",
                    "expanded_nodes"):
            self.assertEqual(resumed[key], full[key], key)


class PlanMultiStartDynamicCostsTests(unittest.TestCase):
    def setUp(self):
        # Entering (1, 0) at t == 1 is expensive; every other cell and
        # frame costs 1.
        self.dynamic_costs = [
            [[1, 1], [1, 1], [1, 1]],
            [[1, 9], [1, 1], [1, 1]],
            [[1, 1], [1, 1], [1, 1]],
            [[1, 1], [1, 1], [1, 1]],
        ]

    def test_dynamic_costs_pick_the_winner(self):
        # Statically (0, 0) wins with cost 1; with the time-varying
        # frames its direct step costs 9 and its best detour costs 3,
        # so (1, 2) wins with cost 2.
        static = plan_multi_start(2, 3, [], [(0, 0), (1, 2)], (1, 0))
        self.assertEqual(static["path"], [(0, 0), (1, 0)])
        result = plan_multi_start(2, 3, [], [(0, 0), (1, 2)], (1, 0),
                                  dynamic_costs=self.dynamic_costs)
        self.assertEqual(result["path"], [(1, 2), (1, 1), (1, 0)])
        self.assertEqual(result["cost"], 2)
        expected = plan(2, 3, [], (1, 2), (1, 0),
                        dynamic_costs=self.dynamic_costs)
        self.assertEqual(result["path"], expected["path"])
        self.assertEqual(result["cost"], expected["cost"])

    def test_trace_records_space_time_triples(self):
        result = plan_multi_start(2, 3, [], [(0, 0), (1, 2)], (1, 0),
                                  dynamic_costs=self.dynamic_costs,
                                  trace=True)
        self.assertTrue(all(len(node) == 3
                            for node in result["expanded_nodes"]))
        self.assertEqual(result["expanded"], len(result["expanded_nodes"]))

    def test_start_equals_goal(self):
        result = plan_multi_start(2, 2, [], [(1, 1), (0, 0)], (1, 1),
                                  dynamic_costs=frames_all(2, 2, 3, 2))
        self.assertEqual(result, {"path": [(1, 1)], "cost": 0,
                                  "expanded": 1})

    def test_validation_errors(self):
        starts, goal = [(0, 0)], (1, 1)
        with self.assertRaises(TypeError):
            plan_multi_start(2, 2, [], starts, goal, dynamic_costs=7)
        with self.assertRaises(TypeError):
            plan_multi_start(2, 2, [], starts, goal,
                             dynamic_costs=[[[1, 1.5], [1, 1]]])
        with self.assertRaises(ValueError):
            plan_multi_start(2, 2, [], starts, goal,
                             dynamic_costs=[[[1, -1], [1, 1]]])
        with self.assertRaises(ValueError):
            plan_multi_start(2, 2, [], starts, goal,
                             dynamic_costs=[[[1], [1]]])
        with self.assertRaises(ValueError):
            plan_multi_start(2, 2, [], starts, goal,
                             costs=[[1, 1], [1, 1]],
                             dynamic_costs=[[[1, 1], [1, 1]]])
        # The starts checks still precede the dynamic_costs checks.
        with self.assertRaises(ValueError):
            plan_multi_start(2, 2, [], [], goal, dynamic_costs="x")

    def test_checkpoint_records_dynamic_costs_and_resumes(self):
        dynamic_costs = [
            [[1 + ((x + y + t) % 3) for x in range(4)] for y in range(3)]
            for t in range(6)
        ]
        full = plan_multi_start(4, 3, [(1, 1)], [(0, 0), (3, 0)], (3, 2),
                                dynamic_costs=dynamic_costs,
                                trace=True, max_expanded=10 ** 9)
        part = plan_multi_start(4, 3, [(1, 1)], [(0, 0), (3, 0)], (3, 2),
                                dynamic_costs=dynamic_costs,
                                max_expanded=2, snapshot=True)
        self.assertEqual(part["status"], "budget_exhausted")
        checkpoint = part["checkpoint"]
        json.dumps(checkpoint)  # must be JSON-serializable
        self.assertEqual(checkpoint["dynamic_costs"],
                         [[list(row) for row in frame]
                          for frame in dynamic_costs])
        resumed = resume(checkpoint, max_expanded=10 ** 9)
        for key in ("path", "cost", "expanded", "status",
                    "expanded_nodes"):
            self.assertEqual(resumed[key], full[key], key)

    def test_corrupt_multi_start_checkpoint(self):
        dynamic_costs = [
            [[1 + ((x + y + t) % 3) for x in range(4)] for y in range(3)]
            for t in range(6)
        ]
        part = plan_multi_start(4, 3, [(1, 1)], [(0, 0), (3, 0)], (3, 2),
                                dynamic_costs=dynamic_costs,
                                max_expanded=2, snapshot=True)
        self.assertEqual(part["status"], "budget_exhausted")
        base = json.loads(json.dumps(part["checkpoint"]))

        checkpoint = dict(base, dynamic_costs="nope")
        with self.assertRaises(TypeError):
            resume(checkpoint)

        checkpoint = dict(base, dynamic_costs=[[[1, 1, 1, 1]]])
        with self.assertRaises(ValueError):
            resume(checkpoint)

        checkpoint = json.loads(json.dumps(base))
        checkpoint["dynamic_costs"][0][0][0] = 0
        with self.assertRaises(ValueError):
            resume(checkpoint)

        checkpoint = dict(base, costs=[[1] * 4 for _ in range(3)])
        with self.assertRaises(ValueError):
            resume(checkpoint)

        checkpoint = json.loads(json.dumps(base))
        checkpoint["dynamic_costs"][1] = [[99] * 4 for _ in range(3)]
        with self.assertRaises(ValueError):
            resume(checkpoint)


class PlanKDynamicCostsTests(unittest.TestCase):
    def setUp(self):
        self.dynamic_costs = [
            [[1, 1, 1], [1, 1, 1]],
            [[1, 9, 1], [1, 1, 1]],
            [[1, 1, 9], [1, 1, 1]],
            [[1, 1, 1], [1, 1, 1]],
            [[1, 1, 1], [1, 1, 1]],
        ]

    def test_routes_ranked_by_dynamic_cost(self):
        result = plan_k(3, 2, [], (0, 0), (2, 0), 3,
                        dynamic_costs=self.dynamic_costs)
        self.assertEqual(result["paths"], [
            [(0, 0), (0, 1), (1, 1), (1, 0), (2, 0)],
            [(0, 0), (0, 1), (1, 1), (2, 1), (2, 0)],
            [(0, 0), (1, 0), (1, 1), (2, 1), (2, 0)],
        ])
        self.assertEqual(result["costs"], [4, 4, 12])
        # The first entry is the route ``plan`` returns.
        first = plan(3, 2, [], (0, 0), (2, 0),
                     dynamic_costs=self.dynamic_costs)
        self.assertEqual(result["paths"][0], first["path"])
        self.assertEqual(result["costs"][0], first["cost"])

    def test_every_route_verifies_in_replay(self):
        result = plan_k(3, 2, [], (0, 0), (2, 0), 3,
                        dynamic_costs=self.dynamic_costs)
        for path, cost in zip(result["paths"], result["costs"]):
            checked = replay(3, 2, [], (0, 0), (2, 0), path,
                             dynamic_costs=self.dynamic_costs)
            self.assertEqual(checked, {"valid": True, "cost": cost,
                                       "steps": len(path) - 1})

    def test_max_cost_caps_dynamic_accumulation(self):
        result = plan_k(3, 2, [], (0, 0), (2, 0), 3,
                        dynamic_costs=self.dynamic_costs, max_cost=4)
        self.assertEqual(result["status"], "found")
        self.assertEqual(result["costs"], [4, 4])
        result = plan_k(3, 2, [], (0, 0), (2, 0), 3,
                        dynamic_costs=self.dynamic_costs, max_cost=3)
        self.assertEqual(result["status"], "cost_exhausted")
        self.assertEqual(result["paths"], [])
        self.assertEqual(result["costs"], [])

    def test_budget_truncation(self):
        limited = plan_k(3, 2, [], (0, 0), (2, 0), 3,
                         dynamic_costs=self.dynamic_costs, max_expanded=3)
        self.assertEqual(limited["status"], "budget_exhausted")
        self.assertEqual(limited["expanded"], 3)
        # The budget only truncates: the returned prefix matches the
        # unlimited ranking.
        full = plan_k(3, 2, [], (0, 0), (2, 0), 3,
                      dynamic_costs=self.dynamic_costs)
        self.assertEqual(limited["paths"],
                         full["paths"][:len(limited["paths"])])

    def test_start_equals_goal(self):
        result = plan_k(2, 2, [], (1, 1), (1, 1), 3,
                        dynamic_costs=frames_all(2, 2, 2, 2))
        self.assertEqual(result, {"paths": [[(1, 1)]], "costs": [0],
                                  "expanded": 1})

    def test_validation_errors(self):
        with self.assertRaises(TypeError):
            plan_k(2, 2, [], (0, 0), (1, 1), 1, dynamic_costs="x")
        with self.assertRaises(TypeError):
            plan_k(2, 2, [], (0, 0), (1, 1), 1,
                   dynamic_costs=[[[1, True], [1, 1]]])
        with self.assertRaises(ValueError):
            plan_k(2, 2, [], (0, 0), (1, 1), 1,
                   dynamic_costs=[[[1, 0], [1, 1]]])
        with self.assertRaises(ValueError):
            plan_k(2, 2, [], (0, 0), (1, 1), 1,
                   dynamic_costs=[[[1, 1]]])
        with self.assertRaises(ValueError):
            plan_k(2, 2, [], (0, 0), (1, 1), 1,
                   costs=[[1, 1], [1, 1]],
                   dynamic_costs=[[[1, 1], [1, 1]]])

    def test_validation_order(self):
        # The k check still precedes the dynamic_costs checks.
        with self.assertRaises(ValueError):
            plan_k(2, 2, [], (0, 0), (1, 1), 0, dynamic_costs="x")
        with self.assertRaises(TypeError):
            plan_k(2, 2, [], (0, 0), (1, 1), "x", dynamic_costs="x")
        # The limit checks still precede the dynamic_costs checks.
        with self.assertRaises(ValueError):
            plan_k(2, 2, [], (0, 0), (1, 1), 1, max_cost=-1,
                   dynamic_costs="x")
        with self.assertRaises(TypeError):
            plan_k(2, 2, [], (0, 0), (1, 1), 1, max_expanded="x",
                   dynamic_costs="x")


if __name__ == "__main__":
    unittest.main()
