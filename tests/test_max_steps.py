import json
import unittest

from app import (plan, plan_any, plan_k, plan_batch, plan_multi_start,
                 replay, resume, verify_trace, verify_any_trace,
                 verify_batch_trace)


def without_checkpoint(result):
    return {key: value for key, value in result.items()
            if key != "checkpoint"}


def smallest_exhausting_budget(call, args, kwargs):
    # The smallest budget under which ``call`` stops with
    # ``budget_exhausted`` (``0`` always qualifies for a non-empty search).
    full = call(*args, trace=True, **kwargs)
    budget = 0
    while budget <= full["expanded"]:
        result = call(*args, trace=True, max_expanded=budget,
                      snapshot=True, **kwargs)
        if result["status"] == "budget_exhausted":
            return budget, full
        budget += 1
    raise AssertionError("no budget exhausted the search")


class MaxStepsValidationTest(unittest.TestCase):
    def test_type_errors(self):
        for bad in ("5", 1.5, True, False, [5], (5,), 2.0):
            with self.assertRaises(TypeError, msg=f"max_steps={bad!r}"):
                plan(3, 3, set(), (0, 0), (2, 2), max_steps=bad)
            with self.assertRaises(TypeError, msg=f"max_steps={bad!r}"):
                plan_any(3, 3, set(), (0, 0), [(2, 2)], max_steps=bad)
            with self.assertRaises(TypeError, msg=f"max_steps={bad!r}"):
                plan_k(3, 3, set(), (0, 0), (2, 2), 3, max_steps=bad)
            with self.assertRaises(TypeError, msg=f"max_steps={bad!r}"):
                plan_batch(3, 3, set(), [[(0, 0), (2, 2)]], max_steps=bad)
            with self.assertRaises(TypeError, msg=f"max_steps={bad!r}"):
                plan_multi_start(3, 3, set(), [(0, 0)], (2, 2),
                                 max_steps=bad)
            with self.assertRaises(TypeError, msg=f"max_steps={bad!r}"):
                replay(3, 3, set(), (0, 0), (2, 2),
                       [(0, 0), (1, 0)], max_steps=bad)

    def test_negative_is_value_error(self):
        for bad in (-1, -100):
            with self.assertRaises(ValueError, msg=f"max_steps={bad!r}"):
                plan(3, 3, set(), (0, 0), (2, 2), max_steps=bad)
            with self.assertRaises(ValueError, msg=f"max_steps={bad!r}"):
                plan_any(3, 3, set(), (0, 0), [(2, 2)], max_steps=bad)
            with self.assertRaises(ValueError, msg=f"max_steps={bad!r}"):
                plan_k(3, 3, set(), (0, 0), (2, 2), 3, max_steps=bad)
            with self.assertRaises(ValueError, msg=f"max_steps={bad!r}"):
                plan_batch(3, 3, set(), [[(0, 0), (2, 2)]], max_steps=bad)
            with self.assertRaises(ValueError, msg=f"max_steps={bad!r}"):
                plan_multi_start(3, 3, set(), [(0, 0)], (2, 2),
                                 max_steps=bad)

    def test_zero_is_accepted(self):
        result = plan(3, 3, set(), (1, 1), (1, 1), max_steps=0)
        self.assertEqual(result["status"], "found")
        self.assertEqual(result["path"], [(1, 1)])
        self.assertEqual(result["cost"], 0)
        self.assertEqual(result["expanded"], 1)

    def test_validated_after_all_existing_checks(self):
        with self.assertRaises(ValueError):
            plan(0, 3, set(), (0, 0), (1, 1), max_steps="x")
        with self.assertRaises(ValueError):
            plan(2, 2, set(), (0, 0), (1, 1), costs=[[1]], max_steps="x")
        with self.assertRaises(TypeError):
            plan(3, 3, set(), (0, 0), (2, 2), trace=1, max_steps=-1)
        with self.assertRaises(TypeError):
            plan(3, 3, set(), (0, 0), (2, 2), dynamic_blocked="x",
                 max_steps="x")
        with self.assertRaises(ValueError):
            plan(3, 3, set(), (0, 0), (2, 2), max_expanded=-1,
                 max_steps="x")
        with self.assertRaises(TypeError):
            plan(3, 3, set(), (0, 0), (2, 2), snapshot=1, max_steps=-1)
        with self.assertRaises(ValueError):
            plan(3, 3, set(), (0, 0), (2, 2), max_cost=-1, max_steps="x")
        with self.assertRaises(ValueError):
            plan(3, 3, set(), (0, 0), (2, 2),
                 costs=[[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                 dynamic_costs=[[[1, 1, 1], [1, 1, 1], [1, 1, 1]]],
                 max_steps="x")
        # plan_k validates k before the step window.
        with self.assertRaises(ValueError):
            plan_k(3, 3, set(), (0, 0), (2, 2), 0, max_steps="x")

    def test_batch_never_returns_partial_results(self):
        with self.assertRaises(TypeError):
            plan_batch(3, 3, set(), [[(0, 0), (2, 2)]], max_steps=True)
        with self.assertRaises(ValueError):
            plan_batch(3, 3, set(), [[(0, 0), (2, 2)]], max_steps=-1)

    def test_omitted_or_none_keeps_legacy_shape(self):
        for kwargs in ({}, {"max_steps": None}):
            legacy = plan(5, 3, {(2, 0), (2, 1)}, (0, 0), (4, 2),
                          trace=True, **kwargs)
            self.assertEqual(set(legacy),
                             {"path", "cost", "expanded", "expanded_nodes"})
            self.assertNotIn("status", legacy)
            k_result = plan_k(3, 3, set(), (0, 0), (2, 2), 3, **kwargs)
            self.assertEqual(set(k_result), {"paths", "costs", "expanded"})


class MaxStepsSearchTest(unittest.TestCase):
    def test_zero_window_start_equals_goal(self):
        result = plan(3, 3, set(), (1, 1), (1, 1), max_steps=0, trace=True)
        self.assertEqual(result, {"path": [(1, 1)], "cost": 0,
                                  "expanded": 1, "status": "found",
                                  "expanded_nodes": [(1, 1)]})

    def test_zero_window_different_endpoint(self):
        result = plan(4, 1, set(), (0, 0), (3, 0), max_steps=0, trace=True)
        self.assertEqual(result["status"], "step_exhausted")
        self.assertIsNone(result["path"])
        self.assertIsNone(result["cost"])
        self.assertEqual(result["expanded"], 1)
        self.assertEqual(result["expanded_nodes"], [(0, 0)])

    def test_window_caps_path_length(self):
        result = plan(3, 3, set(), (0, 0), (2, 2), max_steps=4, trace=True)
        self.assertEqual(result["status"], "found")
        self.assertEqual(len(result["path"]), 5)
        self.assertEqual(len(result["path"]) - 1, 4)
        self.assertEqual(result["expanded"],
                         len(result["expanded_nodes"]))

    def test_window_one_step_too_short(self):
        result = plan(3, 3, set(), (0, 0), (2, 2), max_steps=3, trace=True)
        self.assertEqual(result["status"], "step_exhausted")
        self.assertIsNone(result["path"])
        self.assertIsNone(result["cost"])
        # Exactly the simple monotone prefixes fit in three steps.
        self.assertEqual(result["expanded"], 15)
        self.assertEqual(result["expanded"],
                         len(result["expanded_nodes"]))

    def test_candidates_at_the_limit_are_not_expanded(self):
        result = plan(4, 1, set(), (0, 0), (3, 0), max_steps=1, trace=True)
        self.assertEqual(result["status"], "step_exhausted")
        self.assertEqual(result["expanded_nodes"], [(0, 0), (1, 0)])
        dynamic = plan(3, 3, set(), (0, 0), (2, 2),
                       dynamic_blocked=[[], [], [], []],
                       max_steps=2, trace=True)
        self.assertEqual(dynamic["status"], "step_exhausted")
        self.assertTrue(all(t <= 2 for _x, _y, t
                            in dynamic["expanded_nodes"]))

    def test_waits_count_as_steps(self):
        frames = [[], [(1, 0)], []]
        feasible = plan(2, 1, set(), (0, 0), (1, 0),
                        dynamic_blocked=frames, allow_wait=True,
                        max_steps=2)
        self.assertEqual(feasible["status"], "found")
        self.assertEqual(feasible["path"], [(0, 0), (0, 0), (1, 0)])
        checked = replay(2, 1, set(), (0, 0), (1, 0), feasible["path"],
                         dynamic_blocked=frames, allow_wait=True,
                         max_steps=2)
        self.assertEqual(checked, {"valid": True, "cost": 1, "steps": 2})
        tight = plan(2, 1, set(), (0, 0), (1, 0),
                     dynamic_blocked=frames, allow_wait=True, max_steps=1)
        self.assertEqual(tight["status"], "step_exhausted")

    def test_consecutive_waits_in_static_mode(self):
        # Waiting needs provided frames, so it is only exercised there.
        # The goal is blocked at frames 1 and 2, forcing two consecutive
        # waits before the move on frame 3.
        frames = [[], [(1, 0)], [(1, 0)], []]
        result = plan(2, 1, set(), (0, 0), (1, 0),
                      dynamic_blocked=frames, allow_wait=True,
                      max_steps=3, trace=True)
        self.assertEqual(result["status"], "found")
        self.assertEqual(len(result["path"]) - 1, 3)
        self.assertEqual(result["path"],
                         [(0, 0), (0, 0), (0, 0), (1, 0)])

    def test_generous_window_keeps_path_and_cost(self):
        blocked = {(1, 2), (2, 1)}
        limited = plan(4, 4, blocked, (0, 0), (3, 3), max_steps=100)
        legacy = plan(4, 4, blocked, (0, 0), (3, 3))
        self.assertEqual(limited["path"], legacy["path"])
        self.assertEqual(limited["cost"], legacy["cost"])
        self.assertEqual(limited["status"], "found")

    def test_budget_has_priority_over_step_window(self):
        result = plan(3, 3, set(), (0, 0), (2, 2), max_expanded=2,
                      max_steps=1)
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(result["expanded"], 2)

    def test_step_window_has_priority_over_cost_limit(self):
        result = plan(3, 3, set(), (0, 0), (2, 2), max_cost=1,
                      max_steps=1)
        self.assertEqual(result["status"], "step_exhausted")
        cost_only = plan(3, 3, set(), (0, 0), (2, 2), max_cost=1,
                         max_steps=100)
        self.assertEqual(cost_only["status"], "cost_exhausted")

    def test_unreachable_without_cutoff_stays_unreachable(self):
        wall = {(1, y) for y in range(3)}
        result = plan(3, 3, wall, (0, 0), (2, 2), max_steps=100)
        self.assertEqual(result["status"], "unreachable")
        self.assertIsNone(result["path"])

    def test_zero_budget_rule_still_wins(self):
        result = plan(3, 3, set(), (1, 1), (1, 1), max_expanded=0,
                      max_steps=0)
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(result["expanded"], 0)

    def test_static_trace_records_pairs_dynamic_triples(self):
        static = plan(3, 3, set(), (0, 0), (2, 2), max_steps=4, trace=True)
        self.assertTrue(all(len(node) == 2 for node
                            in static["expanded_nodes"]))
        dynamic = plan(3, 3, set(), (0, 0), (2, 2),
                       dynamic_blocked=[[], [], [], []],
                       max_steps=4, trace=True)
        self.assertTrue(all(len(node) == 3 for node
                            in dynamic["expanded_nodes"]))

    def test_reservations_and_dynamic_costs_keep_their_semantics(self):
        reservations = [[(2, 2), (2, 1)]]
        result = plan(3, 3, set(), (0, 0), (2, 2),
                      reservations=reservations, max_steps=6)
        self.assertEqual(result["status"], "found")
        self.assertEqual(result["path"][-1], (2, 2))
        checked = replay(3, 3, set(), (0, 0), (2, 2), result["path"],
                         reservations=reservations, max_steps=6)
        self.assertTrue(checked["valid"])
        frames = [[[1, 1, 1], [1, 1, 1], [1, 1, 1]]]
        costed = plan(3, 3, set(), (0, 0), (2, 2),
                      dynamic_costs=frames, max_steps=4)
        self.assertEqual(costed["status"], "found")
        self.assertEqual(costed["cost"], 4)

    def test_results_are_json_serializable(self):
        result = plan(3, 3, set(), (0, 0), (2, 2), max_steps=3, trace=True)
        json.loads(json.dumps(result))
        found = plan(3, 3, set(), (0, 0), (2, 2), max_steps=4, trace=True)
        json.loads(json.dumps(found))


class MaxStepsPlanAnyTest(unittest.TestCase):
    def test_tie_break_and_endpoint_unchanged(self):
        goals = [(2, 0), (0, 2)]
        unlimited = plan_any(3, 3, set(), (0, 0), goals)
        limited = plan_any(3, 3, set(), (0, 0), goals, max_steps=100)
        self.assertEqual(limited["path"], unlimited["path"])
        self.assertEqual(limited["cost"], unlimited["cost"])

    def test_too_short_window(self):
        result = plan_any(3, 3, set(), (0, 0), [(2, 2)], max_steps=3)
        self.assertEqual(result["status"], "step_exhausted")
        self.assertIsNone(result["path"])

    def test_start_in_goals_zero_window(self):
        result = plan_any(3, 3, set(), (1, 1), [(1, 1), (2, 2)],
                          max_steps=0)
        self.assertEqual(result, {"path": [(1, 1)], "cost": 0,
                                  "expanded": 1, "status": "found"})

    def test_unique_path_ordering_independent_of_goal_order(self):
        a = plan_any(3, 3, set(), (0, 0), [(2, 2), (0, 2)], max_steps=4,
                     trace=True)
        b = plan_any(3, 3, set(), (0, 0), [(0, 2), (2, 2)], max_steps=4,
                     trace=True)
        self.assertEqual(a["path"], b["path"])
        self.assertEqual(a["expanded_nodes"], b["expanded_nodes"])


class MaxStepsPlanKTest(unittest.TestCase):
    def test_ranking_unchanged_with_generous_window(self):
        unlimited = plan_k(3, 3, set(), (0, 0), (2, 2), 6)
        limited = plan_k(3, 3, set(), (0, 0), (2, 2), 6, max_steps=100)
        self.assertEqual(limited["paths"], unlimited["paths"])
        self.assertEqual(limited["costs"], unlimited["costs"])
        self.assertEqual(limited["status"], "found")
        self.assertNotIn("status", unlimited)

    def test_window_filters_long_routes(self):
        result = plan_k(3, 3, set(), (0, 0), (2, 2), 5, max_steps=4)
        self.assertEqual(result["status"], "found")
        self.assertEqual(len(result["paths"]), 5)
        for path in result["paths"]:
            self.assertLessEqual(len(path) - 1, 4)
        unique = {tuple(tuple(point) for point in path)
                  for path in result["paths"]}
        self.assertEqual(len(unique), 5)

    def test_step_exhausted(self):
        result = plan_k(3, 3, set(), (0, 0), (2, 2), 5, max_steps=3)
        self.assertEqual(result["status"], "step_exhausted")
        self.assertEqual(result["paths"], [])
        self.assertEqual(result["costs"], [])

    def test_fewer_routes_than_k_is_still_found(self):
        # A line grid offers exactly one route; the heap drains inside
        # the window with fewer than k routes found.
        result = plan_k(4, 1, set(), (0, 0), (3, 0), 5, max_steps=3)
        self.assertEqual(result["status"], "found")
        self.assertEqual(len(result["paths"]), 1)

    def test_budget_priority(self):
        result = plan_k(3, 3, set(), (0, 0), (2, 2), 5,
                        max_expanded=1, max_steps=1)
        self.assertEqual(result["status"], "budget_exhausted")

    def test_every_route_replays(self):
        result = plan_k(3, 3, set(), (0, 0), (2, 2), 5, max_steps=4)
        for path, cost in zip(result["paths"], result["costs"]):
            checked = replay(3, 3, set(), path[0], path[-1], path,
                             max_steps=4)
            self.assertEqual(
                checked,
                {"valid": True, "cost": cost, "steps": len(path) - 1})


class MaxStepsBatchAndMultiStartTest(unittest.TestCase):
    def test_batch_preserves_request_order(self):
        requests = [[(0, 0), (2, 2)], [(1, 1), (1, 1)],
                    [(2, 2), (0, 0)]]
        result = plan_batch(3, 3, set(), requests, max_steps=4, trace=True)
        self.assertEqual([item["status"] for item in result["results"]],
                         ["found", "found", "found"])
        for item, (start, goal) in zip(result["results"], requests):
            self.assertEqual(item["path"][0], tuple(start))
            self.assertEqual(item["path"][-1], tuple(goal))
            self.assertLessEqual(len(item["path"]) - 1, 4)

    def test_batch_results_match_independent_plan_calls(self):
        requests = [[(0, 0), (2, 2)], [(1, 1), (1, 1)]]
        batch = plan_batch(3, 3, set(), requests, max_steps=3, trace=True)
        first = plan(3, 3, set(), (0, 0), (2, 2), max_steps=3, trace=True)
        second = plan(3, 3, set(), (1, 1), (1, 1), max_steps=3, trace=True)
        self.assertEqual(batch["results"][0], first)
        self.assertEqual(batch["results"][1], second)

    def test_multi_start_winner_is_order_independent(self):
        a = plan_multi_start(4, 3, set(), [(0, 0), (3, 0)], (3, 2),
                             max_steps=5)
        b = plan_multi_start(4, 3, set(), [(3, 0), (0, 0)], (3, 2),
                             max_steps=5)
        self.assertEqual(a["path"], b["path"])
        self.assertEqual(a["cost"], b["cost"])
        self.assertEqual(a["status"], "found")

    def test_multi_start_too_short_window(self):
        result = plan_multi_start(5, 1, set(), [(0, 0), (2, 0)], (4, 0),
                                  max_steps=1)
        self.assertEqual(result["status"], "step_exhausted")
        self.assertIsNone(result["path"])

    def test_multi_start_zero_window_goal_start(self):
        result = plan_multi_start(3, 3, set(), [(1, 1), (0, 0)], (1, 1),
                                  max_steps=0)
        self.assertEqual(result["status"], "found")
        self.assertEqual(result["path"], [(1, 1)])


class MaxStepsReplayTest(unittest.TestCase):
    PATH = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2)]

    def test_omitted_window_keeps_behavior(self):
        checked = replay(3, 3, set(), (0, 0), (0, 2), self.PATH)
        self.assertEqual(checked, {"valid": True, "cost": 6, "steps": 6})
        explicit = replay(3, 3, set(), (0, 0), (0, 2), self.PATH,
                          max_steps=None)
        self.assertEqual(explicit, checked)

    def test_otherwise_valid_long_path_is_step_limit(self):
        plain = replay(3, 3, set(), (0, 0), (0, 2), self.PATH, max_steps=4)
        self.assertEqual(plain, {"valid": False, "cost": None,
                                 "steps": None})
        diagnosed = replay(3, 3, set(), (0, 0), (0, 2), self.PATH,
                           max_steps=4, diagnose=True)
        self.assertEqual(diagnosed["error"], "step_limit")
        self.assertEqual(diagnosed["error_index"], 5)

    def test_path_exactly_at_the_limit_is_valid(self):
        checked = replay(3, 3, set(), (0, 0), (2, 2),
                         [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)],
                         max_steps=4)
        self.assertEqual(checked, {"valid": True, "cost": 4, "steps": 4})

    def test_existing_error_priority_unchanged(self):
        cases = [
            ("start_mismatch",
             replay(3, 3, set(), (1, 0), (0, 2), self.PATH,
                    max_steps=0, diagnose=True)),
            ("goal_mismatch",
             replay(3, 3, set(), (0, 0), (2, 2),
                    self.PATH[:5] + [(1, 1)],
                    max_steps=0, diagnose=True)),
            ("non_adjacent",
             replay(3, 3, set(), (0, 0), (2, 0),
                    [(0, 0), (2, 0)], max_steps=0, diagnose=True)),
            ("repeated_coordinate",
             replay(3, 3, set(), (0, 0), (1, 0),
                    [(0, 0), (1, 0), (0, 0), (1, 0)],
                    max_steps=1, diagnose=True)),
        ]
        for expected, result in cases:
            self.assertEqual(result["error"], expected)

    def test_dynamic_blocked_precedes_step_limit(self):
        frames = [[], [(1, 0)], [], [], [], [], []]
        result = replay(3, 3, set(), (0, 0), (0, 2), self.PATH,
                        dynamic_blocked=frames, max_steps=0, diagnose=True)
        self.assertEqual(result["error"], "dynamic_blocked")
        self.assertEqual(result["error_index"], 1)

    def test_wait_path_under_and_over_window(self):
        frames = [[], [(1, 0)], []]
        under = replay(2, 1, set(), (0, 0), (1, 0),
                       [(0, 0), (0, 0), (1, 0)],
                       dynamic_blocked=frames, allow_wait=True,
                       max_steps=2, diagnose=True)
        self.assertTrue(under["valid"])
        over = replay(2, 1, set(), (0, 0), (1, 0),
                      [(0, 0), (0, 0), (1, 0)],
                      dynamic_blocked=frames, allow_wait=True,
                      max_steps=1, diagnose=True)
        self.assertEqual(over["error"], "step_limit")
        self.assertEqual(over["error_index"], 2)


class MaxStepsSnapshotTest(unittest.TestCase):
    def test_checkpoint_records_max_steps(self):
        result = plan(4, 4, set(), (0, 0), (3, 3), max_steps=6,
                      max_expanded=2, snapshot=True)
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(result["checkpoint"]["max_steps"], 6)
        self.assertIn("step_limited", result["checkpoint"]["state"])
        checkpoint = json.loads(json.dumps(result["checkpoint"]))
        self.assertEqual(checkpoint, result["checkpoint"])

    def test_checkpoint_omits_field_without_window(self):
        result = plan(4, 4, set(), (0, 0), (3, 3), max_expanded=3,
                      snapshot=True)
        self.assertNotIn("max_steps", result["checkpoint"])
        self.assertNotIn("step_limited", result["checkpoint"]["state"])

    def test_step_exhausted_carries_no_checkpoint(self):
        result = plan(3, 3, set(), (0, 0), (2, 2), max_steps=3,
                      snapshot=True)
        self.assertEqual(result["status"], "step_exhausted")
        self.assertNotIn("checkpoint", result)

    def assert_resume_matches(self, call, args, kwargs):
        budget, full = smallest_exhausting_budget(call, args, kwargs)
        first = call(*args, trace=True, max_expanded=budget,
                     snapshot=True, **kwargs)
        self.assertEqual(
            without_checkpoint(first),
            without_checkpoint(call(*args, trace=True,
                                   max_expanded=budget, **kwargs)))
        # An unbudgeted resume equals the unpaused traced call.
        self.assertEqual(without_checkpoint(resume(first["checkpoint"])),
                         without_checkpoint(full))

    def test_static_plan_resume(self):
        self.assert_resume_matches(
            plan, (4, 4, {(1, 2), (2, 1)}, (0, 0), (3, 3)),
            {"max_steps": 6})

    def test_dynamic_plan_resume(self):
        self.assert_resume_matches(
            plan, (3, 3, set(), (1, 0), (2, 2)),
            {"dynamic_blocked": [[], [(2, 1)], []], "max_steps": 5})

    def test_plan_any_resume(self):
        self.assert_resume_matches(
            plan_any, (4, 4, set(), (0, 0), [(3, 3), (0, 3)]),
            {"max_steps": 6})

    def test_multi_start_resume(self):
        self.assert_resume_matches(
            plan_multi_start, (4, 3, set(), [(0, 0), (3, 0)], (3, 2)),
            {"max_steps": 5})

    def test_chained_resumes(self):
        kwargs = {"max_steps": 10}
        first = plan(6, 6, set(), (0, 0), (5, 5), max_expanded=0,
                     snapshot=True, trace=True, **kwargs)
        second = resume(first["checkpoint"], max_expanded=4)
        self.assertEqual(second["status"], "budget_exhausted")
        self.assertEqual(second["checkpoint"]["max_steps"], 10)
        direct = plan(6, 6, set(), (0, 0), (5, 5), max_expanded=4,
                      snapshot=True, trace=True, **kwargs)
        self.assertEqual(without_checkpoint(second),
                         without_checkpoint(direct))

    def test_old_checkpoint_without_field_resumes_as_none(self):
        first = plan(3, 3, set(), (0, 0), (2, 2), max_expanded=3,
                     snapshot=True)
        self.assertNotIn("max_steps", first["checkpoint"])
        resumed = resume(first["checkpoint"])
        self.assertNotIn("status", resumed)
        self.assertEqual(resumed["path"],
                         plan(3, 3, set(), (0, 0), (2, 2))["path"])

    def test_checkpoint_max_steps_type_errors(self):
        first = plan(4, 4, set(), (0, 0), (3, 3), max_steps=6,
                     max_expanded=2, snapshot=True)
        for bad in ("5", 1.5, True, [5]):
            checkpoint = json.loads(json.dumps(first["checkpoint"]))
            checkpoint["max_steps"] = bad
            with self.assertRaises(TypeError, msg=f"max_steps={bad!r}"):
                resume(checkpoint)

    def test_checkpoint_max_steps_negative_is_value_error(self):
        first = plan(4, 4, set(), (0, 0), (3, 3), max_steps=6,
                     max_expanded=2, snapshot=True)
        checkpoint = json.loads(json.dumps(first["checkpoint"]))
        checkpoint["max_steps"] = -1
        with self.assertRaises(ValueError):
            resume(checkpoint)

    def test_checkpoint_state_inconsistent_with_window(self):
        # After six closes the pending candidates already extend beyond
        # a zero-step window, so recording ``max_steps == 0`` is
        # internally inconsistent.
        first = plan(4, 4, set(), (0, 0), (3, 3), max_steps=6,
                     max_expanded=6, snapshot=True)
        checkpoint = json.loads(json.dumps(first["checkpoint"]))
        checkpoint["max_steps"] = 0
        with self.assertRaises(ValueError):
            resume(checkpoint)

    def test_step_limited_flag_without_window_is_value_error(self):
        # A dynamic route-tree checkpoint keeps its tree shape after the
        # window field is removed, so the orphaned discard flag is the
        # detected inconsistency.
        first = plan(3, 3, set(), (1, 0), (2, 2),
                     dynamic_blocked=[[], [(2, 1)], []], max_steps=5,
                     max_expanded=2, snapshot=True)
        checkpoint = json.loads(json.dumps(first["checkpoint"]))
        del checkpoint["max_steps"]
        checkpoint["state"]["step_limited"] = True
        with self.assertRaises(ValueError):
            resume(checkpoint)

    def test_step_limited_flag_type_error(self):
        first = plan(4, 4, set(), (0, 0), (3, 3), max_steps=6,
                     max_expanded=6, snapshot=True)
        checkpoint = json.loads(json.dumps(first["checkpoint"]))
        checkpoint["state"]["step_limited"] = 1
        with self.assertRaises(TypeError):
            resume(checkpoint)


class MaxStepsVerifyTest(unittest.TestCase):
    def test_verify_trace_accepts_consistent_record(self):
        record = plan(4, 4, {(1, 2)}, (0, 0), (3, 3), max_steps=6,
                      trace=True)
        self.assertTrue(verify_trace(4, 4, {(1, 2)}, (0, 0), (3, 3),
                                     record, max_steps=6)["valid"])
        round_trip = json.loads(json.dumps(record))
        self.assertTrue(verify_trace(4, 4, {(1, 2)}, (0, 0), (3, 3),
                                     round_trip, max_steps=6)["valid"])

    def test_verify_trace_detects_window_mismatch(self):
        record = plan(4, 4, {(1, 2)}, (0, 0), (3, 3), max_steps=6,
                      trace=True)
        report = verify_trace(4, 4, {(1, 2)}, (0, 0), (3, 3), record,
                              max_steps=3)
        self.assertFalse(report["valid"])
        self.assertEqual(report["mismatch"], "path")

    def test_verify_trace_validates_window_before_record(self):
        good = {"path": None, "cost": None, "expanded": 0,
                "expanded_nodes": []}
        with self.assertRaises(TypeError):
            verify_trace(3, 3, set(), (0, 0), (2, 2), good, max_steps=True)
        with self.assertRaises(ValueError):
            verify_trace(3, 3, set(), (0, 0), (2, 2), good, max_steps=-1)

    def test_verify_any_trace(self):
        record = plan_any(4, 4, set(), (0, 0), [(3, 3), (0, 3)],
                          max_steps=6, trace=True)
        self.assertTrue(verify_any_trace(
            4, 4, set(), (0, 0), [(3, 3), (0, 3)], record,
            max_steps=6)["valid"])

    def test_verify_batch_trace(self):
        requests = [[(0, 0), (2, 2)], [(1, 1), (1, 1)]]
        record = plan_batch(3, 3, set(), requests, max_steps=4, trace=True)
        self.assertTrue(verify_batch_trace(3, 3, set(), requests, record,
                                           max_steps=4)["valid"])
        tampered = json.loads(json.dumps(record))
        tampered["results"][0]["status"] = "found"
        tampered["results"][0]["path"] = None
        report = verify_batch_trace(3, 3, set(), requests, tampered,
                                    max_steps=4)
        self.assertFalse(report["valid"])
        self.assertEqual(report["request_index"], 0)
        self.assertEqual(report["mismatch"], "path")


if __name__ == '__main__':
    unittest.main()
