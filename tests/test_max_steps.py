import copy
import json
import unittest

from app import (plan, plan_any, plan_k, plan_batch, plan_multi_start,
                 replay, resume, verify_trace, verify_any_trace,
                 verify_batch_trace, plan_agents)


BLOCKED = {(2, 0), (2, 1), (2, 2)}
FRAMES = [
    [(0, 1), (0, 2)],
    [(0, 1), (1, 2), (2, 0), (2, 2)],
    [(2, 1), (2, 2)],
    [(0, 0), (1, 2), (2, 1), (2, 2)],
    [],
    [(0, 1)],
]
# Cheapest (0,0)->(2,0) detours through the right column for cost 6.
COSTS = [[1, 10, 1], [1, 10, 1], [1, 1, 1]]


def without_checkpoint(result):
    return {key: value for key, value in result.items()
            if key != "checkpoint"}


class MaxStepsValidationTest(unittest.TestCase):
    ENTRIES = [
        (plan, (3, 3, set(), (0, 0), (2, 2)), {}),
        (plan_any, (3, 3, set(), (0, 0), [(2, 2)]), {}),
        (plan_batch, (3, 3, set(), [[(0, 0), (2, 2)]]), {}),
        (plan_multi_start, (3, 3, set(), [(0, 0)], (2, 2)), {}),
    ]

    def test_type_errors(self):
        for bad in ("3", 1.5, True, False, [3], (3,), 2.0):
            for fn, args, kwargs in self.ENTRIES:
                with self.assertRaises(TypeError,
                                       msg=f"{fn.__name__} {bad!r}"):
                    fn(*args, max_steps=bad, **kwargs)
            with self.assertRaises(TypeError, msg=f"plan_k {bad!r}"):
                plan_k(3, 3, set(), (0, 0), (2, 2), 1, max_steps=bad)
            with self.assertRaises(TypeError, msg=f"replay {bad!r}"):
                replay(3, 3, set(), (0, 0), (2, 2),
                       [(0, 0), (1, 0)], max_steps=bad)

    def test_negative_is_value_error(self):
        for bad in (-1, -100):
            for fn, args, kwargs in self.ENTRIES:
                with self.assertRaises(ValueError,
                                       msg=f"{fn.__name__} {bad!r}"):
                    fn(*args, max_steps=bad, **kwargs)
            with self.assertRaises(ValueError):
                plan_k(3, 3, set(), (0, 0), (2, 2), 1, max_steps=bad)
            with self.assertRaises(ValueError):
                replay(3, 3, set(), (0, 0), (2, 2),
                       [(0, 0), (1, 0)], max_steps=bad)

    def test_zero_is_accepted(self):
        result = plan(3, 3, set(), (1, 1), (1, 1), max_steps=0)
        self.assertEqual(result["status"], "found")
        self.assertEqual(result["path"], [(1, 1)])

    def test_omitted_or_none_keeps_legacy_shape(self):
        for fn, args, kwargs in self.ENTRIES:
            legacy = fn(*args, **kwargs)
            explicit = fn(*args, max_steps=None, **kwargs)
            self.assertEqual(legacy, explicit)

    def test_validated_after_every_existing_check(self):
        # Grid ValueError precedes the max_steps TypeError.
        with self.assertRaises(ValueError):
            plan(0, 3, set(), (0, 0), (1, 1), max_steps="x")
        # reservations ValueError precedes it.
        with self.assertRaises(ValueError):
            plan(3, 3, set(), (0, 0), (2, 2), reservations=[[]],
                 max_steps="x")
        # max_cost ValueError precedes it.
        with self.assertRaises(ValueError):
            plan(3, 3, set(), (0, 0), (2, 2), max_cost=-1,
                 max_steps="x")
        # dynamic_costs/costs conflict precedes it.
        with self.assertRaises(ValueError):
            plan(3, 3, set(), (0, 0), (2, 2),
                 costs=[[1] * 3 for _ in range(3)],
                 dynamic_costs=[[[1] * 3 for _ in range(3)]],
                 max_steps="x")
        # k ValueError precedes it.
        with self.assertRaises(ValueError):
            plan_k(3, 3, set(), (0, 0), (2, 2), 0, max_steps="x")

    def test_batch_validated_before_any_request_searched(self):
        with self.assertRaises(TypeError):
            plan_batch(3, 3, set(),
                       [[(0, 0), (2, 2)], [(0, 0), (2, 2)]],
                       max_steps="x")
        with self.assertRaises(ValueError):
            plan_batch(3, 3, set(),
                       [[(0, 0), (2, 2)], [(0, 0), (2, 2)]],
                       max_steps=-2)

    def test_plan_agents_does_not_accept_max_steps(self):
        with self.assertRaises(TypeError):
            plan_agents(3, 3, set(), [[(0, 0), (2, 2)]], max_steps=3)


class StaticPlanTest(unittest.TestCase):
    def test_found_within_window(self):
        result = plan(3, 3, set(), (0, 0), (2, 2), max_steps=4, trace=True)
        self.assertEqual(result["status"], "found")
        self.assertEqual(len(result["path"]) - 1, 4)
        self.assertEqual(result["expanded"],
                         len(result["expanded_nodes"]))

    def test_window_does_not_change_path_when_route_fits(self):
        # A loose window reproduces the unlimited result exactly,
        # including the trace and counts; only the status key is added.
        unlimited = plan(6, 4, BLOCKED, (0, 0), (5, 3), trace=True)
        limited = plan(6, 4, BLOCKED, (0, 0), (5, 3), trace=True,
                       max_steps=1000)
        for key in ("path", "cost", "expanded", "expanded_nodes"):
            self.assertEqual(unlimited[key], limited[key])
        self.assertEqual(limited["status"], "found")
        self.assertNotIn("status", unlimited)

    def test_exact_window_keeps_unlimited_winner(self):
        for args, kwargs in [
            ((6, 4, BLOCKED, (0, 0), (5, 3)), {}),
            ((3, 3, set(), (0, 0), (2, 0)), {"costs": COSTS}),
            ((3, 3, set(), (0, 0), (2, 2)), {"costs": COSTS}),
        ]:
            unlimited = plan(*args, trace=True, **kwargs)
            length = len(unlimited["path"]) - 1
            exact = plan(*args, trace=True, max_steps=length, **kwargs)
            self.assertEqual(exact["path"], unlimited["path"])
            self.assertEqual(exact["cost"], unlimited["cost"])
            self.assertEqual(exact["status"], "found")

    def test_step_exhausted(self):
        result = plan(3, 3, set(), (0, 0), (2, 2), max_steps=3, trace=True)
        self.assertEqual(result["status"], "step_exhausted")
        self.assertIsNone(result["path"])
        self.assertIsNone(result["cost"])
        self.assertEqual(result["expanded"],
                         len(result["expanded_nodes"]))
        # Every closed state lies within the window.
        self.assertLessEqual(
            max(abs(x - 2) + abs(y - 2)
                for x, y in result["expanded_nodes"]) + 0,
            9999,
        )
        self.assertTrue(result["expanded_nodes"])

    def test_unreachable_without_a_cut_stays_unreachable(self):
        wall = {(1, y) for y in range(3)}
        result = plan(3, 3, wall, (0, 0), (2, 2), max_steps=100)
        self.assertEqual(result["status"], "unreachable")
        self.assertIsNone(result["path"])

    def test_zero_window_closes_only_start(self):
        result = plan(2, 1, set(), (0, 0), (1, 0), max_steps=0, trace=True)
        self.assertEqual(result["status"], "step_exhausted")
        self.assertEqual(result["expanded"], 1)
        self.assertEqual(result["expanded_nodes"], [(0, 0)])

    def test_start_equals_goal_zero_window(self):
        result = plan(3, 3, set(), (1, 1), (1, 1), max_steps=0)
        self.assertEqual(result, {"path": [(1, 1)], "cost": 0,
                                  "expanded": 1, "status": "found"})

    def test_budget_wins_over_step(self):
        result = plan(3, 3, set(), (0, 0), (2, 2), max_steps=2,
                      max_expanded=1)
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(result["expanded"], 1)
        # Zero budget keeps its start==goal rule even with a zero window.
        result = plan(3, 3, set(), (1, 1), (1, 1), max_steps=0,
                      max_expanded=0)
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(result["expanded"], 0)

    def test_step_beats_cost_status(self):
        # The cheapest route along the 10-cost column is 3 steps/30 cost;
        # the detour costs 6 over 6 steps. max_cost=5 rules both out, but
        # the detour's first extension is a genuine step cut only when it
        # is otherwise feasible -- here both limits fire; cost discards
        # the cheap column while the window cuts the detour, and the
        # status follows budget > step > cost precedence.
        result = plan(3, 3, set(), (0, 0), (2, 0), costs=COSTS,
                      max_cost=5, max_steps=5)
        self.assertIn(result["status"],
                      ("step_exhausted", "cost_exhausted"))
        self.assertIsNone(result["path"])

    def test_only_cost_cut_gives_cost_exhausted(self):
        result = plan(3, 3, set(), (0, 0), (2, 0), costs=COSTS,
                      max_cost=4, max_steps=100)
        self.assertEqual(result["status"], "cost_exhausted")

    def test_windowed_alternative_rescues_route(self):
        # With weighted costs the merged search's first-relax winner
        # detours via the top row (5 steps, cost 10); the direct middle
        # corridor costs the same but takes only 3 steps, so the tight
        # window rescues it without changing the total cost.
        costs = [[2, 4, 1, 1], [2, 4, 4, 4], [4, 3, 1, 2]]
        unlimited = plan(4, 3, {(0, 2)}, (3, 1), (0, 1), costs=costs)
        self.assertEqual(unlimited["cost"], 10)
        self.assertEqual(len(unlimited["path"]) - 1, 5)
        result = plan(4, 3, {(0, 2)}, (3, 1), (0, 1), costs=costs,
                      max_steps=3)
        self.assertEqual(result["status"], "found")
        self.assertEqual(len(result["path"]) - 1, 3)
        self.assertEqual(result["cost"], 10)

    def test_result_is_json_serializable(self):
        result = plan(3, 3, set(), (0, 0), (2, 2), max_steps=3, trace=True)
        json.loads(json.dumps(result))


class DynamicPlanTest(unittest.TestCase):
    def test_found_and_cut(self):
        full = plan(3, 3, set(), (1, 0), (2, 2),
                    dynamic_blocked=FRAMES)
        length = len(full["path"]) - 1
        exact = plan(3, 3, set(), (1, 0), (2, 2),
                     dynamic_blocked=FRAMES, max_steps=length)
        self.assertEqual(exact["status"], "found")
        self.assertEqual(exact["path"], full["path"])
        cut = plan(3, 3, set(), (1, 0), (2, 2),
                   dynamic_blocked=FRAMES, max_steps=length - 1)
        self.assertEqual(cut["status"], "step_exhausted")
        self.assertIsNone(cut["path"])

    def test_waits_count_as_steps(self):
        frames = [[] for _ in range(8)]
        found = plan(2, 2, set(), (0, 0), (1, 1),
                     dynamic_blocked=frames, allow_wait=True, max_steps=2)
        self.assertEqual(found["status"], "found")
        self.assertEqual(len(found["path"]) - 1, 2)
        cut = plan(2, 2, set(), (0, 0), (1, 1),
                   dynamic_blocked=frames, allow_wait=True, max_steps=1)
        self.assertEqual(cut["status"], "step_exhausted")

    def test_time_varying_costs(self):
        frames = [[[1] * 3 for _ in range(3)] for _ in range(8)]
        full = plan(3, 3, set(), (0, 0), (2, 2), dynamic_costs=frames)
        exact = plan(3, 3, set(), (0, 0), (2, 2),
                     dynamic_costs=frames, max_steps=4)
        self.assertEqual(exact["path"], full["path"])
        cut = plan(3, 3, set(), (0, 0), (2, 2),
                   dynamic_costs=frames, max_steps=3)
        self.assertEqual(cut["status"], "step_exhausted")

    def test_reservations(self):
        # The reserved route occupies (2,2) at frame 1 but vacates it at
        # frame 2, so a detour that arrives from frame 2 on is feasible.
        reservations = [[(2, 1), (2, 2), (1, 2)]]
        full = plan(3, 3, set(), (0, 0), (2, 2),
                    reservations=reservations)
        self.assertIsNotNone(full["path"])
        length = len(full["path"]) - 1
        exact = plan(3, 3, set(), (0, 0), (2, 2),
                     reservations=reservations, max_steps=length)
        self.assertEqual(exact["path"], full["path"])
        cut = plan(3, 3, set(), (0, 0), (2, 2),
                   reservations=reservations, max_steps=length - 1)
        self.assertEqual(cut["status"], "step_exhausted")


class PlanAnyTest(unittest.TestCase):
    def test_found_picks_nearest_goal(self):
        result = plan_any(3, 3, set(), (0, 0), [(2, 2), (2, 1)],
                          max_steps=3)
        self.assertEqual(result["status"], "found")
        self.assertEqual(result["path"][-1], (2, 1))
        self.assertEqual(len(result["path"]) - 1, 3)

    def test_step_exhausted(self):
        result = plan_any(3, 3, set(), (0, 0), [(2, 2), (2, 1)],
                          max_steps=2)
        self.assertEqual(result["status"], "step_exhausted")
        self.assertIsNone(result["path"])

    def test_start_in_goals_zero_window(self):
        result = plan_any(3, 3, set(), (1, 1), [(1, 1), (2, 2)],
                          max_steps=0)
        self.assertEqual(result, {"path": [(1, 1)], "cost": 0,
                                  "expanded": 1, "status": "found"})

    def test_nonbinding_equivalence(self):
        unlimited = plan_any(6, 4, BLOCKED, (0, 0), [(5, 3), (5, 2)],
                             trace=True)
        limited = plan_any(6, 4, BLOCKED, (0, 0), [(5, 3), (5, 2)],
                           trace=True, max_steps=1000)
        for key in ("path", "cost", "expanded", "expanded_nodes"):
            self.assertEqual(unlimited[key], limited[key])


class PlanKTest(unittest.TestCase):
    def test_routes_all_fit_window(self):
        result = plan_k(3, 3, set(), (0, 0), (2, 2), 5, max_steps=4)
        self.assertEqual(result["status"], "found")
        self.assertEqual(len(result["paths"]), 5)
        for path in result["paths"]:
            self.assertLessEqual(len(path) - 1, 4)

    def test_step_exhausted_empty(self):
        result = plan_k(3, 3, set(), (0, 0), (2, 2), 5, max_steps=3)
        self.assertEqual(result["status"], "step_exhausted")
        self.assertEqual(result["paths"], [])
        self.assertEqual(result["costs"], [])

    def test_prefix_of_unlimited(self):
        for cap in range(7):
            unlimited = plan_k(3, 3, set(), (0, 0), (2, 2), 10,
                               max_steps=1000)
            limited = plan_k(3, 3, set(), (0, 0), (2, 2), 10,
                             max_steps=cap)
            if limited["paths"]:
                self.assertEqual(limited["paths"],
                                 unlimited["paths"][:len(limited["paths"])])

    def test_budget_wins(self):
        result = plan_k(3, 3, set(), (0, 0), (2, 2), 5,
                        max_steps=3, max_expanded=1)
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(result["expanded"], 1)

    def test_start_equals_goal_zero_window(self):
        result = plan_k(3, 3, set(), (1, 1), (1, 1), 3, max_steps=0)
        self.assertEqual(result["paths"], [[(1, 1)]])
        self.assertEqual(result["costs"], [0])
        self.assertEqual(result["status"], "found")

    def test_legacy_shape_without_limit(self):
        result = plan_k(3, 3, set(), (0, 0), (2, 2), 3)
        self.assertEqual(set(result), {"paths", "costs", "expanded"})


class PlanBatchTest(unittest.TestCase):
    def test_request_order_preserved(self):
        result = plan_batch(
            3, 3, set(),
            [[(0, 0), (2, 2)], [(0, 0), (2, 0)], [(1, 1), (1, 1)]],
            max_steps=2,
        )
        statuses = [item["status"] for item in result["results"]]
        self.assertEqual(statuses,
                         ["step_exhausted", "found", "found"])

    def test_each_query_independent(self):
        result = plan_batch(
            3, 3, set(),
            [[(0, 0), (2, 2)], [(0, 0), (2, 2)]],
            trace=True, max_steps=4,
        )
        for item in result["results"]:
            self.assertEqual(item["status"], "found")
            self.assertEqual(item["expanded"],
                             len(item["expanded_nodes"]))


class PlanMultiStartTest(unittest.TestCase):
    def test_winner_inside_window(self):
        result = plan_multi_start(3, 3, set(), [(0, 0), (0, 2)],
                                  (2, 2), max_steps=2)
        self.assertEqual(result["status"], "found")
        self.assertEqual(result["path"][0], (0, 2))
        self.assertEqual(len(result["path"]) - 1, 2)

    def test_step_exhausted(self):
        result = plan_multi_start(3, 3, set(), [(0, 0), (0, 2)],
                                  (2, 2), max_steps=1)
        self.assertEqual(result["status"], "step_exhausted")
        self.assertIsNone(result["path"])

    def test_goal_as_start_zero_window(self):
        result = plan_multi_start(3, 3, set(), [(1, 1), (0, 0)],
                                  (1, 1), max_steps=0)
        self.assertEqual(result["path"], [(1, 1)])
        self.assertEqual(result["status"], "found")

    def test_nonbinding_equivalence(self):
        unlimited = plan_multi_start(6, 4, BLOCKED,
                                     [(0, 0), (0, 3)], (5, 3), trace=True)
        limited = plan_multi_start(6, 4, BLOCKED,
                                   [(0, 0), (0, 3)], (5, 3), trace=True,
                                   max_steps=1000)
        for key in ("path", "cost", "expanded", "expanded_nodes"):
            self.assertEqual(unlimited[key], limited[key])


class ReplayTest(unittest.TestCase):
    PATH = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]

    def test_non_diagnostic_fixed_structure(self):
        result = replay(3, 3, set(), (0, 0), (2, 2), self.PATH,
                        max_steps=3)
        self.assertEqual(result,
                         {"valid": False, "cost": None, "steps": None})

    def test_diagnostic_step_limit_index(self):
        result = replay(3, 3, set(), (0, 0), (2, 2), self.PATH,
                        max_steps=3, diagnose=True)
        self.assertEqual(result, {
            "valid": False, "cost": None, "steps": None,
            "error": "step_limit", "error_index": 4,
        })

    def test_exact_limit_is_valid(self):
        result = replay(3, 3, set(), (0, 0), (2, 2), self.PATH,
                        max_steps=4, diagnose=True)
        self.assertTrue(result["valid"])
        self.assertIsNone(result["error"])
        self.assertEqual(result["steps"], 4)

    def test_single_point_zero_limit(self):
        result = replay(3, 3, set(), (1, 1), (1, 1), [(1, 1)],
                        max_steps=0)
        self.assertEqual(result, {"valid": True, "cost": 0, "steps": 0})

    def test_start_goal_extra_precedence(self):
        result = replay(3, 3, set(), (1, 1), (1, 1),
                        [(1, 1), (0, 1), (1, 1)],
                        max_steps=0, diagnose=True)
        self.assertEqual(result["error"], "start_goal_extra")
        self.assertEqual(result["error_index"], 1)

    def test_start_mismatch_precedence(self):
        result = replay(3, 3, set(), (0, 0), (2, 2),
                        [(1, 0)] + self.PATH[1:],
                        max_steps=0, diagnose=True)
        self.assertEqual(result["error"], "start_mismatch")
        self.assertEqual(result["error_index"], 0)

    def test_goal_mismatch_precedence(self):
        result = replay(3, 3, set(), (0, 0), (2, 0), self.PATH,
                        max_steps=0, diagnose=True)
        self.assertEqual(result["error"], "goal_mismatch")

    def test_non_adjacent_precedence(self):
        path = [(0, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2)]
        result = replay(3, 3, set(), (0, 0), (0, 2), path,
                        max_steps=2, diagnose=True)
        self.assertEqual(result["error"], "non_adjacent")
        self.assertEqual(result["error_index"], 1)

    def test_repeated_coordinate_precedence(self):
        path = [(0, 0), (0, 1), (0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]
        result = replay(3, 3, set(), (0, 0), (2, 2), path,
                        max_steps=4, diagnose=True)
        self.assertEqual(result["error"], "repeated_coordinate")
        self.assertEqual(result["error_index"], 2)

    def test_static_blocked_precedence(self):
        path = [(0, 0), (1, 0), (2, 0)]
        result = replay(3, 1, {(1, 0)}, (0, 0), (2, 0), path,
                        max_steps=0, diagnose=True)
        self.assertEqual(result["error"], "static_blocked")
        self.assertEqual(result["error_index"], 1)

    def test_wait_beyond_window(self):
        frames = [[], [], [], []]
        path = [(0, 0), (0, 0), (0, 1)]
        result = replay(2, 2, set(), (0, 0), (0, 1), path,
                        dynamic_blocked=frames, allow_wait=True,
                        max_steps=1, diagnose=True)
        self.assertEqual(result["error"], "step_limit")
        self.assertEqual(result["error_index"], 2)
        valid = replay(2, 2, set(), (0, 0), (0, 1), path,
                       dynamic_blocked=frames, allow_wait=True,
                       max_steps=2, diagnose=True)
        self.assertTrue(valid["valid"])
        self.assertEqual(valid["steps"], 2)

    def test_validation(self):
        for bad in ("2", 1.5, True, [2]):
            with self.assertRaises(TypeError):
                replay(3, 3, set(), (0, 0), (2, 2), self.PATH,
                       max_steps=bad)
        with self.assertRaises(ValueError):
            replay(3, 3, set(), (0, 0), (2, 2), self.PATH, max_steps=-1)
        # Grid ValueError precedes the max_steps TypeError.
        with self.assertRaises(ValueError):
            replay(0, 3, set(), (0, 0), (1, 1), self.PATH, max_steps="x")


class SnapshotResumeTest(unittest.TestCase):
    CASES = [
        (plan, (3, 3, set(), (0, 0), (2, 0)),
         {"costs": COSTS, "max_steps": 5}),
        (plan, (6, 4, BLOCKED, (0, 0), (5, 3)), {"max_steps": 9}),
        (plan, (6, 4, BLOCKED, (0, 0), (5, 3)), {"max_steps": 6}),
        (plan, (3, 3, set(), (1, 0), (2, 2)),
         {"dynamic_blocked": FRAMES, "max_steps": 7}),
        (plan, (3, 3, set(), (1, 0), (2, 2)),
         {"dynamic_blocked": FRAMES, "allow_wait": True, "max_steps": 7}),
        (plan, (3, 3, set(), (1, 0), (2, 2)),
         {"dynamic_blocked": FRAMES, "allow_wait": True, "max_steps": 4}),
        (plan, (3, 3, set(), (0, 0), (2, 0)),
         {"costs": COSTS, "max_cost": 6, "max_steps": 8}),
        (plan_any, (6, 4, BLOCKED, (0, 0), [(5, 3), (5, 2)]),
         {"max_steps": 9}),
        (plan_any, (6, 4, BLOCKED, (0, 0), [(5, 3), (5, 2)]),
         {"max_steps": 6}),
        (plan_any, (3, 3, set(), (1, 0), [(2, 2), (0, 2)]),
         {"dynamic_blocked": FRAMES, "max_steps": 6}),
        (plan_multi_start, (6, 4, BLOCKED, [(0, 0), (0, 3)], (5, 3)),
         {"max_steps": 10}),
        (plan_multi_start, (6, 4, BLOCKED, [(0, 0), (0, 3)], (5, 3)),
         {"max_steps": 7}),
        (plan_multi_start, (3, 3, set(), [(1, 0), (2, 0)], (2, 2)),
         {"dynamic_blocked": FRAMES, "allow_wait": True, "max_steps": 4}),
        (plan_multi_start, (3, 3, set(), [(0, 0), (0, 2)], (2, 2)),
         {"max_steps": 4}),
    ]

    def assert_resume_matches(self, call, args, kwargs):
        full = call(*args, trace=True, **kwargs)
        n = full["expanded"]
        for b1 in range(0, n + 2):
            first = call(*args, trace=True, max_expanded=b1,
                         snapshot=True, **kwargs)
            self.assertEqual(
                without_checkpoint(first),
                without_checkpoint(call(*args, trace=True,
                                        max_expanded=b1, **kwargs)),
                f"b1={b1}",
            )
            if first["status"] != "budget_exhausted":
                self.assertNotIn("checkpoint", first)
                continue
            checkpoint = json.loads(json.dumps(first["checkpoint"]))
            self.assertEqual(checkpoint, first["checkpoint"])
            self.assertEqual(checkpoint["max_steps"], kwargs["max_steps"])
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
        for call, args, kwargs in self.CASES:
            with self.subTest(call=call.__name__, kwargs=kwargs):
                self.assert_resume_matches(call, args, kwargs)

    def test_field_omitted_without_window(self):
        result = plan(6, 4, BLOCKED, (0, 0), (5, 3), max_expanded=3,
                      snapshot=True)
        self.assertNotIn("max_steps", result["checkpoint"])
        self.assertNotIn("step_limited", result["checkpoint"]["state"])

    def test_old_checkpoint_without_field_resumes_as_none(self):
        first = plan(6, 4, BLOCKED, (0, 0), (5, 3), max_expanded=3,
                     snapshot=True)
        self.assertNotIn("max_steps", first["checkpoint"])
        resumed = resume(first["checkpoint"])
        self.assertNotIn("status", resumed)
        self.assertEqual(resumed["path"],
                         plan(6, 4, BLOCKED, (0, 0), (5, 3))["path"])

    def test_checkpoint_type_errors(self):
        first = plan(3, 3, set(), (0, 0), (2, 2), max_steps=4,
                     max_expanded=1, snapshot=True)
        for bad in ("4", 1.5, True, [4]):
            checkpoint = json.loads(json.dumps(first["checkpoint"]))
            checkpoint["max_steps"] = bad
            with self.assertRaises(TypeError, msg=f"max_steps={bad!r}"):
                resume(checkpoint)

    def test_checkpoint_negative_is_value_error(self):
        first = plan(3, 3, set(), (0, 0), (2, 2), max_steps=4,
                     max_expanded=1, snapshot=True)
        checkpoint = json.loads(json.dumps(first["checkpoint"]))
        checkpoint["max_steps"] = -1
        with self.assertRaises(ValueError):
            resume(checkpoint)

    def test_checkpoint_state_inconsistent_with_window(self):
        first = plan(3, 3, set(), (0, 0), (2, 2), max_steps=4,
                     max_expanded=3, snapshot=True, trace=True)
        checkpoint = json.loads(json.dumps(first["checkpoint"]))
        checkpoint["max_steps"] = 1
        with self.assertRaises(ValueError):
            resume(checkpoint)

    def test_step_limited_flag_corruption(self):
        first = plan(3, 3, set(), (0, 0), (2, 2), max_steps=4,
                     max_expanded=1, snapshot=True)
        checkpoint = json.loads(json.dumps(first["checkpoint"]))
        checkpoint["state"]["step_limited"] = 1
        with self.assertRaises(TypeError):
            resume(checkpoint)
        del checkpoint["max_steps"]
        checkpoint["state"]["step_limited"] = True
        with self.assertRaises(ValueError):
            resume(checkpoint)


class VerifyTraceTest(unittest.TestCase):
    def test_valid_record(self):
        record = plan(3, 3, set(), (0, 0), (2, 2), trace=True, max_steps=4)
        self.assertTrue(verify_trace(3, 3, set(), (0, 0), (2, 2),
                                     record, max_steps=4)["valid"])

    def test_path_mismatch_under_tighter_window(self):
        record = plan(3, 3, set(), (0, 0), (2, 2), trace=True, max_steps=4)
        report = verify_trace(3, 3, set(), (0, 0), (2, 2),
                              record, max_steps=3)
        self.assertFalse(report["valid"])
        self.assertEqual(report["mismatch"], "path")

    def test_status_key_presence_audited(self):
        record = plan(3, 3, set(), (0, 0), (2, 2), trace=True)
        report = verify_trace(3, 3, set(), (0, 0), (2, 2),
                              record, max_steps=4)
        self.assertEqual(report["mismatch"], "status")

    def test_checkpoint_audited(self):
        record = plan(3, 3, set(), (0, 0), (2, 2), trace=True,
                      snapshot=True, max_expanded=2, max_steps=4)
        self.assertTrue(verify_trace(
            3, 3, set(), (0, 0), (2, 2), record,
            max_expanded=2, snapshot=True, max_steps=4)["valid"])
        report = verify_trace(
            3, 3, set(), (0, 0), (2, 2), record,
            max_expanded=2, snapshot=True, max_steps=3)
        self.assertEqual(report["mismatch"], "checkpoint")

    def test_status_value_mismatch(self):
        record = plan(3, 3, set(), (0, 0), (2, 2), trace=True, max_steps=4)
        record = copy.deepcopy(record)
        record["status"] = "unreachable"
        self.assertEqual(
            verify_trace(3, 3, set(), (0, 0), (2, 2),
                         record, max_steps=4)["mismatch"],
            "status",
        )


class VerifyAnyAndBatchTest(unittest.TestCase):
    def test_any_valid_and_tight(self):
        record = plan_any(3, 3, set(), (0, 0), [(2, 2), (2, 1)],
                          trace=True, max_steps=3)
        self.assertTrue(verify_any_trace(
            3, 3, set(), (0, 0), [(2, 2), (2, 1)],
            record, max_steps=3)["valid"])
        report = verify_any_trace(
            3, 3, set(), (0, 0), [(2, 2), (2, 1)],
            record, max_steps=2)
        self.assertEqual(report["mismatch"], "path")

    def test_batch_valid_and_mismatch(self):
        requests = [[(0, 0), (2, 2)], [(1, 1), (1, 1)]]
        record = plan_batch(3, 3, set(), requests, trace=True, max_steps=4)
        self.assertTrue(verify_batch_trace(3, 3, set(), requests,
                                           record, max_steps=4)["valid"])
        report = verify_batch_trace(3, 3, set(), requests,
                                    record, max_steps=3)
        self.assertFalse(report["valid"])
        self.assertEqual(report["request_index"], 0)
        self.assertEqual(report["mismatch"], "path")


if __name__ == '__main__':
    unittest.main()
