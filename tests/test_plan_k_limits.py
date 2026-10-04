import unittest

from app import plan_k, replay


FRAMES = [
    [(0, 1), (0, 2)],
    [(0, 1), (1, 2), (2, 0), (2, 2)],
    [(2, 1), (2, 2)],
    [(0, 0), (1, 2), (2, 1), (2, 2)],
    [],
    [(0, 1)],
]
# Cheapest route (0, 0) -> (2, 0) costs 6 along the cheap border; the
# direct route through the expensive middle column costs 11.
COSTS = [[1, 10, 1], [1, 10, 1], [1, 1, 1]]


class PlanKLimitValidationTest(unittest.TestCase):
    def test_max_expanded_type_errors(self):
        for bad in ("5", 1.5, True, False, [5], (5,), 2.0):
            with self.assertRaises(TypeError, msg=f"max_expanded={bad!r}"):
                plan_k(3, 3, set(), (0, 0), (2, 2), 1, max_expanded=bad)

    def test_max_expanded_negative_is_value_error(self):
        for bad in (-1, -100):
            with self.assertRaises(ValueError, msg=f"max_expanded={bad!r}"):
                plan_k(3, 3, set(), (0, 0), (2, 2), 1, max_expanded=bad)

    def test_max_cost_type_errors(self):
        for bad in ("5", 1.5, True, False, [5], (5,), 2.0):
            with self.assertRaises(TypeError, msg=f"max_cost={bad!r}"):
                plan_k(3, 3, set(), (0, 0), (2, 2), 1, max_cost=bad)

    def test_max_cost_negative_is_value_error(self):
        for bad in (-1, -100):
            with self.assertRaises(ValueError, msg=f"max_cost={bad!r}"):
                plan_k(3, 3, set(), (0, 0), (2, 2), 1, max_cost=bad)

    def test_zero_is_accepted(self):
        result = plan_k(3, 3, set(), (1, 1), (1, 1), 3,
                        max_expanded=0, max_cost=0)
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(result["expanded"], 0)
        self.assertEqual(result["paths"], [])

    def test_limits_validated_after_shared_checks_and_k(self):
        # A width ValueError precedes the limit TypeErrors.
        with self.assertRaises(ValueError):
            plan_k(0, 3, set(), (0, 0), (2, 2), 1, max_expanded="x")
        # The costs-matrix ValueError precedes the limit TypeError.
        with self.assertRaises(ValueError):
            plan_k(3, 3, set(), (0, 0), (2, 2), 1, costs=[[1]],
                   max_cost="x")
        # The dynamic frame-0 start check precedes the limit TypeError.
        with self.assertRaises(ValueError):
            plan_k(3, 3, set(), (0, 0), (2, 2), 1,
                   dynamic_blocked=[[(0, 0)]], max_expanded="x")
        # A bad k still precedes the limit checks.
        with self.assertRaises(TypeError):
            plan_k(3, 3, set(), (0, 0), (2, 2), 1.5, max_expanded="x")
        with self.assertRaises(ValueError):
            plan_k(3, 3, set(), (0, 0), (2, 2), 0, max_expanded=-1)
        with self.assertRaises(TypeError):
            plan_k(3, 3, set(), (0, 0), (2, 2), True, max_cost=-1)
        # max_expanded is validated before max_cost.
        with self.assertRaises(ValueError):
            plan_k(3, 3, set(), (0, 0), (2, 2), 1,
                   max_expanded=-1, max_cost="x")
        with self.assertRaises(TypeError):
            plan_k(3, 3, set(), (0, 0), (2, 2), 1,
                   max_expanded="x", max_cost=-1)

    def test_omitted_or_none_keeps_legacy_shape(self):
        legacy = plan_k(3, 3, set(), (0, 0), (2, 2), 6)
        for kwargs in ({}, {"max_expanded": None}, {"max_cost": None},
                       {"max_expanded": None, "max_cost": None}):
            result = plan_k(3, 3, set(), (0, 0), (2, 2), 6, **kwargs)
            self.assertEqual(result, legacy, kwargs)
            self.assertEqual(set(result), {"paths", "costs", "expanded"})

    def test_still_rejects_trace_and_snapshot(self):
        for kwargs in ({"trace": True}, {"snapshot": True}):
            with self.assertRaises(TypeError, msg=kwargs):
                plan_k(3, 3, set(), (0, 0), (2, 2), 1, **kwargs)


class PlanKBudgetTest(unittest.TestCase):
    def test_generous_budget_matches_unlimited(self):
        full = plan_k(3, 3, set(), (0, 0), (2, 2), 6)
        limited = plan_k(3, 3, set(), (0, 0), (2, 2), 6, max_expanded=1000)
        self.assertEqual(limited["paths"], full["paths"])
        self.assertEqual(limited["costs"], full["costs"])
        self.assertEqual(limited["expanded"], full["expanded"])
        self.assertEqual(limited["status"], "found")

    def test_found_when_k_routes_closed(self):
        result = plan_k(3, 3, set(), (0, 0), (2, 2), 6, max_expanded=1000)
        self.assertEqual(result["status"], "found")
        self.assertEqual(len(result["paths"]), 6)

    def test_found_when_search_exhausts_with_fewer_routes(self):
        # The corridor admits one route even though k asks for many.
        result = plan_k(5, 1, set(), (0, 0), (4, 0), 100, max_expanded=1000)
        self.assertEqual(result["status"], "found")
        self.assertEqual(len(result["paths"]), 1)

    def test_budget_truncates_and_keeps_ranking_prefix(self):
        full = plan_k(3, 3, set(), (0, 0), (2, 2), 6)
        for budget in range(0, full["expanded"]):
            result = plan_k(3, 3, set(), (0, 0), (2, 2), 6,
                            max_expanded=budget)
            self.assertEqual(result["expanded"], budget, f"budget={budget}")
            self.assertEqual(result["paths"],
                             full["paths"][:len(result["paths"])],
                             f"budget={budget}")
            self.assertEqual(result["costs"],
                             full["costs"][:len(result["costs"])],
                             f"budget={budget}")
            self.assertLess(len(result["paths"]), 6)
            self.assertEqual(result["status"], "budget_exhausted",
                             f"budget={budget}")

    def test_budget_zero_is_budget_exhausted(self):
        result = plan_k(3, 3, set(), (0, 0), (2, 2), 6, max_expanded=0)
        self.assertEqual(result, {"paths": [], "costs": [], "expanded": 0,
                                  "status": "budget_exhausted"})

    def test_budget_zero_start_equals_goal(self):
        result = plan_k(3, 3, set(), (1, 1), (1, 1), 3, max_expanded=0)
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(result["paths"], [])
        self.assertEqual(result["expanded"], 0)

    def test_budget_one_keeps_single_point_route(self):
        result = plan_k(3, 3, set(), (1, 1), (1, 1), 3, max_expanded=1)
        self.assertEqual(result, {"paths": [[(1, 1)]], "costs": [0],
                                  "expanded": 1, "status": "found"})

    def test_budget_prefix_replays(self):
        full = plan_k(4, 3, {(1, 1)}, (0, 0), (3, 2), 20)
        result = plan_k(4, 3, {(1, 1)}, (0, 0), (3, 2), 20,
                        max_expanded=full["expanded"] // 2)
        self.assertEqual(result["status"], "budget_exhausted")
        for path, cost in zip(result["paths"], result["costs"]):
            checked = replay(4, 3, {(1, 1)}, (0, 0), (3, 2), path)
            self.assertTrue(checked["valid"])
            self.assertEqual(checked["cost"], cost)


class PlanKCostLimitTest(unittest.TestCase):
    def test_generous_limit_matches_unlimited(self):
        full = plan_k(3, 3, set(), (0, 0), (2, 2), 6)
        limited = plan_k(3, 3, set(), (0, 0), (2, 2), 6, max_cost=1000)
        self.assertEqual(limited["paths"], full["paths"])
        self.assertEqual(limited["costs"], full["costs"])
        self.assertEqual(limited["expanded"], full["expanded"])
        self.assertEqual(limited["status"], "found")

    def test_equality_still_enters(self):
        # Every shortest route costs 4; the boundary is inclusive.
        result = plan_k(3, 3, set(), (0, 0), (2, 2), 6, max_cost=4)
        self.assertEqual(result["status"], "found")
        self.assertEqual(result["costs"], [4] * 6)
        self.assertEqual(len(result["paths"]), 6)

    def test_limit_keeps_exact_tie_break_order(self):
        full = plan_k(3, 3, set(), (0, 0), (2, 2), 6)
        limited = plan_k(3, 3, set(), (0, 0), (2, 2), 6, max_cost=4)
        self.assertEqual(limited["paths"], full["paths"])

    def test_cost_exhausted_with_no_route_within_limit(self):
        result = plan_k(3, 3, set(), (0, 0), (2, 2), 6, max_cost=3)
        self.assertEqual(result["status"], "cost_exhausted")
        self.assertEqual(result["paths"], [])
        self.assertEqual(result["costs"], [])
        self.assertGreaterEqual(result["expanded"], 1)

    def test_natural_exhaustion_with_routes_found_is_found(self):
        # The cheap border route costs 6 and is the only route within the
        # limit: every extension through an expensive cell is discarded,
        # the heap drains naturally and, with one route in hand, the
        # status is ``found`` even though k asked for two.
        result = plan_k(3, 3, set(), (0, 0), (2, 0), 2, costs=COSTS,
                        max_cost=6)
        self.assertEqual(result["status"], "found")
        self.assertEqual(len(result["paths"]), 1)
        self.assertEqual(result["costs"], [6])
        checked = replay(3, 3, set(), (0, 0), (2, 0), result["paths"][0],
                         costs=COSTS)
        self.assertTrue(checked["valid"])
        self.assertEqual(checked["cost"], 6)

    def test_budget_with_routes_found_still_budget_exhausted(self):
        # Picking the budget between the first and last goal closings
        # returns a ranked prefix with the budget status and live
        # candidates still pending.
        full = plan_k(3, 3, set(), (0, 0), (2, 2), 6, max_cost=1000)
        first_goal_expanded = plan_k(
            3, 3, set(), (0, 0), (2, 2), 1, max_cost=1000)["expanded"]
        budget = full["expanded"] - 1
        self.assertGreaterEqual(budget, first_goal_expanded)
        result = plan_k(3, 3, set(), (0, 0), (2, 2), 6,
                        max_expanded=budget, max_cost=1000)
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertTrue(result["paths"])
        self.assertLess(len(result["paths"]), 6)
        self.assertEqual(result["paths"],
                         full["paths"][:len(result["paths"])])
        for path, cost in zip(result["paths"], result["costs"]):
            checked = replay(3, 3, set(), (0, 0), (2, 2), path)
            self.assertTrue(checked["valid"])
            self.assertEqual(checked["cost"], cost)

    def test_zero_limit_discards_every_step(self):
        result = plan_k(2, 1, set(), (0, 0), (1, 0), 4, max_cost=0)
        self.assertEqual(result["status"], "cost_exhausted")
        self.assertEqual(result["paths"], [])
        self.assertEqual(result["expanded"], 1)

    def test_start_equals_goal_zero_limit_is_found(self):
        result = plan_k(3, 3, set(), (1, 1), (1, 1), 3, max_cost=0)
        self.assertEqual(result, {"paths": [[(1, 1)]], "costs": [0],
                                  "expanded": 1, "status": "found"})

    def test_unreachable_without_discards_stays_unreachable(self):
        wall = {(1, y) for y in range(3)}
        result = plan_k(3, 3, wall, (0, 0), (2, 2), 4, max_cost=1000)
        self.assertEqual(result["status"], "unreachable")
        self.assertEqual(result["paths"], [])
        self.assertEqual(result["expanded"], 3)

    def test_unreachable_status_deterministic(self):
        first = plan_k(3, 3, set(), (0, 0), (2, 2), 6, max_cost=3)
        second = plan_k(3, 3, set(), (0, 0), (2, 2), 6, max_cost=3)
        self.assertEqual(first, second)


class PlanKCombinedLimitTest(unittest.TestCase):
    def test_budget_exhausted_takes_precedence(self):
        result = plan_k(6, 4,
                        {(2, 0), (2, 1), (2, 2)}, (0, 0), (5, 3), 8,
                        max_expanded=3, max_cost=2)
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(result["expanded"], 3)

    def test_budget_cannot_replace_determined_routes(self):
        full = plan_k(4, 3, {(1, 1)}, (0, 0), (3, 2), 20)
        for budget in range(1, full["expanded"] + 1):
            result = plan_k(4, 3, {(1, 1)}, (0, 0), (3, 2), 20,
                            max_expanded=budget, max_cost=1000)
            self.assertEqual(
                result["paths"],
                full["paths"][:len(result["paths"])],
                f"budget={budget}",
            )

    def test_cost_limit_then_natural_found(self):
        # Asking for fewer routes than the limit admits still returns
        # found and the unlimited prefix.
        full = plan_k(3, 3, set(), (0, 0), (2, 2), 2)
        result = plan_k(3, 3, set(), (0, 0), (2, 2), 2, max_cost=4)
        self.assertEqual(result["status"], "found")
        self.assertEqual(result["paths"], full["paths"])


class PlanKLimitDynamicTest(unittest.TestCase):
    def test_generous_limits_match_unlimited(self):
        full = plan_k(3, 3, set(), (1, 0), (2, 2), 6,
                      dynamic_blocked=FRAMES)
        limited = plan_k(3, 3, set(), (1, 0), (2, 2), 6,
                         dynamic_blocked=FRAMES, max_expanded=1000,
                         max_cost=1000)
        self.assertEqual(limited["paths"], full["paths"])
        self.assertEqual(limited["costs"], full["costs"])
        self.assertEqual(limited["expanded"], full["expanded"])
        self.assertEqual(limited["status"], "found")

    def test_cost_boundary_keeps_dynamic_routes(self):
        full = plan_k(3, 3, set(), (1, 0), (2, 2), 6,
                      dynamic_blocked=FRAMES)
        boundary = full["costs"][-1]
        limited = plan_k(3, 3, set(), (1, 0), (2, 2), 6,
                         dynamic_blocked=FRAMES, max_cost=boundary)
        self.assertEqual(limited["status"], "found")
        self.assertEqual(limited["paths"], full["paths"])
        self.assertEqual(limited["costs"], full["costs"])

    def test_cost_below_minimum_is_cost_exhausted(self):
        minimum = plan_k(3, 3, set(), (1, 0), (2, 2), 1,
                         dynamic_blocked=FRAMES)["costs"][0]
        result = plan_k(3, 3, set(), (1, 0), (2, 2), 6,
                        dynamic_blocked=FRAMES, max_cost=minimum - 1)
        self.assertEqual(result["status"], "cost_exhausted")
        self.assertEqual(result["paths"], [])

    def test_budget_prefix_dynamic(self):
        full = plan_k(3, 3, set(), (1, 0), (2, 2), 6,
                      dynamic_blocked=FRAMES)
        result = plan_k(3, 3, set(), (1, 0), (2, 2), 6,
                        dynamic_blocked=FRAMES,
                        max_expanded=full["expanded"] - 1)
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(result["expanded"], full["expanded"] - 1)
        self.assertEqual(result["paths"],
                         full["paths"][:len(result["paths"])])
        for path, cost in zip(result["paths"], result["costs"]):
            checked = replay(3, 3, set(), (1, 0), (2, 2), path,
                             dynamic_blocked=FRAMES)
            self.assertTrue(checked["valid"])
            self.assertEqual(checked["cost"], cost)

    def test_zero_budget_dynamic(self):
        result = plan_k(3, 3, set(), (1, 0), (2, 2), 6,
                        dynamic_blocked=FRAMES, max_expanded=0)
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(result["expanded"], 0)
        self.assertEqual(result["paths"], [])

    def test_frame_permutation_invariance_with_limits(self):
        orderings = [
            FRAMES,
            [list(reversed(f)) for f in FRAMES],
            [set(f) for f in FRAMES],
            [tuple(f) + (f[0],) if f else () for f in FRAMES],
        ]
        results = [
            plan_k(6, 4, [(2, 0), (2, 0)], (0, 0), (5, 3), 8,
                   dynamic_blocked=f, max_expanded=20, max_cost=9)
            for f in orderings
        ]
        for other in results[1:]:
            self.assertEqual(other, results[0])

    def test_blocked_permutation_invariance_with_limits(self):
        blocked = {(2, 0), (2, 1), (2, 2), (0, 2), (4, 1)}
        orderings = (
            sorted(blocked),
            sorted(blocked, reverse=True),
            list(blocked)[2:] + list(blocked)[:2],
            [(2, 0), (2, 0), (2, 1), (2, 1), (2, 2), (0, 2), (4, 1)],
        )
        results = [
            plan_k(6, 4, cells, (0, 0), (5, 3), 8,
                   max_expanded=20, max_cost=9)
            for cells in orderings
        ]
        for other in results[1:]:
            self.assertEqual(other, results[0])


if __name__ == '__main__':
    unittest.main()
