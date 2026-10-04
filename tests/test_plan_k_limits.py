import itertools
import unittest

from app import plan, plan_k, replay


FRAMES = [
    [(0, 1), (0, 2)],
    [(0, 1), (1, 2), (2, 0), (2, 2)],
    [(2, 1), (2, 2)],
    [(0, 0), (1, 2), (2, 1), (2, 2)],
    [],
    [(0, 1)],
]
COSTS = [[1, 10, 1], [1, 10, 1], [1, 1, 1]]


class PlanKLimitValidationTest(unittest.TestCase):
    def test_type_errors(self):
        for bad in ("5", 1.5, True, False, [5], (5,), 2.0):
            with self.assertRaises(TypeError, msg=f"max_expanded={bad!r}"):
                plan_k(3, 3, set(), (0, 0), (2, 2), 3, max_expanded=bad)
            with self.assertRaises(TypeError, msg=f"max_cost={bad!r}"):
                plan_k(3, 3, set(), (0, 0), (2, 2), 3, max_cost=bad)

    def test_negative_is_value_error(self):
        for bad in (-1, -100):
            with self.assertRaises(ValueError, msg=f"max_expanded={bad!r}"):
                plan_k(3, 3, set(), (0, 0), (2, 2), 3, max_expanded=bad)
            with self.assertRaises(ValueError, msg=f"max_cost={bad!r}"):
                plan_k(3, 3, set(), (0, 0), (2, 2), 3, max_cost=bad)

    def test_zero_is_accepted(self):
        result = plan_k(3, 3, set(), (1, 1), (1, 1), 3, max_cost=0)
        self.assertEqual(result["status"], "found")
        self.assertEqual(result["costs"], [0])

    def test_validated_after_shared_checks_and_k(self):
        # Shared grid/costs/frame failures precede the limit type errors.
        with self.assertRaises(ValueError):
            plan_k(0, 3, set(), (0, 0), (2, 2), 3, max_cost="x")
        with self.assertRaises(TypeError):
            plan_k(3, 3, 42, (0, 0), (2, 2), 3, max_expanded="x")
        with self.assertRaises(ValueError):
            plan_k(3, 3, set(), (0, 0), (2, 2), 3, costs=[[1]],
                   max_cost="x")
        with self.assertRaises(ValueError):
            plan_k(3, 3, set(), (0, 0), (2, 2), 3,
                   dynamic_blocked=[[(0, 0)]], max_cost="x")
        # The k check precedes the limit checks.
        with self.assertRaises(ValueError):
            plan_k(3, 3, set(), (0, 0), (2, 2), 0, max_cost="x")
        with self.assertRaises(TypeError):
            plan_k(3, 3, set(), (0, 0), (2, 2), True, max_expanded="x")
        # Once all of those pass, the limit errors surface.
        with self.assertRaises(TypeError):
            plan_k(3, 3, set(), (0, 0), (2, 2), 3, max_expanded=1.5)
        with self.assertRaises(ValueError):
            plan_k(3, 3, set(), (0, 0), (2, 2), 3, max_cost=-2)

    def test_omitted_or_none_keeps_legacy_shape(self):
        legacy = plan_k(3, 3, set(), (0, 0), (2, 2), 6)
        self.assertEqual(set(legacy), {"paths", "costs", "expanded"})
        for kwargs in ({"max_expanded": None}, {"max_cost": None},
                       {"max_expanded": None, "max_cost": None}):
            self.assertEqual(
                plan_k(3, 3, set(), (0, 0), (2, 2), 6, **kwargs), legacy
            )

    def test_still_rejects_trace_and_snapshot(self):
        for kwargs in ({"trace": True}, {"snapshot": True}):
            with self.assertRaises(TypeError, msg=kwargs):
                plan_k(3, 3, set(), (0, 0), (2, 2), 3, **kwargs)


class PlanKMaxCostTest(unittest.TestCase):
    def test_found_when_k_routes_reach_goal(self):
        result = plan_k(3, 3, set(), (0, 0), (2, 2), 6, max_cost=4)
        self.assertEqual(result["status"], "found")
        self.assertEqual(len(result["paths"]), 6)
        self.assertEqual(result["costs"], [4] * 6)

    def test_cost_exhausted_with_empty_results(self):
        result = plan_k(3, 3, set(), (0, 0), (2, 2), 6, max_cost=3)
        self.assertEqual(result["status"], "cost_exhausted")
        self.assertEqual(result["paths"], [])
        self.assertEqual(result["costs"], [])
        # Every cost-feasible route-tree node is still actually closed
        # (one count per distinct simple history): the open 3x3 grid has
        # 15 simple route prefixes of at most four nodes ending off the
        # goal; only the g == 4 goal children are discarded at generation.
        self.assertEqual(result["expanded"], 15)

    def test_candidate_exactly_at_limit_still_competes(self):
        result = plan_k(3, 3, set(), (0, 0), (2, 2), 1, max_cost=4)
        self.assertEqual(result["status"], "found")
        self.assertEqual(result["costs"], [4])

    def test_limit_is_a_pure_filter_of_the_unlimited_ranking(self):
        unlimited = plan_k(3, 3, set(), (0, 0), (2, 0), 10, costs=COSTS)
        limited = plan_k(3, 3, set(), (0, 0), (2, 0), 10, costs=COSTS,
                         max_cost=6)
        surviving = [
            (p, c) for p, c in zip(unlimited["paths"], unlimited["costs"])
            if c <= 6
        ]
        self.assertEqual(
            list(zip(limited["paths"], limited["costs"])), surviving
        )
        # The top route is exactly what plan chooses.
        single = plan(3, 3, set(), (0, 0), (2, 0), costs=COSTS)
        self.assertEqual(limited["paths"][0], single["path"])
        self.assertEqual(limited["costs"][0], single["cost"])

    def test_natural_exhaustion_below_k_is_found(self):
        # The 2x2 grid offers exactly two simple routes; limiting to the
        # cheaper one exhausts the heap naturally with one route found.
        result = plan_k(2, 2, set(), (0, 0), (1, 1), 4,
                        costs=[[1, 1], [3, 1]], max_cost=2)
        self.assertEqual(result["status"], "found")
        self.assertEqual(result["paths"],
                         [[(0, 0), (1, 0), (1, 1)]])
        self.assertEqual(result["costs"], [2])

    def test_zero_limit_discards_every_step(self):
        result = plan_k(2, 1, set(), (0, 0), (1, 0), 4, max_cost=0)
        self.assertEqual(result["status"], "cost_exhausted")
        self.assertEqual(result["expanded"], 1)
        self.assertEqual(result["paths"], [])

    def test_start_equals_goal_zero_limit(self):
        self.assertEqual(
            plan_k(3, 3, set(), (1, 1), (1, 1), 3, max_cost=0),
            {"paths": [[(1, 1)]], "costs": [0], "expanded": 1,
             "status": "found"},
        )

    def test_unreachable_without_discards_stays_unreachable(self):
        wall = {(1, y) for y in range(3)}
        result = plan_k(3, 3, wall, (0, 0), (2, 2), 5, max_cost=100)
        self.assertEqual(result["status"], "unreachable")
        self.assertEqual(result["paths"], [])
        self.assertEqual(result["costs"], [])
        self.assertEqual(result["expanded"], 3)

    def test_returned_routes_replay(self):
        result = plan_k(3, 3, set(), (0, 0), (2, 0), 10, costs=COSTS,
                        max_cost=6)
        for path, cost in zip(result["paths"], result["costs"]):
            checked = replay(3, 3, set(), (0, 0), (2, 0), path, costs=COSTS)
            self.assertTrue(checked["valid"])
            self.assertEqual(checked["cost"], cost)
            self.assertEqual(checked["steps"], len(path) - 1)


class PlanKMaxExpandedTest(unittest.TestCase):
    def test_zero_budget_closes_nothing(self):
        self.assertEqual(
            plan_k(3, 3, set(), (0, 0), (2, 2), 3, max_expanded=0),
            {"paths": [], "costs": [], "expanded": 0,
             "status": "budget_exhausted"},
        )

    def test_zero_budget_start_equals_goal(self):
        result = plan_k(3, 3, set(), (1, 1), (1, 1), 3, max_expanded=0)
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(result["expanded"], 0)
        self.assertEqual(result["paths"], [])

    def test_every_budget_prefix_matches_the_unlimited_ranking(self):
        full = plan_k(3, 3, set(), (0, 0), (2, 2), 20)
        for budget in range(1, full["expanded"]):
            result = plan_k(3, 3, set(), (0, 0), (2, 2), 20,
                            max_expanded=budget)
            self.assertEqual(result["expanded"], budget, f"budget={budget}")
            self.assertEqual(result["status"], "budget_exhausted",
                             f"budget={budget}")
            self.assertEqual(
                result["paths"], full["paths"][:len(result["paths"])],
                f"budget={budget}",
            )
            self.assertEqual(
                result["costs"], full["costs"][:len(result["costs"])],
                f"budget={budget}",
            )
        # A budget large enough to finish reproduces the unlimited answer.
        done = plan_k(3, 3, set(), (0, 0), (2, 2), 20,
                      max_expanded=full["expanded"])
        self.assertEqual(done["status"], "found")
        self.assertEqual(done["paths"], full["paths"])
        self.assertEqual(done["costs"], full["costs"])

    def test_natural_exhaustion_with_fewer_than_k_is_found(self):
        result = plan_k(5, 1, set(), (0, 0), (4, 0), 100, max_expanded=100)
        self.assertEqual(result["status"], "found")
        self.assertEqual(
            result["paths"],
            [[(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]],
        )
        self.assertEqual(result["costs"], [4])
        self.assertEqual(result["expanded"], 5)

    def test_budget_truncation_with_pending_candidates(self):
        wall = {(1, y) for y in range(3)}
        result = plan_k(3, 3, wall, (0, 0), (2, 2), 5, max_expanded=2)
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(result["expanded"], 2)

    def test_completing_k_within_budget_is_found(self):
        result = plan_k(3, 3, set(), (0, 0), (2, 2), 1, max_expanded=100)
        self.assertEqual(result["status"], "found")
        self.assertEqual(len(result["paths"]), 1)

    def test_budget_only_truncates_never_reorders(self):
        # The first closed goal route under any budget equals plan's route.
        first = plan_k(3, 3, set(), (0, 0), (2, 2), 1)
        single = plan(3, 3, set(), (0, 0), (2, 2))
        self.assertEqual(first["paths"][0], single["path"])
        self.assertEqual(first["costs"][0], single["cost"])


class PlanKCombinedLimitsTest(unittest.TestCase):
    def test_budget_exhausted_takes_priority(self):
        blocked = {(2, 0), (2, 1), (2, 2)}
        result = plan_k(6, 4, blocked, (0, 0), (5, 3), 8,
                        max_expanded=3, max_cost=2)
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(result["expanded"], 3)

    def test_cost_exhausted_when_heap_empties_at_the_boundary(self):
        # With a zero cost limit every generated child is discarded, so no
        # candidate remains pending when the one-close budget is reached:
        # the cost-limit status applies (same rule as ``plan``).
        result = plan_k(3, 3, set(), (0, 0), (2, 2), 6,
                        max_expanded=1, max_cost=0)
        self.assertEqual(result["status"], "cost_exhausted")
        self.assertEqual(result["paths"], [])

    def test_routes_closed_before_truncation_are_kept(self):
        # The cost limit removes every longer candidate after the six
        # cost-4 goal routes close; the search then exhausts naturally.
        # A budget merely larger than the cost-feasible route tree must
        # not turn that natural exhaustion into a budget stop.
        result = plan_k(3, 3, set(), (0, 0), (2, 2), 20,
                        max_expanded=1000, max_cost=4)
        self.assertEqual(result["status"], "found")
        self.assertEqual(len(result["paths"]), 6)
        self.assertEqual(result["costs"], [4] * 6)

    def test_paths_and_costs_stay_aligned_and_ranked(self):
        for kwargs in ({"max_expanded": 7}, {"max_cost": 5},
                       {"max_expanded": 7, "max_cost": 5}):
            result = plan_k(3, 3, set(), (0, 0), (2, 2), 20, **kwargs)
            self.assertEqual(len(result["paths"]), len(result["costs"]))
            keys = list(zip(result["costs"],
                            [tuple(p) for p in result["paths"]]))
            self.assertEqual(keys, sorted(keys))


class PlanKLimitsDynamicTest(unittest.TestCase):
    def test_high_limits_match_unlimited_search(self):
        full = plan_k(3, 3, set(), (1, 0), (2, 2), 6,
                      dynamic_blocked=FRAMES)
        limited = plan_k(3, 3, set(), (1, 0), (2, 2), 6,
                         dynamic_blocked=FRAMES, max_cost=10 ** 9)
        self.assertEqual(limited["status"], "found")
        self.assertEqual(limited["paths"], full["paths"])
        self.assertEqual(limited["costs"], full["costs"])
        self.assertEqual(limited["expanded"], full["expanded"])

    def test_cost_limit_keeps_cheapest_route_and_replays(self):
        full = plan_k(3, 3, set(), (1, 0), (2, 2), 6,
                      dynamic_blocked=FRAMES)
        result = plan_k(3, 3, set(), (1, 0), (2, 2), 6,
                        dynamic_blocked=FRAMES, max_cost=full["costs"][0])
        self.assertEqual(result["paths"][0], full["paths"][0])
        self.assertEqual(result["costs"][0], full["costs"][0])
        for path, cost in zip(result["paths"], result["costs"]):
            checked = replay(3, 3, set(), (1, 0), (2, 2), path,
                             dynamic_blocked=FRAMES)
            self.assertTrue(checked["valid"])
            self.assertEqual(checked["cost"], cost)
            self.assertEqual(checked["steps"], len(path) - 1)

    def test_cost_exhausted_when_detour_is_too_expensive(self):
        # The direct arrival at (2, 0) at frame 1 is blocked; every
        # feasible route needs a detour.
        frames = [[], [(2, 0)], []]
        full = plan_k(3, 2, set(), (0, 0), (2, 1), 10,
                      dynamic_blocked=frames)
        too_small = plan_k(3, 2, set(), (0, 0), (2, 1), 10,
                           dynamic_blocked=frames,
                           max_cost=full["costs"][0] - 1)
        self.assertEqual(too_small["status"], "cost_exhausted")
        self.assertEqual(too_small["paths"], [])
        at_limit = plan_k(3, 2, set(), (0, 0), (2, 1), 10,
                          dynamic_blocked=frames, max_cost=full["costs"][0])
        self.assertEqual(at_limit["status"], "found")
        self.assertEqual(at_limit["paths"][0], full["paths"][0])

    def test_budget_prefixes_in_dynamic_mode(self):
        full = plan_k(3, 3, set(), (1, 0), (2, 2), 6,
                      dynamic_blocked=FRAMES)
        for budget in range(1, full["expanded"]):
            result = plan_k(3, 3, set(), (1, 0), (2, 2), 6,
                            dynamic_blocked=FRAMES, max_expanded=budget)
            self.assertEqual(result["expanded"], budget)
            self.assertEqual(result["status"], "budget_exhausted")
            self.assertEqual(
                result["paths"], full["paths"][:len(result["paths"])]
            )

    def test_persistent_frame_unreachable_statuses(self):
        frames = [[], [(2, 0)]]
        result = plan_k(3, 1, set(), (0, 0), (2, 0), 4,
                        dynamic_blocked=frames, max_expanded=100)
        self.assertEqual(result["status"], "unreachable")
        self.assertEqual(result["expanded"], 2)
        truncated = plan_k(3, 1, set(), (0, 0), (2, 0), 4,
                           dynamic_blocked=frames, max_expanded=1)
        self.assertEqual(truncated["status"], "budget_exhausted")
        self.assertEqual(truncated["expanded"], 1)


class PlanKLimitsDeterminismTest(unittest.TestCase):
    def test_blocked_permutation_invariance(self):
        cells = [(2, 0), (2, 1), (0, 2), (4, 1)]
        results = [
            plan_k(6, 4, list(ordering), (0, 0), (5, 3), 8,
                   max_expanded=40, max_cost=20)
            for ordering in itertools.permutations(cells)
        ]
        for other in results[1:]:
            self.assertEqual(other, results[0])

    def test_frame_permutation_invariance(self):
        orderings = [
            FRAMES,
            [list(reversed(frame)) for frame in FRAMES],
            [set(frame) for frame in FRAMES],
            [tuple(frame) + (frame[0],) if frame else ()
             for frame in FRAMES],
        ]
        results = [
            plan_k(3, 3, [(2, 0), (2, 0)], (1, 0), (2, 2), 6,
                   dynamic_blocked=frames, max_expanded=40, max_cost=9)
            for frames in orderings
        ]
        for other in results[1:]:
            self.assertEqual(other, results[0])


if __name__ == '__main__':
    unittest.main()
