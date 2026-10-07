import json
import unittest

from app import plan, plan_agents, replay


FRAMES = [[(1, 0)], [], []]
COSTS = [[1, 10, 1], [1, 1, 1]]
DYNAMIC_COSTS = [[[1, 1, 1], [1, 1, 1]], [[5, 5, 5], [1, 1, 1]]]
RESERVATIONS = [[(0, 1), (1, 1), (2, 1)]]


def occupant(path, t):
    # The cell a route occupies at frame ``t``; the final coordinate
    # persists past the last frame.
    return path[t] if t < len(path) else path[-1]


def assert_conflict_free(testcase, paths):
    # No two routes share a vertex at the same frame (goal occupancy
    # persists) and no two routes traverse the same edge in opposite
    # directions on the same transition.
    paths = [path for path in paths if path is not None]
    horizon = max(len(path) for path in paths)
    for t in range(horizon):
        cells = [occupant(path, t) for path in paths]
        testcase.assertEqual(len(set(cells)), len(cells),
                             msg=f"vertex conflict at frame {t}: {cells}")
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            first, second = paths[i], paths[j]
            for t in range(max(len(first), len(second)) - 1):
                edge_a = (occupant(first, t), occupant(first, t + 1))
                edge_b = (occupant(second, t), occupant(second, t + 1))
                swapped = (edge_a[0] == edge_b[1]
                           and edge_a[1] == edge_b[0]
                           and edge_a[0] != edge_a[1])
                testcase.assertFalse(
                    swapped,
                    msg=f"reverse edge swap at frame {t}: "
                        f"{edge_a} vs {edge_b}",
                )


class FoundResultTest(unittest.TestCase):
    def test_result_shape(self):
        result = plan_agents(3, 2, set(), [[(0, 0), (2, 0)], [(2, 0), (0, 0)]])
        self.assertEqual(list(result.keys()),
                         ["status", "results", "expanded"])
        self.assertEqual(result["status"], "found")
        self.assertEqual(len(result["results"]), 2)
        self.assertEqual(
            result["expanded"],
            sum(item["expanded"] for item in result["results"]),
        )
        for item in result["results"]:
            self.assertEqual(set(item.keys()), {"path", "cost", "expanded"})

    def test_single_agent_matches_plan(self):
        result = plan_agents(3, 3, {(1, 1)}, [[(0, 0), (2, 2)]])
        expected = plan(3, 3, {(1, 1)}, (0, 0), (2, 2))
        self.assertEqual(result["results"][0], expected)
        self.assertEqual(result["expanded"], expected["expanded"])

    def test_first_agent_matches_plan_in_dynamic_mode(self):
        requests = [[(0, 0), (2, 0)], [(0, 1), (2, 1)]]
        result = plan_agents(3, 2, set(), requests,
                             dynamic_blocked=FRAMES, trace=True)
        expected = plan(3, 2, set(), (0, 0), (2, 0),
                        dynamic_blocked=FRAMES, trace=True)
        self.assertEqual(result["results"][0], expected)

    def test_start_equals_goal(self):
        result = plan_agents(3, 3, set(), [[(1, 1), (1, 1)]])
        self.assertEqual(result["status"], "found")
        self.assertEqual(result["results"][0],
                         {"path": [(1, 1)], "cost": 0, "expanded": 1})

    def test_trace_adds_expanded_nodes_per_item(self):
        result = plan_agents(3, 2, set(), [[(0, 0), (2, 0)], [(2, 0), (0, 0)]],
                             trace=True)
        for item in result["results"]:
            self.assertIn("expanded_nodes", item)
            self.assertEqual(item["expanded"], len(item["expanded_nodes"]))
        # The first agent searches the static grid (coordinate pairs);
        # later agents search the time-expanded space ((x, y, t) triples).
        self.assertEqual(result["results"][0]["expanded_nodes"][0], (0, 0))
        self.assertEqual(result["results"][1]["expanded_nodes"][0], (2, 0, 0))

    def test_no_trace_key_by_default(self):
        result = plan_agents(3, 2, set(), [[(0, 0), (2, 0)], [(2, 0), (0, 0)]])
        for item in result["results"]:
            self.assertNotIn("expanded_nodes", item)

    def test_json_serializable(self):
        result = plan_agents(3, 2, set(), [[(0, 0), (2, 0)], [(2, 0), (0, 0)]],
                             trace=True)
        self.assertEqual(json.loads(json.dumps(result)),
                         json.loads(json.dumps(result)))


class PriorityTest(unittest.TestCase):
    def test_later_agent_detours_around_earlier_route(self):
        result = plan_agents(3, 2, set(), [[(0, 0), (2, 0)], [(2, 0), (0, 0)]])
        self.assertEqual(result["status"], "found")
        self.assertEqual(
            result["results"][0],
            {"path": [(0, 0), (1, 0), (2, 0)], "cost": 2, "expanded": 3})
        self.assertEqual(
            result["results"][1],
            {"path": [(2, 0), (2, 1), (1, 1), (0, 1), (0, 0)], "cost": 4,
             "expanded": 7})
        self.assertEqual(result["expanded"], 10)

    def test_request_order_fixes_priority(self):
        forward = plan_agents(3, 2, set(), [[(0, 0), (2, 0)], [(2, 0), (0, 0)]])
        swapped = plan_agents(3, 2, set(), [[(2, 0), (0, 0)], [(0, 0), (2, 0)]])
        self.assertEqual(forward["results"][0]["path"],
                         [(0, 0), (1, 0), (2, 0)])
        self.assertEqual(swapped["results"][0]["path"],
                         [(2, 0), (1, 0), (0, 0)])
        # Whoever plans first keeps the direct corridor.
        self.assertNotEqual(forward["results"][1]["path"],
                            swapped["results"][1]["path"])

    def test_duplicate_requests_are_kept(self):
        result = plan_agents(3, 3, set(), [[(0, 0), (2, 0)], [(0, 0), (2, 0)]])
        # The duplicate's start is occupied at frame 0 by the first
        # agent's route, so the second identical request cannot start.
        self.assertEqual(result["status"], "unreachable")
        self.assertEqual(result["failed_index"], 1)
        self.assertEqual(len(result["results"]), 2)
        self.assertEqual(result["results"][1]["expanded"], 0)

    def test_earlier_goal_stays_occupied(self):
        # The second agent's goal is the first agent's final cell, whose
        # occupancy persists: no arrival can ever be safe.
        result = plan_agents(3, 2, set(), [[(0, 0), (2, 0)], [(0, 1), (2, 0)]])
        self.assertEqual(result["status"], "unreachable")
        self.assertEqual(result["failed_index"], 1)
        self.assertIsNone(result["results"][1]["path"])
        self.assertIsNone(result["results"][1]["cost"])

    def test_goal_on_earlier_route_entered_after_it_is_vacated(self):
        # (1, 0) is a mid-route cell of the first agent, occupied only at
        # frame 1; the second agent ends there once it is free for good.
        result = plan_agents(4, 3, set(), [[(0, 0), (3, 0)], [(0, 1), (1, 0)]])
        self.assertEqual(result["status"], "found")
        self.assertEqual(result["results"][1]["path"],
                         [(0, 1), (0, 0), (1, 0)])
        self.assertEqual(result["results"][1]["cost"], 2)

    def test_goal_may_not_be_parked_before_an_earlier_route_passes(self):
        # The first agent crosses (2, 1) at frame 3 on its way to (2, 0);
        # the second agent could reach (2, 1) at frame 2, but parking
        # there would violate the first agent at frame 3, and no feasible
        # later arrival exists on this grid.
        result = plan_agents(3, 2, set(), [[(0, 0), (2, 0)], [(0, 1), (2, 1)]],
                             dynamic_costs=DYNAMIC_COSTS)
        self.assertEqual(result["status"], "unreachable")
        self.assertEqual(result["failed_index"], 1)
        self.assertEqual(result["results"][0]["cost"], 8)

    def test_start_equals_goal_on_reserved_cell(self):
        # The single-point route would occupy (1, 0) from frame 0 on, but
        # the first agent crosses it at frame 1.
        result = plan_agents(3, 1, set(), [[(0, 0), (2, 0)], [(1, 0), (1, 0)]])
        self.assertEqual(result["status"], "unreachable")
        self.assertEqual(result["failed_index"], 1)
        self.assertEqual(result["results"][1]["expanded"], 0)

    def test_start_equals_goal_on_free_cell(self):
        result = plan_agents(3, 2, set(), [[(0, 0), (2, 0)], [(1, 1), (1, 1)]])
        self.assertEqual(result["status"], "found")
        self.assertEqual(result["results"][1],
                         {"path": [(1, 1)], "cost": 0, "expanded": 1})


class FailureTest(unittest.TestCase):
    def test_first_agent_unreachable_stops_batch(self):
        blocked = {(1, 0), (0, 1)}
        result = plan_agents(3, 3, blocked,
                             [[(0, 0), (2, 2)], [(1, 1), (2, 2)]])
        self.assertEqual(list(result.keys()),
                         ["status", "failed_index", "results", "expanded"])
        self.assertEqual(result["status"], "unreachable")
        self.assertEqual(result["failed_index"], 0)
        # The second request is never searched.
        self.assertEqual(len(result["results"]), 1)
        self.assertEqual(result["results"][0],
                         {"path": None, "cost": None, "expanded": 1})
        self.assertEqual(result["expanded"], 1)

    def test_mid_batch_failure_keeps_successful_prefix(self):
        blocked = {(1, 2), (2, 1)}
        requests = [[(0, 0), (1, 0)], [(0, 2), (2, 2)], [(0, 0), (1, 1)]]
        result = plan_agents(3, 3, blocked, requests)
        self.assertEqual(result["status"], "unreachable")
        self.assertEqual(result["failed_index"], 1)
        self.assertEqual(len(result["results"]), 2)  # third never searched
        self.assertEqual(result["results"][0],
                         {"path": [(0, 0), (1, 0)], "cost": 1, "expanded": 2})
        failed = result["results"][1]
        self.assertIsNone(failed["path"])
        self.assertIsNone(failed["cost"])
        self.assertEqual(failed["expanded"], 4)
        # The total counts only the agents actually attempted.
        self.assertEqual(result["expanded"], 6)

    def test_failed_item_carries_trace(self):
        blocked = {(1, 2), (2, 1)}
        result = plan_agents(3, 3, blocked,
                             [[(0, 0), (1, 0)], [(0, 2), (2, 2)]],
                             trace=True)
        failed = result["results"][1]
        self.assertIn("expanded_nodes", failed)
        self.assertEqual(failed["expanded"], len(failed["expanded_nodes"]))


class SharedParameterTest(unittest.TestCase):
    def test_costs(self):
        requests = [[(0, 0), (2, 0)], [(2, 0), (0, 0)]]
        result = plan_agents(3, 2, set(), requests, costs=COSTS)
        self.assertEqual(result["status"], "found")
        assert_conflict_free(self, [item["path"] for item in result["results"]])
        for (start, goal), item in zip(requests, result["results"]):
            outcome = replay(3, 2, set(), start, goal, item["path"],
                             costs=COSTS)
            self.assertEqual(
                outcome,
                {"valid": True, "cost": item["cost"],
                 "steps": len(item["path"]) - 1})

    def test_dynamic_blocked(self):
        requests = [[(0, 0), (2, 0)], [(0, 1), (2, 1)]]
        result = plan_agents(3, 2, set(), requests, dynamic_blocked=FRAMES)
        self.assertEqual(result["status"], "found")
        assert_conflict_free(self, [item["path"] for item in result["results"]])
        for (start, goal), item in zip(requests, result["results"]):
            outcome = replay(3, 2, set(), start, goal, item["path"],
                             dynamic_blocked=FRAMES)
            self.assertEqual(outcome["cost"], item["cost"])
            self.assertTrue(outcome["valid"])

    def test_dynamic_costs(self):
        requests = [[(0, 0), (2, 0)], [(0, 1), (2, 1)]]
        result = plan_agents(3, 2, set(), requests,
                             dynamic_costs=DYNAMIC_COSTS, allow_wait=True)
        # The second agent cannot park on (2, 1) before the first agent
        # passes through it, and no feasible later arrival exists here.
        self.assertEqual(result["status"], "unreachable")
        self.assertEqual(result["failed_index"], 1)
        assert_conflict_free(self, [item["path"] for item in result["results"]])
        for (start, goal), item in zip(requests, result["results"]):
            if item["path"] is None:
                continue
            outcome = replay(3, 2, set(), start, goal, item["path"],
                             dynamic_costs=DYNAMIC_COSTS, allow_wait=True)
            self.assertEqual(outcome["cost"], item["cost"])
            self.assertTrue(outcome["valid"])

    def test_allow_wait(self):
        requests = [[(0, 0), (2, 0)], [(2, 0), (0, 1)]]
        result = plan_agents(3, 2, set(), requests, allow_wait=True)
        self.assertEqual(result["status"], "found")
        assert_conflict_free(self, [item["path"] for item in result["results"]])
        for (start, goal), item in zip(requests, result["results"]):
            outcome = replay(3, 2, set(), start, goal, item["path"],
                             allow_wait=True)
            self.assertEqual(
                outcome,
                {"valid": True, "cost": item["cost"],
                 "steps": len(item["path"]) - 1})

    def test_reservations(self):
        requests = [[(0, 0), (2, 0)], [(2, 0), (0, 0)]]
        result = plan_agents(3, 2, set(), requests,
                             reservations=RESERVATIONS)
        assert_conflict_free(
            self,
            [item["path"] for item in result["results"]]
            + [tuple(RESERVATIONS[0])])
        for (start, goal), item in zip(requests, result["results"]):
            if item["path"] is None:
                continue
            outcome = replay(3, 2, set(), start, goal, item["path"],
                             reservations=RESERVATIONS)
            self.assertEqual(outcome["cost"], item["cost"])
            self.assertTrue(outcome["valid"])

    def test_goal_on_reservation_tail_is_unreachable(self):
        result = plan_agents(3, 3, set(), [[(0, 0), (2, 1)]],
                             reservations=RESERVATIONS)
        self.assertEqual(result["status"], "unreachable")
        self.assertEqual(result["failed_index"], 0)

    def test_goal_on_reservation_mid_cell_waits_for_it(self):
        # (1, 1) is occupied by the reservation only at frame 1, so the
        # agent ends there at frame 2.
        result = plan_agents(3, 3, set(), [[(0, 0), (1, 1)]],
                             reservations=RESERVATIONS)
        self.assertEqual(result["status"], "found")
        self.assertEqual(result["results"][0]["path"],
                         [(0, 0), (0, 1), (1, 1)])
        outcome = replay(3, 3, set(), (0, 0), (1, 1),
                         result["results"][0]["path"],
                         reservations=RESERVATIONS)
        self.assertEqual(outcome, {"valid": True, "cost": 2, "steps": 2})

    def test_three_agents_stay_conflict_free(self):
        requests = [[(0, 0), (3, 3)], [(3, 0), (0, 3)], [(0, 3), (3, 0)]]
        result = plan_agents(4, 4, set(), requests)
        self.assertEqual(result["status"], "found")
        assert_conflict_free(self, [item["path"] for item in result["results"]])
        for (start, goal), item in zip(requests, result["results"]):
            outcome = replay(4, 4, set(), start, goal, item["path"])
            self.assertEqual(
                outcome,
                {"valid": True, "cost": item["cost"],
                 "steps": len(item["path"]) - 1})


class ValidationTest(unittest.TestCase):
    def test_requests_type_errors(self):
        for value in [None, "requests", b"requests", 42, {(0, 0)}]:
            with self.assertRaises(TypeError, msg=repr(value)):
                plan_agents(3, 3, set(), value)
        # An element that is not a [start, goal] pair.
        with self.assertRaises(TypeError):
            plan_agents(3, 3, set(), ["pair"])
        with self.assertRaises(TypeError):
            plan_agents(3, 3, set(), [[(0, 0)]])
        with self.assertRaises(TypeError):
            plan_agents(3, 3, set(), [[(0, 0), (1, 1), (2, 2)]])
        with self.assertRaises(TypeError):
            plan_agents(3, 3, set(), [[(0, 0), "goal"]])
        with self.assertRaises(TypeError):
            plan_agents(3, 3, set(), [[(0, 0), (True, 1)]])

    def test_empty_requests_value_error(self):
        with self.assertRaises(ValueError):
            plan_agents(3, 3, set(), [])

    def test_endpoint_value_errors(self):
        with self.assertRaises(ValueError):
            plan_agents(3, 3, set(), [[(0, 0), (3, 0)]])
        with self.assertRaises(ValueError):
            plan_agents(3, 3, set(), [[(-1, 0), (2, 2)]])
        with self.assertRaises(ValueError):
            plan_agents(3, 3, {(1, 1)}, [[(1, 1), (2, 2)]])
        with self.assertRaises(ValueError):
            plan_agents(3, 3, {(1, 1)}, [[(0, 0), (1, 1)]])

    def test_shared_parameter_type_errors(self):
        requests = [[(0, 0), (2, 2)]]
        with self.assertRaises(TypeError):
            plan_agents(3, 3, set(), requests, costs="costs")
        with self.assertRaises(TypeError):
            plan_agents(3, 3, set(), requests, trace=1)
        with self.assertRaises(TypeError):
            plan_agents(3, 3, set(), requests, allow_wait=1)
        with self.assertRaises(TypeError):
            plan_agents(3, 3, set(), requests, dynamic_blocked="frames")
        with self.assertRaises(TypeError):
            plan_agents(3, 3, set(), requests, dynamic_costs="frames")
        with self.assertRaises(TypeError):
            plan_agents(3, 3, set(), requests, reservations="paths")

    def test_shared_parameter_value_errors(self):
        requests = [[(0, 0), (2, 2)]]
        with self.assertRaises(ValueError):
            plan_agents(3, 3, set(), requests,
                        dynamic_blocked=[[(0, 0)]])  # start at frame 0
        with self.assertRaises(ValueError):
            plan_agents(3, 3, set(), requests,
                        dynamic_blocked=[[(3, 3)]])  # out of bounds
        with self.assertRaises(ValueError):
            plan_agents(3, 3, set(), requests, costs=[[1, 1], [1, 1]])
        with self.assertRaises(ValueError):
            plan_agents(3, 3, set(), requests,
                        costs=[[1] * 3] * 3,
                        dynamic_costs=[[[1] * 3] * 3])
        with self.assertRaises(ValueError):
            plan_agents(3, 3, set(), requests, reservations=[[]])
        with self.assertRaises(ValueError):
            plan_agents(3, 3, {(1, 1)}, requests,
                        reservations=[[(1, 1)]])

    def test_invalid_batch_never_returns_partial_results(self):
        # A structurally valid first request does not protect an invalid
        # later one: the error is raised before any search starts.
        with self.assertRaises(ValueError):
            plan_agents(3, 3, set(), [[(0, 0), (2, 2)], [(0, 0), (9, 9)]])

    def test_dimension_validation(self):
        with self.assertRaises(TypeError):
            plan_agents("3", 3, set(), [[(0, 0), (2, 2)]])
        with self.assertRaises(ValueError):
            plan_agents(0, 3, set(), [[(0, 0), (2, 2)]])


class BudgetShapeTest(unittest.TestCase):
    def test_omitted_budget_leaves_everything_untouched(self):
        requests = [[(0, 0), (2, 0)], [(2, 0), (0, 0)]]
        omitted = plan_agents(3, 2, set(), requests)
        explicit = plan_agents(3, 2, set(), requests, max_expanded=None)
        self.assertEqual(omitted, explicit)
        self.assertEqual(list(omitted.keys()),
                         ["status", "results", "expanded"])
        for item in omitted["results"]:
            self.assertEqual(set(item.keys()), {"path", "cost", "expanded"})
            self.assertNotIn("status", item)

    def test_found_items_carry_status_when_budget_is_given(self):
        requests = [[(0, 0), (2, 0)], [(2, 0), (0, 0)]]
        result = plan_agents(3, 2, set(), requests, max_expanded=10)
        self.assertEqual(result["status"], "found")
        self.assertEqual(result["expanded"], 10)
        self.assertEqual([item["status"] for item in result["results"]],
                         ["found", "found"])

    def test_large_budget_matches_unbudgeted_result(self):
        requests = [[(0, 0), (2, 0)], [(2, 0), (0, 0)]]
        unbudgeted = plan_agents(3, 2, set(), requests, trace=True)
        budgeted = plan_agents(3, 2, set(), requests, trace=True,
                               max_expanded=1000)
        self.assertEqual(budgeted["status"], "found")
        self.assertEqual(
            [{k: v for k, v in item.items() if k != "status"}
             for item in budgeted["results"]],
            unbudgeted["results"],
        )
        self.assertEqual(budgeted["expanded"], unbudgeted["expanded"])


class SharedBudgetTest(unittest.TestCase):
    REQUESTS = [[(0, 0), (2, 0)], [(2, 0), (0, 0)]]

    def test_budget_is_shared_cumulatively_across_agents(self):
        # The first agent closes 3 static nodes; the second's
        # time-expanded search needs 5 closings to settle its goal, so a
        # total budget of 7 leaves it 4 and stops the batch mid-search.
        result = plan_agents(3, 2, set(), self.REQUESTS,
                             max_expanded=7, trace=True)
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(result["failed_index"], 1)
        self.assertEqual(result["expanded"], 7)
        self.assertEqual(len(result["results"]), 2)
        first, second = result["results"]
        self.assertEqual(first["status"], "found")
        self.assertEqual(first["path"], [(0, 0), (1, 0), (2, 0)])
        self.assertEqual(first["expanded"], 3)
        self.assertIsNone(second["path"])
        self.assertIsNone(second["cost"])
        self.assertEqual(second["status"], "budget_exhausted")
        self.assertEqual(second["expanded"], 4)
        self.assertEqual(len(second["expanded_nodes"]), 4)

    def test_budget_carries_over_to_later_agents(self):
        # The three searches close 4, 18 and 11 nodes. With 22 the first
        # two finish and the third never closes a node; with 23 it gets
        # exactly one closing before the shared allowance runs out.
        requests = [[(0, 0), (3, 0)], [(3, 0), (0, 0)], [(0, 1), (3, 1)]]
        none_left = plan_agents(4, 2, set(), requests, max_expanded=22)
        self.assertEqual(none_left["status"], "budget_exhausted")
        self.assertEqual(none_left["failed_index"], 2)
        self.assertEqual(
            [item["expanded"] for item in none_left["results"]],
            [4, 18, 0])
        self.assertEqual(
            [item["status"] for item in none_left["results"]],
            ["found", "found", "budget_exhausted"])
        one_left = plan_agents(4, 2, set(), requests, max_expanded=23)
        self.assertEqual(one_left["status"], "budget_exhausted")
        self.assertEqual(one_left["failed_index"], 2)
        self.assertEqual(one_left["expanded"], 23)
        self.assertEqual(
            [item["expanded"] for item in one_left["results"]],
            [4, 18, 1])

    def test_closing_goal_exactly_on_budget_succeeds(self):
        # Remaining allowance 5 closes the second agent's goal as its
        # fifth node: an exact budget is a success, not exhaustion.
        result = plan_agents(3, 2, set(), self.REQUESTS, max_expanded=8)
        self.assertEqual(result["status"], "found")
        self.assertEqual(result["expanded"], 8)
        second = result["results"][1]
        self.assertEqual(second["status"], "found")
        self.assertEqual(second["expanded"], 5)
        self.assertEqual(second["path"],
                         [(2, 0), (2, 1), (1, 1), (0, 1), (0, 0)])

    def test_exhaustion_can_hit_the_first_agent(self):
        result = plan_agents(3, 2, set(), self.REQUESTS,
                             max_expanded=2, trace=True)
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(result["failed_index"], 0)
        self.assertEqual(result["expanded"], 2)
        self.assertEqual(len(result["results"]), 1)  # second never searched
        item = result["results"][0]
        self.assertIsNone(item["path"])
        self.assertIsNone(item["cost"])
        self.assertEqual(item["status"], "budget_exhausted")
        self.assertEqual(item["expanded"], 2)
        # The static first agent records coordinate pairs, only actually
        # closed nodes, and its trace matches its closing count.
        self.assertEqual(item["expanded_nodes"], [(0, 0), (1, 0)])

    def test_zero_budget_exhausts_even_for_start_equals_goal(self):
        result = plan_agents(3, 3, set(), [[(1, 1), (1, 1)]],
                             max_expanded=0, trace=True)
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(result["failed_index"], 0)
        self.assertEqual(result["expanded"], 0)
        item = result["results"][0]
        self.assertEqual(item["status"], "budget_exhausted")
        self.assertIsNone(item["path"])
        self.assertEqual(item["expanded"], 0)
        self.assertEqual(item["expanded_nodes"], [])
        # One unit of budget closes the single-point route.
        one = plan_agents(3, 3, set(), [[(1, 1), (1, 1)]], max_expanded=1)
        self.assertEqual(one["status"], "found")
        self.assertEqual(one["results"][0]["path"], [(1, 1)])

    def test_expanded_never_exceeds_budget(self):
        for budget in range(0, 11):
            result = plan_agents(3, 2, set(), self.REQUESTS,
                                 max_expanded=budget, trace=True)
            self.assertLessEqual(result["expanded"], budget)
            for item in result["results"]:
                self.assertEqual(item["expanded"],
                                 len(item["expanded_nodes"]))


class BudgetExhaustionTest(unittest.TestCase):
    def test_batch_shape_keeps_prefix_and_current_item(self):
        requests = [[(0, 0), (2, 0)], [(2, 0), (0, 0)],
                    [(0, 1), (2, 1)]]
        result = plan_agents(3, 2, set(), requests, max_expanded=5)
        self.assertEqual(list(result.keys()),
                         ["status", "failed_index", "results", "expanded"])
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(result["failed_index"], 1)
        # The third request is never searched; the current item is kept.
        self.assertEqual(len(result["results"]), 2)
        self.assertEqual(result["results"][0]["status"], "found")
        current = result["results"][1]
        self.assertEqual(current["status"], "budget_exhausted")
        self.assertIsNone(current["path"])
        self.assertIsNone(current["cost"])
        self.assertEqual(current["expanded"], 2)
        self.assertEqual(result["expanded"], 5)

    def test_trace_records_only_actually_closed_nodes(self):
        requests = [[(0, 0), (2, 0)], [(2, 0), (0, 0)]]
        result = plan_agents(3, 2, set(), requests,
                             max_expanded=4, trace=True)
        self.assertEqual(result["status"], "budget_exhausted")
        first, second = result["results"]
        # The successful prefix keeps its complete static trace.
        self.assertEqual(first["expanded_nodes"],
                         [(0, 0), (1, 0), (2, 0)])
        self.assertEqual(second["expanded_nodes"], [(2, 0, 0)])
        self.assertEqual(second["expanded"], 1)
        # No trace key is added when tracing is off.
        plain = plan_agents(3, 2, set(), requests, max_expanded=4)
        for item in plain["results"]:
            self.assertNotIn("expanded_nodes", item)

    def test_result_is_json_serializable(self):
        requests = [[(0, 0), (2, 0)], [(2, 0), (0, 0)]]
        result = plan_agents(3, 2, set(), requests,
                             max_expanded=6, trace=True)
        self.assertEqual(result["status"], "budget_exhausted")
        decoded = json.loads(json.dumps(result))
        self.assertEqual(decoded["status"], "budget_exhausted")
        self.assertEqual(decoded["failed_index"], 1)
        self.assertEqual(decoded["expanded"], result["expanded"])


class BudgetVsUnreachableTest(unittest.TestCase):
    def test_natural_exhaustion_under_budget_stays_unreachable(self):
        # The start cell is isolated; its single closing exhausts the
        # candidates, so a generous budget still reports unreachable.
        blocked = {(1, 0), (0, 1)}
        result = plan_agents(
            3, 3, blocked, [[(0, 0), (2, 2)], [(1, 1), (2, 2)]],
            max_expanded=100)
        self.assertEqual(list(result.keys()),
                         ["status", "failed_index", "results", "expanded"])
        self.assertEqual(result["status"], "unreachable")
        self.assertEqual(result["failed_index"], 0)
        self.assertEqual(len(result["results"]), 1)
        item = result["results"][0]
        self.assertEqual(item["status"], "unreachable")
        self.assertIsNone(item["path"])
        self.assertIsNone(item["cost"])
        self.assertEqual(item["expanded"], 1)
        self.assertEqual(result["expanded"], 1)

    def test_reserved_start_is_unreachable_without_spending_budget(self):
        # The duplicate second request starts on the first route's
        # frame-0 vertex: its search has no root, closes nothing and is
        # unreachable rather than budget-exhausted.
        result = plan_agents(
            3, 3, set(), [[(0, 0), (2, 0)], [(0, 0), (2, 0)]],
            max_expanded=100)
        self.assertEqual(result["status"], "unreachable")
        self.assertEqual(result["failed_index"], 1)
        self.assertEqual(result["results"][1]["status"], "unreachable")
        self.assertEqual(result["results"][1]["expanded"], 0)


class BudgetWaitTest(unittest.TestCase):
    FRAMES = [[], [(1, 0)], []]

    def test_waits_close_spatiotemporal_nodes_that_count(self):
        # The goal corridor is blocked at frame 1, so the route waits at
        # the start once: the wait node (0, 0, 1) is a real closing.
        result = plan_agents(
            3, 1, set(), [[(0, 0), (2, 0)]],
            dynamic_blocked=self.FRAMES, allow_wait=True,
            max_expanded=2, trace=True)
        self.assertEqual(result["status"], "budget_exhausted")
        item = result["results"][0]
        self.assertEqual(item["expanded"], 2)
        self.assertEqual(item["expanded_nodes"],
                         [(0, 0, 0), (0, 0, 1)])
        # The fourth closing is the goal itself: an exact finish.
        found = plan_agents(
            3, 1, set(), [[(0, 0), (2, 0)]],
            dynamic_blocked=self.FRAMES, allow_wait=True, max_expanded=4)
        self.assertEqual(found["status"], "found")
        self.assertEqual(found["results"][0]["path"],
                         [(0, 0), (0, 0), (1, 0), (2, 0)])
        self.assertEqual(found["results"][0]["cost"], 2)


class BudgetValidationTest(unittest.TestCase):
    REQUESTS = [[(0, 0), (2, 2)]]

    def test_type_errors(self):
        for value in [True, False, 1.5, "5", b"5", [5], (5,)]:
            with self.assertRaises(TypeError, msg=repr(value)):
                plan_agents(3, 3, set(), self.REQUESTS, max_expanded=value)

    def test_negative_value_error(self):
        with self.assertRaises(ValueError):
            plan_agents(3, 3, set(), self.REQUESTS, max_expanded=-1)

    def test_budget_is_checked_after_requests(self):
        with self.assertRaises(TypeError):
            plan_agents(3, 3, set(), "requests", max_expanded="bad")
        with self.assertRaises(TypeError):
            plan_agents(3, 3, set(), "requests", max_expanded=-1)

    def test_budget_is_checked_after_grid_and_endpoint_checks(self):
        with self.assertRaises(ValueError):
            plan_agents(3, 3, set(), [[(0, 0), (9, 9)]],
                        max_expanded="bad")
        with self.assertRaises(ValueError):
            plan_agents(0, 3, set(), self.REQUESTS, max_expanded="bad")

    def test_budget_is_checked_after_frames(self):
        with self.assertRaises(ValueError):
            plan_agents(3, 3, set(), [[(0, 0), (2, 2)]],
                        dynamic_blocked=[[(0, 0)]], max_expanded="bad")

    def test_budget_is_checked_after_cost_conflict(self):
        with self.assertRaises(ValueError):
            plan_agents(3, 3, set(), self.REQUESTS,
                        costs=[[1] * 3 for _ in range(3)],
                        dynamic_costs=[[[1] * 3 for _ in range(3)]],
                        max_expanded="bad")

    def test_budget_is_checked_after_reservations(self):
        with self.assertRaises(TypeError):
            plan_agents(3, 3, set(), self.REQUESTS,
                        reservations="paths", max_expanded="bad")
        with self.assertRaises(ValueError):
            plan_agents(3, 3, set(), self.REQUESTS,
                        reservations=[[]], max_expanded=-1)

    def test_invalid_budget_never_returns_partial_results(self):
        with self.assertRaises(ValueError):
            plan_agents(3, 2, set(),
                        [[(0, 0), (2, 0)], [(2, 0), (0, 0)]],
                        max_expanded=-1)


class BudgetReplayAndDeterminismTest(unittest.TestCase):
    def test_successful_prefix_replays_under_predecessor_routes(self):
        requests = [[(0, 0), (3, 0)], [(3, 0), (0, 0)], [(0, 1), (3, 1)]]
        full = plan_agents(4, 2, set(), requests)
        budgeted = plan_agents(4, 2, set(), requests,
                               max_expanded=full["expanded"])
        self.assertEqual(budgeted["status"], "found")
        planned = []
        for (start, goal), item in zip(requests, budgeted["results"]):
            outcome = replay(4, 2, set(), start, goal, item["path"],
                             reservations=planned)
            self.assertEqual(
                outcome,
                {"valid": True, "cost": item["cost"],
                 "steps": len(item["path"]) - 1})
            planned.append(tuple(item["path"]))

    def test_successful_prefix_replays_with_dynamic_frames_and_predecessors(
            self):
        requests = [[(0, 0), (2, 0)], [(0, 1), (2, 1)]]
        result = plan_agents(3, 2, set(), requests,
                             dynamic_blocked=FRAMES,
                             max_expanded=100)
        self.assertEqual(result["status"], "found")
        planned = []
        for (start, goal), item in zip(requests, result["results"]):
            # Replay under the same dynamic frames, with every earlier
            # agent's route supplied as a reservation: cost and steps
            # must match exactly.
            outcome = replay(3, 2, set(), start, goal, item["path"],
                             dynamic_blocked=FRAMES,
                             reservations=planned)
            self.assertTrue(outcome["valid"])
            self.assertEqual(outcome["cost"], item["cost"])
            self.assertEqual(outcome["steps"], len(item["path"]) - 1)
            planned.append(tuple(item["path"]))

    def test_reservation_permutations_and_duplicates_are_stable(self):
        requests = [[(0, 0), (2, 0)], [(2, 0), (0, 0)]]
        single = [[(0, 1), (1, 1), (2, 1)]]
        duplicated = [[(0, 1), (0, 1), (1, 1), (2, 1)],
                      [(0, 1), (1, 1), (2, 1)]]
        first = plan_agents(3, 2, set(), requests, reservations=single,
                            max_expanded=6)
        second = plan_agents(3, 2, set(), requests, reservations=duplicated,
                             max_expanded=6)
        self.assertEqual(first, second)

    def test_zero_budget_is_deterministic(self):
        requests = [[(0, 0), (2, 0)], [(2, 0), (0, 0)]]
        first = plan_agents(3, 2, set(), requests,
                            max_expanded=0, trace=True)
        second = plan_agents(3, 2, set(), requests,
                             max_expanded=0, trace=True)
        self.assertEqual(first, second)
        self.assertEqual(first["status"], "budget_exhausted")
        self.assertEqual(first["expanded"], 0)
        self.assertEqual(first["results"][0]["expanded_nodes"], [])


if __name__ == "__main__":
    unittest.main()
