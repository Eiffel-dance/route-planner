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


CORRIDOR = [[(0, 0), (2, 0)], [(2, 0), (0, 0)]]
CORRIDOR_ROUTE = [(0, 0), (1, 0), (2, 0)]
THREE_REQUESTS = [[(0, 0), (2, 0)], [(2, 0), (0, 0)],
                  [(0, 1), (2, 1)]]


class BudgetShapeTest(unittest.TestCase):
    def test_omitted_and_none_keep_legacy_item_keys(self):
        for kwargs in ({}, {"max_expanded": None}):
            result = plan_agents(3, 2, set(), CORRIDOR, **kwargs)
            self.assertEqual(result["status"], "found")
            self.assertNotIn("failed_index", result)
            for item in result["results"]:
                self.assertEqual(set(item.keys()),
                                 {"path", "cost", "expanded"})

    def test_found_items_carry_found_status_with_budget(self):
        result = plan_agents(3, 2, set(), CORRIDOR, max_expanded=10)
        self.assertEqual(result["status"], "found")
        self.assertEqual([item["status"] for item in result["results"]],
                         ["found", "found"])

    def test_budget_exhausted_partial_result_shape(self):
        result = plan_agents(3, 2, set(), THREE_REQUESTS, max_expanded=5)
        # Agent 0 closes 3, agent 1 spends 2 more without finishing; the
        # third request is never searched.
        self.assertEqual(list(result.keys()),
                         ["status", "failed_index", "results", "expanded"])
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(result["failed_index"], 1)
        self.assertEqual(len(result["results"]), 2)
        self.assertEqual(result["expanded"], 5)
        exhausted = result["results"][1]
        self.assertIsNone(exhausted["path"])
        self.assertIsNone(exhausted["cost"])
        self.assertEqual(exhausted["expanded"], 2)
        self.assertEqual(exhausted["status"], "budget_exhausted")

    def test_exact_budget_to_close_goal_is_found(self):
        # An open 3x3 grid closes 5 cells on the winning route.
        tight = plan_agents(3, 3, set(), [[(0, 0), (2, 2)]], max_expanded=5)
        self.assertEqual(tight["status"], "found")
        self.assertEqual(tight["results"][0]["status"], "found")
        self.assertEqual(tight["results"][0]["expanded"], 5)
        short = plan_agents(3, 3, set(), [[(0, 0), (2, 2)]], max_expanded=4)
        self.assertEqual(short["status"], "budget_exhausted")
        self.assertIsNone(short["results"][0]["path"])
        self.assertEqual(short["results"][0]["expanded"], 4)

    def test_zero_budget_stops_before_first_close(self):
        result = plan_agents(3, 2, set(), CORRIDOR, max_expanded=0)
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(result["failed_index"], 0)
        self.assertEqual(len(result["results"]), 1)
        self.assertEqual(result["results"][0],
                         {"path": None, "cost": None, "expanded": 0,
                          "status": "budget_exhausted"})
        self.assertEqual(result["expanded"], 0)

    def test_start_equals_goal_budget_boundary(self):
        # The single-point route needs exactly one close.
        zero = plan_agents(3, 3, set(), [[(1, 1), (1, 1)]],
                           max_expanded=0)["results"][0]
        self.assertEqual(zero,
                         {"path": None, "cost": None, "expanded": 0,
                          "status": "budget_exhausted"})
        one = plan_agents(3, 3, set(), [[(1, 1), (1, 1)]],
                          max_expanded=1)["results"][0]
        self.assertEqual(one,
                         {"path": [(1, 1)], "cost": 0, "expanded": 1,
                          "status": "found"})

    def test_remainder_of_zero_exhausts_next_agent_immediately(self):
        # Agent 0 spends the whole budget; agent 1 gets a zero remainder
        # and reports the exhaustion without closing a node.
        result = plan_agents(3, 2, set(), CORRIDOR, max_expanded=3)
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(result["failed_index"], 1)
        self.assertEqual(result["results"][0]["path"], CORRIDOR_ROUTE)
        self.assertEqual(result["results"][1],
                         {"path": None, "cost": None, "expanded": 0,
                          "status": "budget_exhausted"})
        self.assertEqual(result["expanded"], 3)

    def test_budget_accumulates_across_agents(self):
        result = plan_agents(3, 2, set(), CORRIDOR, max_expanded=4)
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual([item["expanded"] for item in result["results"]],
                         [3, 1])
        self.assertEqual(
            result["expanded"],
            sum(item["expanded"] for item in result["results"]))

    def test_enough_budget_reproduces_unlimited_routes(self):
        unlimited = plan_agents(3, 2, set(), CORRIDOR)
        bounded = plan_agents(3, 2, set(), CORRIDOR, max_expanded=100)
        self.assertEqual(bounded["status"], "found")
        self.assertEqual(bounded["expanded"], unlimited["expanded"])
        for bounded_item, unlimited_item in zip(
                bounded["results"], unlimited["results"]):
            for key in ("path", "cost", "expanded"):
                self.assertEqual(bounded_item[key], unlimited_item[key])

    def test_natural_exhaustion_stays_unreachable_under_budget(self):
        blocked = {(1, 2), (2, 1)}
        requests = [[(0, 0), (1, 0)], [(0, 2), (2, 2)]]
        result = plan_agents(3, 3, blocked, requests, max_expanded=100)
        self.assertEqual(result["status"], "unreachable")
        self.assertEqual(result["failed_index"], 1)
        self.assertEqual(result["results"][0]["status"], "found")
        failed = result["results"][1]
        self.assertEqual(failed["status"], "unreachable")
        self.assertIsNone(failed["path"])
        self.assertIsNone(failed["cost"])
        self.assertEqual(failed["expanded"], 4)
        self.assertEqual(result["expanded"], 6)

    def test_exhaustion_in_dynamic_mode_records_space_time_closes(self):
        # Agent 0 uses the static grid; agent 1's search is time-expanded
        # by the reserved route, so its two closes are (x, y, t) triples.
        result = plan_agents(3, 2, set(), CORRIDOR, max_expanded=5,
                             trace=True)
        self.assertEqual(result["status"], "budget_exhausted")
        first, second = result["results"]
        self.assertEqual(first["expanded_nodes"],
                         [(0, 0), (1, 0), (2, 0)])
        self.assertEqual(second["expanded"], 2)
        self.assertEqual(second["expanded_nodes"],
                         [(2, 0, 0), (2, 1, 1)])
        for item in result["results"]:
            self.assertEqual(item["expanded"],
                             len(item["expanded_nodes"]))

    def test_trace_only_records_actual_closes(self):
        result = plan_agents(3, 2, set(), THREE_REQUESTS, max_expanded=3,
                             trace=True)
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(len(result["results"]), 2)
        self.assertEqual(result["results"][1]["expanded"], 0)
        self.assertEqual(result["results"][1]["expanded_nodes"], [])

    def test_json_serializable_under_budget(self):
        for status_call in (
                lambda: plan_agents(3, 2, set(), CORRIDOR, max_expanded=5,
                                    trace=True),
                lambda: plan_agents(3, 2, set(), CORRIDOR, max_expanded=10,
                                    trace=True)):
            result = status_call()
            self.assertEqual(json.loads(json.dumps(result)),
                             json.loads(json.dumps(result)))

    def test_waits_count_toward_the_shared_budget(self):
        # With waiting allowed the agent survives frame 1 by staying:
        # three closes do not finish, the fourth (on (1, 0)) does.
        frames = [[], [(1, 0)], []]
        short = plan_agents(3, 1, set(), [[(0, 0), (2, 0)]],
                            dynamic_blocked=frames, allow_wait=True,
                            max_expanded=3)
        self.assertEqual(short["status"], "budget_exhausted")
        self.assertEqual(short["results"][0]["expanded"], 3)
        tight = plan_agents(3, 1, set(), [[(0, 0), (2, 0)]],
                            dynamic_blocked=frames, allow_wait=True,
                            max_expanded=4)
        self.assertEqual(tight["status"], "found")
        self.assertEqual(tight["results"][0]["path"],
                         [(0, 0), (0, 0), (1, 0), (2, 0)])
        self.assertEqual(tight["results"][0]["expanded"], 4)

    def test_agent_search_matches_plan_with_planned_reservations(self):
        # Agent 1 must be exactly a ``plan`` call over agent 0's route as
        # a reservation, bounded by the *remaining* budget.
        result = plan_agents(3, 2, set(), CORRIDOR, max_expanded=5)
        expected = plan(3, 2, set(), (2, 0), (0, 0),
                        reservations=[CORRIDOR_ROUTE], max_expanded=2)
        self.assertEqual(result["results"][1], expected)
        full = plan_agents(3, 2, set(), CORRIDOR, max_expanded=10)
        full_expected = plan(3, 2, set(), (2, 0), (0, 0),
                             reservations=[CORRIDOR_ROUTE], max_expanded=7)
        self.assertEqual(full["results"][1], full_expected)

    def test_found_routes_replay_identically(self):
        result = plan_agents(3, 2, set(), CORRIDOR, max_expanded=10)
        for (start, goal), item in zip(CORRIDOR, result["results"]):
            outcome = replay(3, 2, set(), start, goal, item["path"])
            self.assertEqual(
                outcome,
                {"valid": True, "cost": item["cost"],
                 "steps": len(item["path"]) - 1})

    def test_found_routes_replay_with_shared_constraints(self):
        frames = FRAMES
        result = plan_agents(3, 2, set(),
                             [[(0, 0), (2, 0)], [(0, 1), (2, 1)]],
                             dynamic_blocked=frames, allow_wait=True,
                             max_expanded=50)
        self.assertEqual(result["status"], "found")
        planned = []
        for (start, goal), item in zip(
                [[(0, 0), (2, 0)], [(0, 1), (2, 1)]], result["results"]):
            outcome = replay(3, 2, set(), start, goal, item["path"],
                             dynamic_blocked=frames, allow_wait=True,
                             reservations=planned)
            self.assertTrue(outcome["valid"])
            self.assertEqual(outcome["cost"], item["cost"])
            self.assertEqual(outcome["steps"], len(item["path"]) - 1)
            planned.append(item["path"])

    def test_reservation_order_and_duplicates_do_not_change_result(self):
        requests = [[(0, 0), (3, 0)], [(3, 1), (0, 1)]]
        first = [[(0, 1), (1, 1), (2, 1), (3, 1)], [(3, 0), (2, 0)]]
        shuffled = [[(3, 0), (2, 0)], [(3, 0), (2, 0)],
                    [(0, 1), (1, 1), (2, 1), (3, 1)]]
        one = plan_agents(4, 2, set(), requests, reservations=first,
                          max_expanded=6)
        two = plan_agents(4, 2, set(), requests, reservations=shuffled,
                          max_expanded=6)
        self.assertEqual(one, two)
        for budget in (0, 6, 20):
            self.assertEqual(
                plan_agents(4, 2, set(), requests, reservations=first,
                            max_expanded=budget),
                plan_agents(4, 2, set(), requests, reservations=shuffled,
                            max_expanded=budget))

    def test_frame_order_and_duplicates_do_not_change_result(self):
        requests = [[(0, 0), (3, 0)], [(0, 1), (3, 1)]]
        frames_a = [[(1, 0), (2, 1)], []]
        frames_b = [[(2, 1), (1, 0), (1, 0)], []]
        for budget in (0, 8, 50):
            self.assertEqual(
                plan_agents(4, 2, set(), requests,
                            dynamic_blocked=frames_a, max_expanded=budget),
                plan_agents(4, 2, set(), requests,
                            dynamic_blocked=frames_b, max_expanded=budget))


class BudgetValidationTest(unittest.TestCase):
    def test_budget_type_errors(self):
        requests = [[(0, 0), (2, 2)]]
        for value in ("3", 1.5, True, False, [3], (3,)):
            with self.assertRaises(TypeError, msg=repr(value)):
                plan_agents(3, 3, set(), requests, max_expanded=value)

    def test_budget_negative_value_error(self):
        with self.assertRaises(ValueError):
            plan_agents(3, 3, set(), [[(0, 0), (2, 2)]], max_expanded=-1)

    def test_zero_budget_is_accepted(self):
        result = plan_agents(3, 3, set(), [[(0, 0), (2, 2)]],
                             max_expanded=0)
        self.assertEqual(result["status"], "budget_exhausted")

    def test_budget_checked_after_every_existing_check(self):
        requests = [[(0, 0), (2, 2)]]
        # An out-of-bounds request fails before the budget is inspected.
        with self.assertRaises(ValueError):
            plan_agents(3, 3, set(), [[(0, 0), (9, 9)]], max_expanded="x")
        # A bad costs matrix fails before the budget type check.
        with self.assertRaises(ValueError):
            plan_agents(3, 3, set(), requests,
                        costs=[[1, 1], [1, 1]], max_expanded="x")
        with self.assertRaises(TypeError):
            plan_agents(3, 3, set(), requests, costs="costs",
                        max_expanded=-1)
        # A frame-0-blocked start fails before a negative budget.
        with self.assertRaises(ValueError):
            plan_agents(3, 3, set(), requests,
                        dynamic_blocked=[[(0, 0)]], max_expanded=-1)
        # An invalid reservation fails before an invalid budget.
        with self.assertRaises(ValueError):
            plan_agents(3, 3, set(), requests, reservations=[[]],
                        max_expanded="x")
        with self.assertRaises(TypeError):
            plan_agents(3, 3, set(), requests, reservations="paths",
                        max_expanded=-1)
        # An earlier-flag type error also precedes the budget check.
        with self.assertRaises(TypeError):
            plan_agents(3, 3, set(), requests, allow_wait=1,
                        max_expanded="x")

    def test_invalid_budget_never_returns_partial_results(self):
        # Even with several valid requests, the validation failure means
        # no result object at all.
        with self.assertRaises(ValueError):
            plan_agents(3, 2, set(), CORRIDOR, max_expanded=-1)


if __name__ == "__main__":
    unittest.main()
