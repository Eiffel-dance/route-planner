import json
import unittest

from app import plan, plan_agents, replay


BLOCKED = {(2, 0), (2, 1), (2, 2)}
FRAMES = [
    [(0, 1), (0, 2)],
    [(0, 1), (1, 2), (2, 0), (2, 2)],
    [(2, 1), (2, 2)],
    [(0, 0), (1, 2), (2, 1), (2, 2)],
    [],
    [(0, 1)],
]
COSTS = [[1, 10, 1], [1, 10, 1], [1, 1, 1]]
DYNAMIC_COSTS = [
    [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
    [[1, 1, 1], [1, 1, 1], [9, 9, 9]],
]
# A three-agent batch that stays feasible in every mode. The third
# agent's direct route is blocked by the second agent's timing, so it
# detours through (0, 2) and (0, 1).
REQUESTS = [[(0, 0), (1, 0)], [(1, 0), (1, 2)], [(1, 2), (1, 1)]]


def conflict(path_a, path_b):
    # The first vertex conflict or reverse-edge traversal between the
    # two timed routes; each route's final coordinate persists.
    horizon = max(len(path_a), len(path_b))
    for t in range(horizon):
        a = path_a[t] if t < len(path_a) else path_a[-1]
        b = path_b[t] if t < len(path_b) else path_b[-1]
        if a == b:
            return ("vertex", t)
        if t + 1 < horizon:
            a2 = path_a[t + 1] if t + 1 < len(path_a) else path_a[-1]
            b2 = path_b[t + 1] if t + 1 < len(path_b) else path_b[-1]
            if a != a2 and a == b2 and a2 == b:
                return ("edge", t)
    return None


def assert_conflict_free(testcase, paths):
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            testcase.assertIsNone(
                conflict(paths[i], paths[j]),
                msg=f"routes {i} and {j} conflict",
            )


class AgentsResultTest(unittest.TestCase):
    def test_result_shape(self):
        result = plan_agents(3, 3, BLOCKED, REQUESTS)
        self.assertEqual(list(result.keys()),
                         ["status", "results", "expanded"])
        self.assertEqual(result["status"], "found")
        self.assertEqual(len(result["results"]), len(REQUESTS))
        for entry in result["results"]:
            self.assertEqual(list(entry.keys()),
                             ["path", "cost", "expanded"])
        self.assertEqual(
            result["expanded"],
            sum(entry["expanded"] for entry in result["results"]),
        )

    def test_first_agent_matches_plan(self):
        result = plan_agents(3, 3, BLOCKED, REQUESTS)
        start, goal = REQUESTS[0]
        self.assertEqual(result["results"][0],
                         plan(3, 3, BLOCKED, start, goal))

    def test_agents_match_plan_with_accumulated_reservations(self):
        for kwargs in ({}, {"costs": COSTS}, {"dynamic_blocked": FRAMES},
                       {"dynamic_blocked": FRAMES, "allow_wait": True},
                       {"dynamic_costs": DYNAMIC_COSTS}):
            result = plan_agents(3, 3, BLOCKED, REQUESTS, **kwargs)
            self.assertEqual(result["status"], "found")
            reservations = []
            for request, entry in zip(REQUESTS, result["results"]):
                start, goal = request
                expected = plan(3, 3, BLOCKED, start, goal,
                                reservations=reservations or None,
                                **kwargs)
                self.assertEqual(entry, expected,
                                 msg=f"kwargs={kwargs!r} "
                                     f"request={request!r}")
                reservations.append(entry["path"])

    def test_later_agents_detour_around_earlier_routes(self):
        result = plan_agents(3, 3, BLOCKED, REQUESTS)
        self.assertEqual(result["status"], "found")
        # The direct route [(1, 2), (1, 1)] would collide with the
        # second agent at frame 1, so the third agent detours.
        self.assertEqual(result["results"][2]["path"],
                         [(1, 2), (0, 2), (0, 1), (1, 1)])

    def test_routes_are_conflict_free(self):
        for kwargs in ({}, {"costs": COSTS}, {"dynamic_blocked": FRAMES},
                       {"dynamic_blocked": FRAMES, "allow_wait": True},
                       {"dynamic_costs": DYNAMIC_COSTS}):
            result = plan_agents(3, 3, BLOCKED, REQUESTS, **kwargs)
            self.assertEqual(result["status"], "found")
            paths = [entry["path"] for entry in result["results"]]
            assert_conflict_free(self, paths)

    def test_crossing_agents_avoid_each_other(self):
        requests = [[(0, 1), (2, 1)], [(2, 1), (0, 1)]]
        result = plan_agents(3, 3, set(), requests)
        self.assertEqual(result["status"], "found")
        first, second = result["results"]
        self.assertEqual(first["path"], [(0, 1), (1, 1), (2, 1)])
        self.assertNotEqual(second["path"], [(2, 1), (1, 1), (0, 1)])
        assert_conflict_free(self, [first["path"], second["path"]])

    def test_goal_occupancy_persists(self):
        # The third agent's goal is the first agent's goal, which stays
        # reserved for every frame after the first agent arrives.
        requests = [[(0, 0), (2, 0)], [(2, 2), (0, 0)], [(1, 2), (2, 0)]]
        result = plan_agents(3, 3, set(), requests)
        self.assertEqual(result["status"], "unreachable")
        self.assertEqual(result["failed_index"], 2)
        assert_conflict_free(
            self, [entry["path"] for entry in result["results"][:2]])

    def test_input_order_fixes_priority(self):
        forward = plan_agents(3, 3, set(), [[(0, 1), (2, 1)],
                                            [(2, 1), (0, 1)]])
        swapped = plan_agents(3, 3, set(), [[(2, 1), (0, 1)],
                                            [(0, 1), (2, 1)]])
        # Whoever comes first gets the direct route; the second agent
        # detours around it.
        self.assertEqual(forward["results"][0]["path"],
                         [(0, 1), (1, 1), (2, 1)])
        self.assertEqual(swapped["results"][0]["path"],
                         [(2, 1), (1, 1), (0, 1)])
        self.assertNotEqual(forward["results"][1]["path"],
                            [(2, 1), (1, 1), (0, 1)])
        self.assertNotEqual(swapped["results"][1]["path"],
                            [(0, 1), (1, 1), (2, 1)])
        for result in (forward, swapped):
            self.assertEqual(result["status"], "found")
            assert_conflict_free(
                self, [entry["path"] for entry in result["results"]])

    def test_duplicates_are_kept(self):
        # A duplicate request is planned again under the first route's
        # reservation; the goal persists, so the copy cannot stop there.
        requests = [[(0, 0), (1, 0)], [(0, 0), (1, 0)]]
        result = plan_agents(3, 3, set(), requests)
        self.assertEqual(result["status"], "unreachable")
        self.assertEqual(result["failed_index"], 1)
        self.assertEqual(len(result["results"]), 2)
        self.assertEqual(result["results"][0]["path"], [(0, 0), (1, 0)])

    def test_start_equals_goal(self):
        result = plan_agents(3, 3, BLOCKED, [[(1, 1), (1, 1)]])
        self.assertEqual(result["status"], "found")
        self.assertEqual(result["results"][0],
                         {"path": [(1, 1)], "cost": 0, "expanded": 1})
        self.assertEqual(result["expanded"], 1)

    def test_unreachable_stops_immediately(self):
        blocked = {(1, 0), (0, 1)}
        requests = [[(2, 2), (2, 0)], [(0, 0), (2, 2)], [(2, 2), (0, 2)]]
        result = plan_agents(3, 3, blocked, requests)
        self.assertEqual(result["status"], "unreachable")
        self.assertEqual(result["failed_index"], 1)
        # Only the successful prefix and the failed agent are reported;
        # the third request is never searched.
        self.assertEqual(len(result["results"]), 2)
        failed = result["results"][1]
        self.assertIsNone(failed["path"])
        self.assertIsNone(failed["cost"])
        self.assertEqual(failed["expanded"], 1)
        self.assertEqual(
            result["expanded"],
            sum(entry["expanded"] for entry in result["results"]),
        )

    def test_unreachable_first_agent(self):
        blocked = {(1, 0), (0, 1)}
        result = plan_agents(3, 3, blocked, [[(0, 0), (2, 2)],
                                             [(2, 2), (0, 0)]])
        self.assertEqual(result["status"], "unreachable")
        self.assertEqual(result["failed_index"], 0)
        self.assertEqual(len(result["results"]), 1)
        self.assertIsNone(result["results"][0]["path"])
        self.assertEqual(result["expanded"], result["results"][0]["expanded"])

    def test_trace(self):
        result = plan_agents(3, 3, BLOCKED, REQUESTS, trace=True)
        self.assertEqual(result["status"], "found")
        for entry in result["results"]:
            self.assertIn("expanded_nodes", entry)
            self.assertEqual(entry["expanded"], len(entry["expanded_nodes"]))

    def test_trace_records_triples_once_reserved(self):
        result = plan_agents(3, 3, set(), [[(0, 1), (2, 1)],
                                           [(2, 1), (0, 1)]], trace=True)
        first, second = result["results"]
        # The first agent runs the static search (coordinate pairs);
        # the second runs under reservations (space-time triples).
        self.assertEqual(first["expanded_nodes"][0], (0, 1))
        self.assertEqual(second["expanded_nodes"][0], (2, 1, 0))

    def test_external_reservations(self):
        reservations = [[(0, 1), (0, 0)]]
        result = plan_agents(3, 3, BLOCKED, REQUESTS,
                             reservations=reservations)
        self.assertEqual(result["status"], "found")
        planned = list(reservations)
        for request, entry in zip(REQUESTS, result["results"]):
            start, goal = request
            expected = plan(3, 3, BLOCKED, start, goal,
                            reservations=planned)
            self.assertEqual(entry, expected, msg=f"request={request!r}")
            planned.append(entry["path"])

    def test_success_paths_replay(self):
        for kwargs in ({}, {"costs": COSTS}, {"dynamic_blocked": FRAMES},
                       {"dynamic_blocked": FRAMES, "allow_wait": True},
                       {"dynamic_costs": DYNAMIC_COSTS}):
            result = plan_agents(3, 3, BLOCKED, REQUESTS, **kwargs)
            self.assertEqual(result["status"], "found")
            for request, entry in zip(REQUESTS, result["results"]):
                start, goal = request
                self.assertEqual(entry["path"][-1], goal)
                check = replay(3, 3, BLOCKED, start, goal,
                               entry["path"], **kwargs)
                self.assertEqual(
                    check, {"valid": True, "cost": entry["cost"],
                            "steps": len(entry["path"]) - 1},
                    msg=f"kwargs={kwargs!r} request={request!r}",
                )

    def test_results_are_json_serializable(self):
        result = plan_agents(3, 3, BLOCKED, REQUESTS, trace=True,
                             dynamic_blocked=FRAMES, allow_wait=True)
        json.dumps(result)


class AgentsValidationTest(unittest.TestCase):
    def test_requests_type_errors(self):
        for bad in (None, "x", b"x", 5, 1.5, {(0, 0), (1, 1)},
                    {(0, 0): (1, 1)}):
            with self.assertRaises(TypeError, msg=f"requests={bad!r}"):
                plan_agents(3, 3, BLOCKED, bad)

    def test_empty_requests_is_value_error(self):
        for empty in ([], ()):
            with self.assertRaises(ValueError, msg=f"requests={empty!r}"):
                plan_agents(3, 3, BLOCKED, empty)

    def test_element_shape_type_errors(self):
        for bad in ([None], ["x"], [b"x"], [5], [[(0, 0)]],
                    [[(0, 0), (1, 1), (2, 2)]],
                    [[(0, 0), (1, 1)], "x"]):
            with self.assertRaises(TypeError, msg=f"requests={bad!r}"):
                plan_agents(3, 3, BLOCKED, bad)

    def test_coordinate_type_errors(self):
        for bad in ([[(0.0, 0), (1, 1)]], [[(0, 0), (1, True)]],
                    [[(0, 0, 0), (1, 1)]], [[(0, 0), "x"]]):
            with self.assertRaises(TypeError, msg=f"requests={bad!r}"):
                plan_agents(3, 3, BLOCKED, bad)

    def test_out_of_bounds_is_value_error(self):
        for bad in ([[(3, 0), (1, 1)]], [[(0, 0), (-1, 1)]],
                    [[(0, 0), (1, 1)], [(0, 3), (1, 1)]]):
            with self.assertRaises(ValueError, msg=f"requests={bad!r}"):
                plan_agents(3, 3, BLOCKED, bad)

    def test_endpoint_on_obstacle_is_value_error(self):
        with self.assertRaises(ValueError):
            plan_agents(3, 3, BLOCKED, [[(2, 0), (1, 1)]])
        with self.assertRaises(ValueError):
            plan_agents(3, 3, BLOCKED, [[(0, 0), (2, 1)]])

    def test_frame_zero_blocked_start_is_value_error(self):
        with self.assertRaises(ValueError):
            plan_agents(3, 3, BLOCKED, [[(0, 1), (2, 2)]],
                        dynamic_blocked=FRAMES)
        # The batch is validated as a whole: a later bad request fails
        # the call even though the first request would succeed.
        with self.assertRaises(ValueError):
            plan_agents(3, 3, BLOCKED, [[(0, 0), (2, 2)], [(0, 1), (2, 2)]],
                        dynamic_blocked=FRAMES)

    def test_shared_validation_order(self):
        # width ValueError precedes the requests TypeError.
        with self.assertRaises(ValueError):
            plan_agents(0, 3, BLOCKED, "x")
        # requests TypeError precedes the blocked TypeError.
        with self.assertRaises(TypeError):
            plan_agents(3, 3, "x", "x")
        # requests ValueError precedes the costs ValueError.
        with self.assertRaises(ValueError):
            plan_agents(3, 3, BLOCKED, [], costs=[[1]])
        # blocked TypeError precedes the costs ValueError.
        with self.assertRaises(TypeError):
            plan_agents(3, 3, "x", REQUESTS, costs=[[1]])
        # costs ValueError precedes the trace TypeError.
        with self.assertRaises(ValueError):
            plan_agents(3, 3, BLOCKED, REQUESTS, costs=[[1]], trace=1)
        # trace TypeError precedes the dynamic_blocked TypeError.
        with self.assertRaises(TypeError):
            plan_agents(3, 3, BLOCKED, REQUESTS, trace=1,
                        dynamic_blocked="x")
        # dynamic_blocked TypeError precedes the allow_wait TypeError.
        with self.assertRaises(TypeError):
            plan_agents(3, 3, BLOCKED, REQUESTS, dynamic_blocked="x",
                        allow_wait=1)
        # allow_wait TypeError precedes the dynamic_costs TypeError.
        with self.assertRaises(TypeError):
            plan_agents(3, 3, BLOCKED, REQUESTS, allow_wait=1,
                        dynamic_costs="x")
        # dynamic_costs TypeError precedes the reservations TypeError.
        with self.assertRaises(TypeError):
            plan_agents(3, 3, BLOCKED, REQUESTS, dynamic_costs="x",
                        reservations="x")

    def test_shared_option_type_errors(self):
        with self.assertRaises(TypeError):
            plan_agents(3, 3, BLOCKED, REQUESTS, trace=1)
        with self.assertRaises(TypeError):
            plan_agents(3, 3, BLOCKED, REQUESTS, allow_wait=1)
        with self.assertRaises(TypeError):
            plan_agents(3, 3, BLOCKED, REQUESTS, dynamic_costs="x")
        with self.assertRaises(TypeError):
            plan_agents(3, 3, BLOCKED, REQUESTS, reservations="x")

    def test_costs_and_dynamic_costs_is_value_error(self):
        with self.assertRaises(ValueError):
            plan_agents(3, 3, BLOCKED, REQUESTS, costs=COSTS,
                        dynamic_costs=DYNAMIC_COSTS)

    def test_reservations_value_errors(self):
        with self.assertRaises(ValueError):
            plan_agents(3, 3, BLOCKED, REQUESTS, reservations=[[]])
        with self.assertRaises(ValueError):
            plan_agents(3, 3, BLOCKED, REQUESTS,
                        reservations=[[(9, 9)]])
        with self.assertRaises(ValueError):
            plan_agents(3, 3, BLOCKED, REQUESTS,
                        reservations=[[(2, 0)]])

    def test_no_partial_results_on_any_error(self):
        with self.assertRaises(ValueError):
            plan_agents(3, 3, BLOCKED, [[(0, 0), (1, 2)], [(9, 9), (0, 0)]])


if __name__ == "__main__":
    unittest.main()
