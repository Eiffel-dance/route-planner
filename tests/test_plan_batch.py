import json
import unittest

from app import plan, plan_batch, replay, resume


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
# Endpoints avoid the static obstacle column (x == 2) and the cells
# blocked at dynamic frame 0 ((0, 1) and (0, 2)).
REQUESTS = [[(0, 0), (1, 2)], [(1, 0), (0, 0)], [(0, 0), (0, 0)]]


def without_checkpoint(result):
    return {key: value for key, value in result.items()
            if key != "checkpoint"}


class BatchResultTest(unittest.TestCase):
    def test_result_shape(self):
        result = plan_batch(3, 3, BLOCKED, REQUESTS)
        self.assertEqual(list(result.keys()), ["results"])
        self.assertEqual(len(result["results"]), len(REQUESTS))

    def test_matches_individual_plan_calls(self):
        batch = plan_batch(3, 3, BLOCKED, REQUESTS)
        for request, result in zip(REQUESTS, batch["results"]):
            start, goal = request
            self.assertEqual(
                result, plan(3, 3, BLOCKED, start, goal),
                msg=f"request={request!r}",
            )

    def test_order_and_duplicates_preserved(self):
        requests = [[(0, 0), (1, 2)], [(1, 0), (0, 0)], [(0, 0), (1, 2)]]
        batch = plan_batch(3, 3, BLOCKED, requests)
        self.assertEqual(batch["results"][0], batch["results"][2])
        self.assertNotEqual(batch["results"][0], batch["results"][1])
        reversed_batch = plan_batch(3, 3, BLOCKED, list(reversed(requests)))
        self.assertEqual(reversed_batch["results"][0], batch["results"][2])
        self.assertEqual(reversed_batch["results"][2], batch["results"][0])

    def test_request_permutation_keeps_per_query_results(self):
        forward = plan_batch(3, 3, BLOCKED, REQUESTS)
        permuted = plan_batch(3, 3, BLOCKED,
                              [REQUESTS[2], REQUESTS[0], REQUESTS[1]])
        self.assertEqual(permuted["results"][0], forward["results"][2])
        self.assertEqual(permuted["results"][1], forward["results"][0])
        self.assertEqual(permuted["results"][2], forward["results"][1])

    def test_obstacle_permutation_does_not_change_results(self):
        as_set = plan_batch(3, 3, BLOCKED, REQUESTS)
        as_list = plan_batch(3, 3, sorted(BLOCKED, reverse=True), REQUESTS)
        duplicated = plan_batch(3, 3, list(BLOCKED) * 2, REQUESTS)
        self.assertEqual(as_set, as_list)
        self.assertEqual(as_set, duplicated)

    def test_frame_coordinate_permutation_does_not_change_results(self):
        frames = [list(reversed(frame)) for frame in FRAMES]
        base = plan_batch(3, 3, BLOCKED, REQUESTS, dynamic_blocked=FRAMES)
        shuffled = plan_batch(3, 3, BLOCKED, REQUESTS,
                              dynamic_blocked=frames)
        self.assertEqual(base, shuffled)

    def test_start_equals_goal(self):
        batch = plan_batch(3, 3, BLOCKED, [[(1, 1), (1, 1)]])
        self.assertEqual(batch["results"][0],
                         {"path": [(1, 1)], "cost": 0, "expanded": 1})

    def test_unreachable(self):
        blocked = {(1, 0), (0, 1)}
        batch = plan_batch(3, 3, blocked, [[(0, 0), (2, 2)]])
        self.assertEqual(batch["results"][0],
                         {"path": None, "cost": None, "expanded": 1})

    def test_costs(self):
        batch = plan_batch(3, 3, set(), REQUESTS[:2], costs=COSTS)
        for request, result in zip(REQUESTS[:2], batch["results"]):
            start, goal = request
            self.assertEqual(
                result, plan(3, 3, set(), start, goal, costs=COSTS))

    def test_trace(self):
        batch = plan_batch(3, 3, BLOCKED, REQUESTS, trace=True)
        for request, result in zip(REQUESTS, batch["results"]):
            start, goal = request
            expected = plan(3, 3, BLOCKED, start, goal, trace=True)
            self.assertEqual(result, expected)
            self.assertEqual(result["expanded"],
                             len(result["expanded_nodes"]))

    def test_dynamic(self):
        batch = plan_batch(3, 3, BLOCKED, REQUESTS, dynamic_blocked=FRAMES)
        for request, result in zip(REQUESTS, batch["results"]):
            start, goal = request
            self.assertEqual(
                result,
                plan(3, 3, BLOCKED, start, goal, dynamic_blocked=FRAMES))

    def test_budget_applies_per_query(self):
        batch = plan_batch(3, 3, BLOCKED, REQUESTS, max_expanded=2)
        for request, result in zip(REQUESTS, batch["results"]):
            start, goal = request
            expected = plan(3, 3, BLOCKED, start, goal, max_expanded=2)
            self.assertEqual(result, expected)
            self.assertIn("status", result)
            self.assertLessEqual(result["expanded"], 2)

    def test_max_cost_applies_per_query(self):
        batch = plan_batch(3, 3, set(), REQUESTS[:2], costs=COSTS,
                           max_cost=6)
        for request, result in zip(REQUESTS[:2], batch["results"]):
            start, goal = request
            expected = plan(3, 3, set(), start, goal, costs=COSTS,
                            max_cost=6)
            self.assertEqual(result, expected)
            self.assertIn("status", result)

    def test_success_paths_replay(self):
        for kwargs in ({}, {"costs": COSTS}, {"dynamic_blocked": FRAMES}):
            batch = plan_batch(3, 3, BLOCKED, REQUESTS, **kwargs)
            for request, result in zip(REQUESTS, batch["results"]):
                if result["path"] is None:
                    continue
                start, goal = request
                self.assertEqual(result["path"][-1], goal)
                check = replay(3, 3, BLOCKED, start, goal,
                               result["path"], **kwargs)
                self.assertEqual(
                    check, {"valid": True, "cost": result["cost"],
                            "steps": len(result["path"]) - 1})

    def test_snapshot_checkpoint_resumes(self):
        batch = plan_batch(3, 3, BLOCKED, REQUESTS, max_expanded=1,
                           snapshot=True)
        for request, result in zip(REQUESTS, batch["results"]):
            start, goal = request
            expected = plan(3, 3, BLOCKED, start, goal, max_expanded=1,
                            snapshot=True)
            self.assertEqual(without_checkpoint(result),
                             without_checkpoint(expected))
            if result.get("status") != "budget_exhausted":
                self.assertNotIn("checkpoint", result)
                continue
            json.dumps(result["checkpoint"])
            final = resume(result["checkpoint"], max_expanded=100)
            unlimited = plan(3, 3, BLOCKED, start, goal, max_expanded=100,
                             trace=True)
            self.assertEqual(final["path"], unlimited["path"])
            self.assertEqual(final["cost"], unlimited["cost"])
            self.assertEqual(final["expanded"], unlimited["expanded"])

    def test_results_are_json_serializable(self):
        batch = plan_batch(3, 3, BLOCKED, REQUESTS, trace=True,
                           dynamic_blocked=FRAMES, max_expanded=1,
                           snapshot=True)
        json.dumps(batch)


class BatchValidationTest(unittest.TestCase):
    def test_requests_type_errors(self):
        for bad in (None, "x", b"x", 5, 1.5, {(0, 0), (1, 1)},
                    {(0, 0): (1, 1)}):
            with self.assertRaises(TypeError, msg=f"requests={bad!r}"):
                plan_batch(3, 3, BLOCKED, bad)

    def test_empty_requests_is_value_error(self):
        for empty in ([], ()):
            with self.assertRaises(ValueError, msg=f"requests={empty!r}"):
                plan_batch(3, 3, BLOCKED, empty)

    def test_element_shape_type_errors(self):
        for bad in ([None], ["x"], [b"x"], [5], [[(0, 0)]],
                    [[(0, 0), (1, 1), (2, 2)]],
                    [[(0, 0), (1, 1)], "x"]):
            with self.assertRaises(TypeError, msg=f"requests={bad!r}"):
                plan_batch(3, 3, BLOCKED, bad)

    def test_coordinate_type_errors(self):
        for bad in ([[(0.0, 0), (1, 1)]], [[(0, 0), (1, True)]],
                    [[(0, 0, 0), (1, 1)]], [[(0, 0), "x"]]):
            with self.assertRaises(TypeError, msg=f"requests={bad!r}"):
                plan_batch(3, 3, BLOCKED, bad)

    def test_out_of_bounds_is_value_error(self):
        for bad in ([[(3, 0), (1, 1)]], [[(0, 0), (-1, 1)]],
                    [[(0, 0), (1, 1)], [(0, 3), (1, 1)]]):
            with self.assertRaises(ValueError, msg=f"requests={bad!r}"):
                plan_batch(3, 3, BLOCKED, bad)

    def test_endpoint_on_obstacle_is_value_error(self):
        with self.assertRaises(ValueError):
            plan_batch(3, 3, BLOCKED, [[(2, 0), (1, 1)]])
        with self.assertRaises(ValueError):
            plan_batch(3, 3, BLOCKED, [[(0, 0), (2, 1)]])

    def test_frame_zero_blocked_start_is_value_error(self):
        with self.assertRaises(ValueError):
            plan_batch(3, 3, BLOCKED, [[(0, 1), (2, 2)]],
                       dynamic_blocked=FRAMES)
        # Only the blocked request fails; the batch is validated as a
        # whole, so no partial results are returned.
        with self.assertRaises(ValueError):
            plan_batch(3, 3, BLOCKED, [[(0, 0), (2, 2)], [(0, 1), (2, 2)]],
                       dynamic_blocked=FRAMES)

    def test_shared_validation_order(self):
        # width ValueError precedes the requests TypeError.
        with self.assertRaises(ValueError):
            plan_batch(0, 3, BLOCKED, "x")
        # requests TypeError precedes the blocked TypeError.
        with self.assertRaises(TypeError):
            plan_batch(3, 3, "x", "x")
        # requests ValueError precedes the costs ValueError.
        with self.assertRaises(ValueError):
            plan_batch(3, 3, BLOCKED, [], costs=[[1]])
        # blocked TypeError precedes the costs ValueError.
        with self.assertRaises(TypeError):
            plan_batch(3, 3, "x", REQUESTS, costs=[[1]])
        # costs ValueError precedes the trace TypeError.
        with self.assertRaises(ValueError):
            plan_batch(3, 3, BLOCKED, REQUESTS, costs=[[1]], trace=1)
        # trace TypeError precedes the dynamic_blocked TypeError.
        with self.assertRaises(TypeError):
            plan_batch(3, 3, BLOCKED, REQUESTS, trace=1,
                       dynamic_blocked="x")
        # dynamic_blocked TypeError precedes the budget TypeError.
        with self.assertRaises(TypeError):
            plan_batch(3, 3, BLOCKED, REQUESTS, dynamic_blocked="x",
                       max_expanded="x")
        # budget TypeError precedes the snapshot TypeError.
        with self.assertRaises(TypeError):
            plan_batch(3, 3, BLOCKED, REQUESTS, max_expanded="x",
                       snapshot=1)
        # snapshot TypeError precedes the max_cost TypeError.
        with self.assertRaises(TypeError):
            plan_batch(3, 3, BLOCKED, REQUESTS, snapshot=1, max_cost="x")

    def test_shared_option_type_errors(self):
        with self.assertRaises(TypeError):
            plan_batch(3, 3, BLOCKED, REQUESTS, max_expanded=1.5)
        with self.assertRaises(TypeError):
            plan_batch(3, 3, BLOCKED, REQUESTS, max_expanded=True)
        with self.assertRaises(TypeError):
            plan_batch(3, 3, BLOCKED, REQUESTS, snapshot=1)
        with self.assertRaises(TypeError):
            plan_batch(3, 3, BLOCKED, REQUESTS, max_cost="6")

    def test_shared_option_value_errors(self):
        with self.assertRaises(ValueError):
            plan_batch(3, 3, BLOCKED, REQUESTS, max_expanded=-1)
        with self.assertRaises(ValueError):
            plan_batch(3, 3, BLOCKED, REQUESTS, max_cost=-1)

    def test_zero_budget_is_budget_exhausted_per_query(self):
        batch = plan_batch(3, 3, BLOCKED, REQUESTS, max_expanded=0)
        for result in batch["results"]:
            self.assertEqual(result["status"], "budget_exhausted")
            self.assertEqual(result["expanded"], 0)
            self.assertIsNone(result["path"])

    def test_no_partial_results_on_any_error(self):
        # The first request is valid and would succeed, the second is
        # out of bounds: the whole call raises instead of returning the
        # first result.
        with self.assertRaises(ValueError):
            plan_batch(3, 3, BLOCKED, [[(0, 0), (1, 2)], [(9, 9), (0, 0)]])


if __name__ == "__main__":
    unittest.main()
