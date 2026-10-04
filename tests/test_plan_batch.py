import json
import random
import unittest

from app import plan, plan_batch, replay, resume


BLOCKED = {(2, 0), (2, 1), (2, 2), (0, 2), (4, 1)}

FRAMES = [
    [(0, 1), (0, 2)],
    [(0, 1), (1, 2), (2, 0), (2, 2)],
    [(2, 1), (2, 2)],
    [(0, 0), (1, 2), (2, 1), (2, 2)],
    [],
    [(0, 1)],
]


class PlanBatchSemanticsTest(unittest.TestCase):
    def test_matches_sequential_plan_calls(self):
        requests = [(0, 0), (5, 3)], [(0, 1), (5, 3)], [(1, 0), (5, 3)]
        batch = plan_batch(6, 4, BLOCKED, requests)
        expected = [
            plan(6, 4, BLOCKED, start, goal) for start, goal in requests
        ]
        self.assertEqual(batch, {"results": expected})
        self.assertEqual(set(batch), {"results"})

    def test_order_and_duplicate_requests_preserved(self):
        requests = [((0, 0), (5, 3)),
                    ((1, 0), (5, 3)),
                    ((0, 0), (5, 3)),
                    ((0, 0), (5, 3))]
        batch = plan_batch(6, 4, BLOCKED, requests)
        self.assertEqual(len(batch["results"]), 4)
        self.assertEqual(batch["results"][0], batch["results"][2])
        self.assertEqual(batch["results"][0], batch["results"][3])
        for index, (start, goal) in enumerate(requests):
            result = batch["results"][index]
            self.assertEqual(result, plan(6, 4, BLOCKED, start, goal))
            self.assertEqual(result["path"][0], start)
            self.assertEqual(result["path"][-1], goal)

    def test_list_coordinates_normalized_and_result_shape(self):
        result = plan_batch(3, 3, [[1, 0]], [[[0, 0], [2, 2]]])
        self.assertEqual(set(result), {"results"})
        single = result["results"][0]
        self.assertEqual(set(single), {"path", "cost", "expanded"})
        self.assertEqual(single["path"][0], (0, 0))
        self.assertEqual(single["path"][-1], (2, 2))

    def test_success_unreachable_and_start_equals_goal_mixed(self):
        wall = {(1, y) for y in range(3)}
        requests = [
            ((0, 0), (2, 2)),   # unreachable: the wall seals the grid
            ((0, 0), (0, 1)),   # reachable inside the sealed side
            ((2, 2), (2, 2)),   # single-point zero-cost route (free side)
        ]
        batch = plan_batch(3, 3, wall, requests)
        for result, (start, goal) in zip(batch["results"], requests):
            self.assertEqual(
                result, plan(3, 3, wall, start, goal),
                msg=f"request={(start, goal)}",
            )
        self.assertIsNone(batch["results"][0]["path"])
        self.assertIsNone(batch["results"][0]["cost"])
        self.assertEqual(batch["results"][0]["expanded"], 3)
        self.assertEqual(batch["results"][2],
                         {"path": [(2, 2)], "cost": 0, "expanded": 1})

    def test_trace_recorded_per_query(self):
        requests = [((0, 0), (5, 3)), ((1, 0), (5, 3))]
        batch = plan_batch(6, 4, BLOCKED, requests, trace=True)
        for result, (start, goal) in zip(batch["results"], requests):
            expected = plan(6, 4, BLOCKED, start, goal, trace=True)
            self.assertEqual(result, expected)
            self.assertEqual(set(result),
                             {"path", "cost", "expanded", "expanded_nodes"})
            self.assertEqual(len(result["expanded_nodes"]),
                             result["expanded"])
            self.assertEqual(result["expanded_nodes"][0], start)
            self.assertEqual(result["expanded_nodes"][-1], goal)
        # trace=False keeps the legacy keys everywhere.
        plain = plan_batch(6, 4, BLOCKED, requests)
        self.assertTrue(all(set(r) == {"path", "cost", "expanded"}
                            for r in plain["results"]))

    def test_budget_is_per_query_not_shared(self):
        # Each request needs more than one closed node; a budget of one
        # must stop every query independently with its own expanded == 1.
        requests = [((0, 0), (2, 2)), ((2, 2), (0, 0))]
        batch = plan_batch(3, 3, set(), requests, max_expanded=1)
        for result, (start, goal) in zip(batch["results"], requests):
            self.assertEqual(
                result,
                plan(3, 3, set(), start, goal, max_expanded=1),
            )
            self.assertEqual(result["status"], "budget_exhausted")
            self.assertIsNone(result["path"])
            self.assertEqual(result["expanded"], 1)
        # start == goal still succeeds under a one-node budget in the same
        # batch as an exhausting query.
        mixed = plan_batch(3, 3, set(),
                           [((1, 1), (1, 1)), ((0, 0), (2, 2))],
                           max_expanded=1)
        self.assertEqual(mixed["results"][0]["status"], "found")
        self.assertEqual(mixed["results"][0]["path"], [(1, 1)])
        self.assertEqual(mixed["results"][1]["status"],
                         "budget_exhausted")
        # Zero budget closes nothing for every query, start == goal too.
        zero = plan_batch(3, 3, set(),
                          [((1, 1), (1, 1)), ((0, 0), (2, 2))],
                          trace=True, max_expanded=0)
        for result in zero["results"]:
            self.assertEqual(result["status"], "budget_exhausted")
            self.assertEqual(result["expanded"], 0)
            self.assertEqual(result["expanded_nodes"], [])

    def test_cost_limit_is_per_query(self):
        costs = [
            [1, 10, 1],
            [1, 10, 1],
            [1, 1, 1],
        ]
        requests = [
            ((0, 0), (2, 0)),   # detour costs 6
            ((2, 2), (2, 0)),   # two cheap steps
        ]
        batch = plan_batch(3, 3, set(), requests, costs=costs, max_cost=3)
        self.assertIsNone(batch["results"][0]["path"])
        self.assertEqual(batch["results"][0]["status"], "cost_exhausted")
        self.assertEqual(batch["results"][1]["status"], "found")
        self.assertEqual(batch["results"][1]["cost"], 2)
        for result, (start, goal) in zip(batch["results"], requests):
            self.assertEqual(
                result,
                plan(3, 3, set(), start, goal, costs=costs, max_cost=3),
            )

    def test_dynamic_queries_share_frames_but_not_state(self):
        requests = [((1, 0), (2, 2)), ((0, 0), (2, 2))]
        batch = plan_batch(3, 3, set(), requests, trace=True,
                           dynamic_blocked=FRAMES)
        for result, (start, goal) in zip(batch["results"], requests):
            expected = plan(3, 3, set(), start, goal, trace=True,
                            dynamic_blocked=FRAMES)
            self.assertEqual(result, expected)
            self.assertTrue(all(len(node) == 3
                                for node in result["expanded_nodes"]))

    def test_every_successful_path_replays_offline(self):
        requests = [((0, 0), (5, 3)), ((1, 0), (5, 3)),
                    ((0, 1), (5, 2))]
        batch = plan_batch(6, 4, BLOCKED, requests)
        for result, (start, goal) in zip(batch["results"], requests):
            checked = replay(6, 4, BLOCKED, start, goal, result["path"])
            self.assertTrue(checked["valid"])
            self.assertEqual(checked["cost"], result["cost"])
            self.assertEqual(checked["steps"], len(result["path"]) - 1)

    def test_dynamic_paths_replay_offline(self):
        requests = [((1, 0), (2, 2)), ((0, 0), (2, 2))]
        batch = plan_batch(3, 3, set(), requests, dynamic_blocked=FRAMES)
        for result, (start, goal) in zip(batch["results"], requests):
            checked = replay(3, 3, set(), start, goal, result["path"],
                             dynamic_blocked=FRAMES)
            self.assertTrue(checked["valid"])
            self.assertEqual(checked["cost"], result["cost"])
            self.assertEqual(checked["steps"], len(result["path"]) - 1)

    def test_checkpoint_hands_straight_to_resume(self):
        requests = [((1, 1), (1, 1)),  # found immediately, no checkpoint
                    ((0, 0), (2, 2))]  # stopped by the one-node budget
        batch = plan_batch(3, 3, set(), requests, trace=True,
                           max_expanded=1, snapshot=True)
        self.assertNotIn("checkpoint", batch["results"][0])
        stopped = batch["results"][1]
        self.assertEqual(stopped["status"], "budget_exhausted")
        self.assertIn("checkpoint", stopped)
        # The checkpoint stands on its own as JSON-serializable data.
        json.loads(json.dumps(stopped["checkpoint"]))
        resumed = resume(stopped["checkpoint"], max_expanded=100)
        uninterrupted = plan(3, 3, set(), (0, 0), (2, 2), trace=True,
                             max_expanded=100)
        self.assertEqual({key: value
                          for key, value in resumed.items()
                          if key != "checkpoint"},
                         uninterrupted)
        checked = replay(3, 3, set(), (0, 0), (2, 2), resumed["path"])
        self.assertEqual(checked["cost"], resumed["cost"])
        self.assertEqual(checked["steps"], len(resumed["path"]) - 1)

    def test_blocked_and_request_permutation_invariant(self):
        requests = [((0, 0), (5, 3)), ((1, 0), (5, 3)),
                    ((0, 1), (5, 2)), ((0, 0), (5, 3))]
        cells = sorted(BLOCKED)
        orderings = [
            (cells, requests),
            (list(reversed(cells)), list(reversed(requests))),
            (cells, [requests[2], requests[0], requests[3], requests[1]]),
        ]
        baselines = [
            plan(6, 4, cells, start, goal) for start, goal in requests
        ]
        for blocked, ordered in orderings:
            batch = plan_batch(6, 4, blocked, ordered)
            for result, request in zip(batch["results"], ordered):
                self.assertEqual(result,
                                 baselines[requests.index(request)])

    def test_result_is_json_serializable(self):
        requests = [((0, 0), (5, 3)), ((1, 0), (5, 3))]
        batch = plan_batch(6, 4, BLOCKED, requests, trace=True)
        restored = json.loads(json.dumps(batch))
        self.assertEqual(len(restored["results"]), 2)
        self.assertEqual(
            restored["results"][0]["path"],
            [list(point) for point in batch["results"][0]["path"]],
        )

    def test_matches_plan_on_random_grids(self):
        rng = random.Random(20261006)
        for _ in range(40):
            w, h = rng.randint(1, 6), rng.randint(1, 6)
            cells = [(x, y) for y in range(h) for x in range(w)]
            blocked = {p for p in cells[1:-1] if rng.random() < 0.2}
            free = [p for p in cells if p not in blocked]
            if len(free) < 2:
                continue
            requests = [(rng.choice(free), rng.choice(free))
                        for _ in range(rng.randint(1, 4))]
            kwargs = {}
            if rng.random() < 0.5:
                kwargs["costs"] = [[rng.randint(1, 9) for _ in range(w)]
                                   for _ in range(h)]
            if rng.random() < 0.5:
                kwargs["trace"] = True
            if rng.random() < 0.5:
                frames = [[p for p in cells if rng.random() < 0.15]
                          for _ in range(rng.randint(1, 4))]
                frames[0] = [p for p in frames[0]
                             if p not in {start for start, _ in requests}]
                kwargs["dynamic_blocked"] = frames
            if rng.random() < 0.5:
                kwargs["max_expanded"] = rng.randint(0, 12)
            batch = plan_batch(w, h, blocked, requests, **kwargs)
            for result, (start, goal) in zip(batch["results"], requests):
                self.assertEqual(
                    result, plan(w, h, blocked, start, goal, **kwargs),
                    msg=f"grid={w}x{h} request={(start, goal)} "
                        f"kwargs={kwargs}",
                )


class PlanBatchValidationTest(unittest.TestCase):
    def test_requests_outer_type_errors(self):
        for bad in (None, 42, "ab", b"ab", 1.5, {((0, 0), (1, 1))},
                    frozenset({((0, 0), (1, 1))})):
            with self.assertRaises(TypeError, msg=f"requests={bad!r}"):
                plan_batch(3, 3, set(), bad)
        # A generator is an iterable but not a sequence.
        with self.assertRaises(TypeError):
            plan_batch(3, 3, set(),
                       (((x, 0), (x, 1)) for x in range(2)))

    def test_requests_empty_is_value_error(self):
        for bad in ((), [], tuple()):
            with self.assertRaises(ValueError, msg=f"requests={bad!r}"):
                plan_batch(3, 3, set(), bad)

    def test_element_shape_and_coordinate_type_errors(self):
        for bad in (42, "ab", b"ab", None, 1.5,
                    ((0, 0),),                         # one coordinate
                    ((0, 0), (1, 1), (2, 2)),          # three coordinates
                    [(0, 0)],                          # element as coordinate
                    ("ab", (1, 1)),                    # non-sequence start
                    ((0, 0), 5),                       # non-sequence goal
                    ((1.5, 0), (1, 1)),                # float coordinate
                    ((0, True), (1, 1)),               # bool coordinate
                    ((0, 0), (None, 1)),               # non-int coordinate
                    ({"x": 0, "y": 0}, (1, 1))):
            with self.assertRaises(TypeError, msg=f"element={bad!r}"):
                plan_batch(3, 3, set(), [bad])

    def test_endpoint_bounds_errors(self):
        for bad_start, bad_goal in (
                ((-1, 0), (1, 1)),
                ((0, -1), (1, 1)),
                ((3, 0), (1, 1)),
                ((0, 3), (1, 1)),
                ((0, 0), (3, 3)),
                ((10, 10), (0, 0)),
        ):
            with self.assertRaises(ValueError,
                                   msg=f"request={(bad_start, bad_goal)}"):
                plan_batch(3, 3, set(), [(bad_start, bad_goal)])

    def test_endpoint_on_static_obstacle(self):
        with self.assertRaises(ValueError):
            plan_batch(3, 3, {(0, 0)}, [((0, 0), (2, 2))])
        with self.assertRaises(ValueError):
            plan_batch(3, 3, {(2, 2)}, [((0, 0), (2, 2))])
        # The bad request is reached even when an earlier one is valid.
        with self.assertRaises(ValueError):
            plan_batch(3, 3, {(2, 2)},
                       [((0, 0), (1, 1)), ((0, 0), (2, 2))])

    def test_start_blocked_at_frame_zero(self):
        with self.assertRaises(ValueError):
            plan_batch(3, 3, set(), [((0, 0), (2, 2))],
                       dynamic_blocked=[[(0, 0)]])
        # Every request's frame-0 start is checked before searching.
        with self.assertRaises(ValueError):
            plan_batch(3, 3, set(),
                       [((1, 1), (2, 2)), ((0, 0), (2, 2))],
                       dynamic_blocked=[[(0, 0)]])

    def test_all_errors_decided_before_any_search(self):
        # Validation failures raise; no partial results are ever returned.
        with self.assertRaises(TypeError):
            plan_batch(3, 3, set(), "not-requests")
        with self.assertRaises(ValueError):
            plan_batch(3, 3, set(), [])
        with self.assertRaises(ValueError):
            plan_batch(3, 3, {(0, 0)}, [((0, 0), (2, 2))])

    def test_shared_argument_validation_order_matches_plan(self):
        good = [((0, 0), (2, 2))]
        # width/height type errors.
        for bad in ("3", 3.0, None, True, [3]):
            with self.assertRaises(TypeError):
                plan_batch(bad, 3, set(), good)
            with self.assertRaises(TypeError):
                plan_batch(3, bad, set(), good)
        # width ValueError precedes the requests structure TypeError.
        with self.assertRaises(ValueError):
            plan_batch(0, 3, set(), "bad-requests")
        # requests structure TypeError (goal slot) precedes blocked
        # TypeError and the trace TypeError.
        with self.assertRaises(TypeError):
            plan_batch(3, 3, 42, "bad-requests", trace=1)
        # request bounds ValueError precedes the blocked TypeError.
        with self.assertRaises(ValueError):
            plan_batch(3, 3, 42, [((3, 3), (0, 0))])
        # obstacle ValueError precedes the costs and trace TypeErrors.
        with self.assertRaises(ValueError):
            plan_batch(3, 3, {(0, 0)}, good, trace=1)
        # costs shape ValueError precedes the trace TypeError.
        with self.assertRaises(ValueError):
            plan_batch(3, 3, set(), good,
                       costs=[[1, 1], [1, 1], [1, 1]], trace=1)
        # trace TypeError precedes the dynamic-blocked TypeError.
        with self.assertRaises(TypeError):
            plan_batch(3, 3, set(), good, trace=1, dynamic_blocked=42)
        # dynamic structure TypeError precedes the frame-0 ValueError.
        with self.assertRaises(TypeError):
            plan_batch(3, 3, set(), good, dynamic_blocked=42)
        # frame-0 ValueError precedes the max_expanded TypeError.
        with self.assertRaises(ValueError):
            plan_batch(3, 3, set(), [((0, 0), (2, 2))],
                       dynamic_blocked=[[(0, 0)]], max_expanded="x")
        # budget TypeError precedes the snapshot TypeError.
        with self.assertRaises(TypeError):
            plan_batch(3, 3, set(), good,
                       max_expanded="x", snapshot=1)
        # snapshot TypeError precedes the max_cost TypeError.
        with self.assertRaises(TypeError):
            plan_batch(3, 3, set(), good,
                       snapshot=1, max_cost="x")

    def test_optional_limit_rules_match_plan(self):
        good = [((0, 0), (2, 2))]
        for bad in ("5", 1.5, True, [5], (5,)):
            with self.assertRaises(TypeError):
                plan_batch(3, 3, set(), good, max_expanded=bad)
            with self.assertRaises(TypeError):
                plan_batch(3, 3, set(), good, max_cost=bad)
        with self.assertRaises(ValueError):
            plan_batch(3, 3, set(), good, max_expanded=-1)
        with self.assertRaises(ValueError):
            plan_batch(3, 3, set(), good, max_cost=-1)
        for bad in (1, 0, "true", None, []):
            with self.assertRaises(TypeError):
                plan_batch(3, 3, set(), good, snapshot=bad)


if __name__ == "__main__":
    unittest.main()
