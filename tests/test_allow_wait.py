import json
import unittest

from app import (plan, plan_any, plan_batch, plan_k, plan_multi_start,
                 replay, resume)


# A one-cell-wide corridor: (1, 0) is blocked only at frame 1, so the
# direct route is infeasible and only a route that waits at the start
# (or later) can reach the goal. Frames past index 2 reuse frame 2.
FRAMES = [[], [(1, 0)], []]


def without_checkpoint(result):
    return {key: value for key, value in result.items()
            if key != "checkpoint"}


class AllowWaitValidationTest(unittest.TestCase):
    def test_non_bool_raises_type_error(self):
        for bad in (0, 1, "x", 1.5, None, [True]):
            with self.assertRaises(TypeError, msg=f"allow_wait={bad!r}"):
                plan(3, 3, set(), (0, 0), (2, 2), allow_wait=bad)
            with self.assertRaises(TypeError, msg=f"allow_wait={bad!r}"):
                plan_any(3, 3, set(), (0, 0), [(2, 2)], allow_wait=bad)
            with self.assertRaises(TypeError, msg=f"allow_wait={bad!r}"):
                plan_batch(3, 3, set(), [[(0, 0), (2, 2)]], allow_wait=bad)
            with self.assertRaises(TypeError, msg=f"allow_wait={bad!r}"):
                plan_multi_start(3, 3, set(), [(0, 0)], (2, 2),
                                 allow_wait=bad)
            with self.assertRaises(TypeError, msg=f"allow_wait={bad!r}"):
                plan_k(3, 3, set(), (0, 0), (2, 2), 2, allow_wait=bad)
            with self.assertRaises(TypeError, msg=f"allow_wait={bad!r}"):
                replay(3, 3, set(), (0, 0), (2, 2), [(0, 0)],
                       allow_wait=bad)

    def test_validated_after_dynamic_blocked(self):
        # The dynamic_blocked TypeError precedes the allow_wait TypeError.
        with self.assertRaises(TypeError) as caught:
            plan(3, 3, set(), (0, 0), (2, 2), dynamic_blocked="x",
                 allow_wait=1)
        self.assertIn("dynamic_blocked", str(caught.exception))
        # The frame-0 ValueError precedes the allow_wait TypeError.
        with self.assertRaises(ValueError):
            plan(3, 3, set(), (0, 0), (2, 2), dynamic_blocked=[[(0, 0)]],
                 allow_wait=1)

    def test_validated_before_budget_and_snapshot(self):
        # The allow_wait TypeError precedes the budget TypeError.
        with self.assertRaises(TypeError) as caught:
            plan(3, 3, set(), (0, 0), (2, 2), max_expanded="x",
                 allow_wait=1)
        self.assertIn("allow_wait", str(caught.exception))
        # The allow_wait TypeError precedes the snapshot TypeError.
        with self.assertRaises(TypeError) as caught:
            plan(3, 3, set(), (0, 0), (2, 2), snapshot=1, allow_wait=1)
        self.assertIn("allow_wait", str(caught.exception))
        # In plan_k the k check keeps its position before allow_wait.
        with self.assertRaises(TypeError) as caught:
            plan_k(3, 3, set(), (0, 0), (2, 2), "k", allow_wait=1)
        self.assertIn("k must be", str(caught.exception))

    def test_default_calls_unchanged(self):
        # Omitting allow_wait or passing False keeps the exact results.
        args = (3, 1, set(), (0, 0), (2, 0))
        default = plan(*args, dynamic_blocked=FRAMES)
        self.assertIsNone(default["path"])
        self.assertEqual(plan(*args, dynamic_blocked=FRAMES,
                              allow_wait=False), default)
        static_default = plan(3, 3, set(), (0, 0), (2, 2))
        self.assertEqual(plan(3, 3, set(), (0, 0), (2, 2),
                              allow_wait=False), static_default)


class AllowWaitSearchTest(unittest.TestCase):
    def test_wait_makes_route_feasible(self):
        result = plan(3, 1, set(), (0, 0), (2, 0), dynamic_blocked=FRAMES,
                      allow_wait=True)
        self.assertEqual(result["path"],
                         [(0, 0), (0, 0), (1, 0), (2, 0)])
        self.assertEqual(result["cost"], 2)

    def test_wait_adds_no_cost(self):
        costs = [[9, 1, 1]]
        result = plan(3, 1, set(), (0, 0), (2, 0), costs=costs,
                      dynamic_blocked=FRAMES, allow_wait=True)
        self.assertEqual(result["cost"], 2)
        self.assertEqual(result["path"], [(0, 0), (0, 0), (1, 0), (2, 0)])

    def test_wait_needs_free_cell_next_frame(self):
        # The start cell itself is blocked at frame 1: no waiting there.
        frames = [[], [(0, 0)], []]
        result = plan(3, 1, set(), (0, 0), (2, 0), dynamic_blocked=frames,
                      allow_wait=True)
        self.assertEqual(result["path"], [(0, 0), (1, 0), (2, 0)])

    def test_no_wait_after_final_frame(self):
        # The obstacle persists through the reused last frame, so waiting
        # can never dodge it.
        frames = [[], [(1, 0)]]
        result = plan(3, 1, set(), (0, 0), (2, 0), dynamic_blocked=frames,
                      allow_wait=True)
        self.assertIsNone(result["path"])

    def test_static_mode_unchanged_by_allow_wait(self):
        for kwargs in ({}, {"costs": [[1, 2, 1], [1, 1, 1], [3, 1, 1]]}):
            self.assertEqual(
                plan(3, 3, set(), (0, 0), (2, 2), allow_wait=True,
                     **kwargs),
                plan(3, 3, set(), (0, 0), (2, 2), **kwargs),
            )
            self.assertEqual(
                plan_any(3, 3, set(), (0, 0), [(2, 2), (0, 2)],
                         allow_wait=True, **kwargs),
                plan_any(3, 3, set(), (0, 0), [(2, 2), (0, 2)], **kwargs),
            )

    def test_start_equals_goal(self):
        result = plan(3, 1, set(), (1, 0), (1, 0), dynamic_blocked=FRAMES,
                      allow_wait=True)
        self.assertEqual(result,
                         {"path": [(1, 0)], "cost": 0, "expanded": 1})

    def test_plan_tie_break_prefers_earliest_arrival(self):
        # On an open grid the direct route and waiting routes tie on
        # cost; plan's (g, t, path) goal key picks the earliest arrival.
        frames = [[], [], []]
        result = plan(3, 1, set(), (0, 0), (2, 0), dynamic_blocked=frames,
                      allow_wait=True)
        self.assertEqual(result["path"], [(0, 0), (1, 0), (2, 0)])

    def test_plan_any_path_tie_break(self):
        # plan_any breaks cost ties by endpoint then the complete path,
        # so the lexicographically smallest waiting route wins.
        frames = [[], [], []]
        result = plan_any(3, 1, set(), (0, 0), [(2, 0)],
                          dynamic_blocked=frames, allow_wait=True)
        self.assertEqual(result["cost"], 2)
        self.assertEqual(result["path"],
                         [(0, 0), (0, 0), (0, 0), (1, 0), (2, 0)])

    def test_trace_records_wait_states(self):
        result = plan(3, 1, set(), (0, 0), (2, 0), dynamic_blocked=FRAMES,
                      allow_wait=True, trace=True)
        self.assertIn((0, 0, 1), result["expanded_nodes"])
        self.assertEqual(result["expanded"], len(result["expanded_nodes"]))

    def test_budget_truncates_wait_states(self):
        result = plan(3, 1, set(), (0, 0), (2, 0), dynamic_blocked=FRAMES,
                      allow_wait=True, max_expanded=2)
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(result["expanded"], 2)
        self.assertIsNone(result["path"])

    def test_max_cost_only_caps_move_cost(self):
        result = plan(3, 1, set(), (0, 0), (2, 0), dynamic_blocked=FRAMES,
                      allow_wait=True, max_cost=2)
        self.assertEqual(result["status"], "found")
        self.assertEqual(result["cost"], 2)
        short = plan(3, 1, set(), (0, 0), (2, 0), dynamic_blocked=FRAMES,
                     allow_wait=True, max_cost=1)
        self.assertEqual(short["status"], "cost_exhausted")

    def test_zero_budget_with_allow_wait(self):
        result = plan(3, 3, set(), (1, 1), (1, 1), max_expanded=0,
                      allow_wait=True)
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(result["expanded"], 0)


class AllowWaitVariantsTest(unittest.TestCase):
    def test_plan_batch(self):
        result = plan_batch(3, 1, set(), [[(0, 0), (2, 0)], [(2, 0), (0, 0)]],
                            dynamic_blocked=FRAMES, allow_wait=True)
        for single, query in zip(result["results"],
                                 [[(0, 0), (2, 0)], [(2, 0), (0, 0)]]):
            direct = plan(3, 1, set(), query[0], query[1],
                          dynamic_blocked=FRAMES, allow_wait=True)
            self.assertEqual(single, direct)

    def test_plan_multi_start(self):
        # The multi-start tie-break is the complete path only, so the
        # lexicographically smallest waiting route wins.
        result = plan_multi_start(3, 1, set(), [(0, 0)], (2, 0),
                                  dynamic_blocked=FRAMES, allow_wait=True)
        self.assertEqual(result["cost"], 2)
        self.assertEqual(result["path"],
                         [(0, 0), (0, 0), (0, 0), (1, 0), (2, 0)])

    def test_plan_k(self):
        result = plan_k(3, 1, set(), (0, 0), (2, 0), 5,
                        dynamic_blocked=FRAMES, allow_wait=True)
        self.assertEqual(result["paths"],
                         [[(0, 0), (0, 0), (0, 0), (1, 0), (2, 0)],
                          [(0, 0), (0, 0), (1, 0), (2, 0)]])
        self.assertEqual(result["costs"], [2, 2])
        # Without waiting no route exists.
        dry = plan_k(3, 1, set(), (0, 0), (2, 0), 5, dynamic_blocked=FRAMES)
        self.assertEqual(dry["paths"], [])
        self.assertEqual(dry["costs"], [])

    def test_successful_paths_replay(self):
        planners = [
            plan(3, 1, set(), (0, 0), (2, 0), dynamic_blocked=FRAMES,
                 allow_wait=True)["path"],
            plan_any(3, 1, set(), (0, 0), [(2, 0)], dynamic_blocked=FRAMES,
                     allow_wait=True)["path"],
            plan_multi_start(3, 1, set(), [(0, 0)], (2, 0),
                             dynamic_blocked=FRAMES, allow_wait=True)["path"],
        ]
        planners += plan_k(3, 1, set(), (0, 0), (2, 0), 5,
                           dynamic_blocked=FRAMES, allow_wait=True)["paths"]
        for path in planners:
            checked = replay(3, 1, set(), (0, 0), (2, 0), path,
                             dynamic_blocked=FRAMES, allow_wait=True)
            self.assertTrue(checked["valid"], msg=path)
            self.assertEqual(checked["cost"], 2)
            self.assertEqual(checked["steps"], len(path) - 1)


class AllowWaitReplayTest(unittest.TestCase):
    PATH = [(0, 0), (0, 0), (1, 0), (2, 0)]

    def test_wait_path_valid_with_allow_wait(self):
        result = replay(3, 1, set(), (0, 0), (2, 0), self.PATH,
                        dynamic_blocked=FRAMES, allow_wait=True)
        self.assertEqual(result, {"valid": True, "cost": 2, "steps": 3})

    def test_wait_path_invalid_without_allow_wait(self):
        result = replay(3, 1, set(), (0, 0), (2, 0), self.PATH,
                        dynamic_blocked=FRAMES)
        self.assertEqual(result, {"valid": False, "cost": None,
                                  "steps": None})
        diagnosed = replay(3, 1, set(), (0, 0), (2, 0), self.PATH,
                           dynamic_blocked=FRAMES, diagnose=True)
        self.assertEqual(diagnosed["error"], "repeated_coordinate")
        self.assertEqual(diagnosed["error_index"], 1)

    def test_consecutive_waits(self):
        path = [(0, 0), (0, 0), (0, 0), (1, 0), (2, 0)]
        result = replay(3, 1, set(), (0, 0), (2, 0), path,
                        dynamic_blocked=FRAMES, allow_wait=True)
        self.assertEqual(result, {"valid": True, "cost": 2, "steps": 4})

    def test_wait_after_final_frame(self):
        path = [(0, 0), (0, 0), (1, 0), (2, 0), (2, 0)]
        result = replay(3, 1, set(), (0, 0), (2, 0), path,
                        dynamic_blocked=FRAMES, allow_wait=True)
        self.assertEqual(result, {"valid": False, "cost": None,
                                  "steps": None})
        diagnosed = replay(3, 1, set(), (0, 0), (2, 0), path,
                           dynamic_blocked=FRAMES, diagnose=True,
                           allow_wait=True)
        self.assertEqual(diagnosed["error"], "wait_after_final_frame")
        self.assertEqual(diagnosed["error_index"], 4)

    def test_wait_into_blocked_frame(self):
        frames = [[], [(0, 0)], []]
        diagnosed = replay(3, 1, set(), (0, 0), (2, 0), self.PATH,
                           dynamic_blocked=frames, diagnose=True,
                           allow_wait=True)
        self.assertEqual(diagnosed["error"], "dynamic_blocked")
        self.assertEqual(diagnosed["error_index"], 1)

    def test_static_stay_stays_invalid(self):
        path = [(0, 0), (0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]
        diagnosed = replay(3, 3, set(), (0, 0), (2, 2), path,
                           diagnose=True, allow_wait=True)
        self.assertEqual(diagnosed["error"], "repeated_coordinate")
        self.assertEqual(diagnosed["error_index"], 1)

    def test_non_consecutive_revisit_stays_invalid(self):
        path = [(0, 0), (1, 0), (0, 0), (0, 1), (1, 1), (2, 1), (2, 2)]
        frames = [[], [], [], [], [], [], []]
        diagnosed = replay(3, 3, set(), (0, 0), (2, 2), path,
                           dynamic_blocked=frames, diagnose=True,
                           allow_wait=True)
        self.assertEqual(diagnosed["error"], "repeated_coordinate")
        self.assertEqual(diagnosed["error_index"], 2)


class AllowWaitSnapshotTest(unittest.TestCase):
    def assert_resume_matches(self, call, args, kwargs):
        full = call(*args, trace=True, **kwargs)
        n = full["expanded"]
        for b1 in range(0, n + 2):
            first = call(*args, trace=True, max_expanded=b1, snapshot=True,
                         **kwargs)
            self.assertEqual(
                without_checkpoint(first),
                call(*args, trace=True, max_expanded=b1, **kwargs),
                f"b1={b1}",
            )
            if first["status"] != "budget_exhausted":
                self.assertNotIn("checkpoint", first)
                continue
            checkpoint = json.loads(json.dumps(first["checkpoint"]))
            self.assertIs(checkpoint["allow_wait"], True)
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
        self.assert_resume_matches(
            plan, (3, 1, set(), (0, 0), (2, 0)),
            {"dynamic_blocked": FRAMES, "allow_wait": True})
        self.assert_resume_matches(
            plan_any, (3, 1, set(), (0, 0), [(2, 0)]),
            {"dynamic_blocked": FRAMES, "allow_wait": True})
        self.assert_resume_matches(
            plan_multi_start, (3, 1, set(), [(0, 0)], (2, 0)),
            {"dynamic_blocked": FRAMES, "allow_wait": True})

    def test_checkpoint_records_allow_wait_only_when_true(self):
        flagged = plan(3, 1, set(), (0, 0), (2, 0), dynamic_blocked=FRAMES,
                       allow_wait=True, max_expanded=2,
                       snapshot=True)["checkpoint"]
        self.assertIs(flagged["allow_wait"], True)
        plain = plan(3, 1, set(), (0, 0), (2, 0), dynamic_blocked=FRAMES,
                     max_expanded=0, snapshot=True)["checkpoint"]
        self.assertNotIn("allow_wait", plain)

    def test_missing_field_defaults_to_false(self):
        legacy = plan(3, 3, set(), (0, 0), (2, 2), max_expanded=2,
                      snapshot=True)["checkpoint"]
        self.assertNotIn("allow_wait", legacy)
        self.assertEqual(resume(legacy, max_expanded=100)["status"],
                         "found")

    def test_non_bool_field_raises_type_error(self):
        checkpoint = plan(3, 1, set(), (0, 0), (2, 0),
                          dynamic_blocked=FRAMES, allow_wait=True,
                          max_expanded=2, snapshot=True)["checkpoint"]
        for bad in (1, "true", None):
            broken = json.loads(json.dumps(checkpoint))
            broken["allow_wait"] = bad
            with self.assertRaises(TypeError, msg=f"allow_wait={bad!r}"):
                resume(broken)

    def test_field_inconsistent_with_state_raises_value_error(self):
        checkpoint = plan(3, 1, set(), (0, 0), (2, 0),
                          dynamic_blocked=FRAMES, allow_wait=True,
                          max_expanded=2, snapshot=True)["checkpoint"]
        # The recorded state contains waiting candidates, so claiming the
        # search did not allow waits is inconsistent.
        false_flag = json.loads(json.dumps(checkpoint))
        false_flag["allow_wait"] = False
        with self.assertRaises(ValueError):
            resume(false_flag)
        missing = json.loads(json.dumps(checkpoint))
        del missing["allow_wait"]
        with self.assertRaises(ValueError):
            resume(missing)


if __name__ == '__main__':
    unittest.main()
