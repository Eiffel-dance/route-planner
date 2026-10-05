import json
import unittest

from app import (plan, plan_any, plan_batch, plan_k, plan_multi_start,
                 replay, resume)


# A one-row corridor: (1, 0) is blocked only at frame 1, so the direct
# route is infeasible without waiting and a single wait at the start lets
# the route through. The last provided frame is index 2.
FRAMES = [[], [(1, 0)], []]
WAIT_PATH = [(0, 0), (0, 0), (1, 0), (2, 0)]
DOUBLE_WAIT_PATH = [(0, 0), (0, 0), (0, 0), (1, 0), (2, 0)]
BLOCKED = {(2, 0), (2, 1), (2, 2)}


def without_checkpoint(result):
    return {key: value for key, value in result.items()
            if key != "checkpoint"}


class AllowWaitValidationTest(unittest.TestCase):
    def test_type_errors(self):
        for bad in (1, 0, "true", None, 1.0, [True]):
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
                plan_k(3, 3, set(), (0, 0), (2, 2), 1, allow_wait=bad)
            with self.assertRaises(TypeError, msg=f"allow_wait={bad!r}"):
                replay(3, 3, set(), (0, 0), (2, 2), [(0, 0), (2, 2)],
                       allow_wait=bad)

    def test_validated_after_dynamic_blocked_checks(self):
        # dynamic_blocked TypeError precedes the allow_wait TypeError.
        with self.assertRaises(TypeError):
            plan(3, 3, set(), (0, 0), (2, 2), dynamic_blocked="x",
                 allow_wait=1)
        # A frame ValueError precedes the allow_wait TypeError.
        with self.assertRaises(ValueError):
            plan(3, 3, set(), (0, 0), (2, 2), dynamic_blocked=[[(9, 9)]],
                 allow_wait=1)
        # The frame-0 check precedes the allow_wait TypeError.
        with self.assertRaises(ValueError):
            plan(3, 3, set(), (0, 0), (2, 2), dynamic_blocked=[[(0, 0)]],
                 allow_wait=1)

    def test_validated_before_budget_snapshot_and_cost_limit(self):
        # The allow_wait TypeError precedes the budget ValueError.
        with self.assertRaises(TypeError):
            plan(3, 3, set(), (0, 0), (2, 2), max_expanded=-1, allow_wait=1)
        # The allow_wait TypeError precedes the snapshot TypeError.
        with self.assertRaises(TypeError):
            plan(3, 3, set(), (0, 0), (2, 2), snapshot=1, allow_wait="x")
        # The allow_wait TypeError precedes the max_cost ValueError.
        with self.assertRaises(TypeError):
            plan(3, 3, set(), (0, 0), (2, 2), max_cost=-1, allow_wait=1)
        # plan_k: the allow_wait TypeError precedes the k ValueError.
        with self.assertRaises(TypeError):
            plan_k(3, 3, set(), (0, 0), (2, 2), -1, allow_wait=1)

    def test_false_or_no_frames_keeps_legacy_behavior(self):
        legacy = plan(6, 4, BLOCKED, (0, 0), (5, 3), trace=True)
        explicit = plan(6, 4, BLOCKED, (0, 0), (5, 3), trace=True,
                        allow_wait=False)
        self.assertEqual(explicit, legacy)
        enabled = plan(6, 4, BLOCKED, (0, 0), (5, 3), trace=True,
                       allow_wait=True)
        self.assertEqual(enabled, legacy)
        for dyn in (None, []):
            result = plan(6, 4, BLOCKED, (0, 0), (5, 3), trace=True,
                          dynamic_blocked=dyn, allow_wait=True)
            self.assertEqual(result, legacy)
        legacy_k = plan_k(6, 4, BLOCKED, (0, 0), (5, 3), 3)
        enabled_k = plan_k(6, 4, BLOCKED, (0, 0), (5, 3), 3, allow_wait=True)
        self.assertEqual(enabled_k, legacy_k)


class AllowWaitSearchTest(unittest.TestCase):
    def test_wait_lets_the_route_through(self):
        blocked_direct = plan(3, 1, set(), (0, 0), (2, 0),
                              dynamic_blocked=FRAMES)
        self.assertIsNone(blocked_direct["path"])
        result = plan(3, 1, set(), (0, 0), (2, 0), dynamic_blocked=FRAMES,
                      allow_wait=True)
        self.assertEqual(result["path"], WAIT_PATH)
        self.assertEqual(result["cost"], 2)

    def test_wait_never_generated_past_the_final_frame(self):
        # Frame 1 blocks (1, 0) and persists: waiting cannot outlast it.
        result = plan(3, 1, set(), (0, 0), (2, 0),
                      dynamic_blocked=[[], [(1, 0)]], allow_wait=True)
        self.assertIsNone(result["path"])

    def test_wait_requires_a_free_cell_next_frame(self):
        # The wait cell itself is blocked at frame 1: no wait is possible.
        result = plan(3, 1, set(), (0, 0), (2, 0),
                      dynamic_blocked=[[], [(0, 0), (1, 0)], []],
                      allow_wait=True)
        self.assertIsNone(result["path"])

    def test_wait_is_free_and_moves_keep_their_cost(self):
        costs = [[5, 7, 3]]
        result = plan(3, 1, set(), (0, 0), (2, 0), costs=costs,
                      dynamic_blocked=FRAMES, allow_wait=True)
        self.assertEqual(result["path"], WAIT_PATH)
        self.assertEqual(result["cost"], 10)

    def test_max_cost_only_caps_move_cost(self):
        costs = [[5, 7, 3]]
        result = plan(3, 1, set(), (0, 0), (2, 0), costs=costs,
                      dynamic_blocked=FRAMES, allow_wait=True, max_cost=10)
        self.assertEqual(result["status"], "found")
        self.assertEqual(result["cost"], 10)
        short = plan(3, 1, set(), (0, 0), (2, 0), costs=costs,
                     dynamic_blocked=FRAMES, allow_wait=True, max_cost=9)
        self.assertEqual(short["status"], "cost_exhausted")
        self.assertIsNone(short["path"])
        # Waits are free: a zero limit still allows waiting, only moves
        # are discarded.
        zero = plan(3, 1, set(), (0, 0), (2, 0), dynamic_blocked=FRAMES,
                    allow_wait=True, max_cost=0, trace=True)
        self.assertEqual(zero["status"], "cost_exhausted")
        self.assertIn((0, 0, 1), zero["expanded_nodes"])

    def test_wait_states_count_and_trace_as_triples(self):
        result = plan(3, 1, set(), (0, 0), (2, 0), dynamic_blocked=FRAMES,
                      allow_wait=True, trace=True)
        self.assertEqual(result["expanded"], len(result["expanded_nodes"]))
        self.assertTrue(all(len(node) == 3
                            for node in result["expanded_nodes"]))
        self.assertIn((0, 0, 1), result["expanded_nodes"])

    def test_budget_truncates_wait_states_and_prefixes_the_trace(self):
        full = plan(3, 1, set(), (0, 0), (2, 0), dynamic_blocked=FRAMES,
                    allow_wait=True, trace=True)
        for budget in range(full["expanded"] + 1):
            result = plan(3, 1, set(), (0, 0), (2, 0),
                          dynamic_blocked=FRAMES, allow_wait=True,
                          trace=True, max_expanded=budget)
            self.assertLessEqual(result["expanded"], budget)
            self.assertEqual(result["expanded_nodes"],
                             full["expanded_nodes"][:result["expanded"]])

    def test_goal_tie_break_prefers_the_earlier_arrival(self):
        # Both wait routes cost 2; plan's (g, t, path) goal key picks the
        # single-wait route (t == 3) over the double-wait one (t == 4).
        result = plan(3, 1, set(), (0, 0), (2, 0), dynamic_blocked=FRAMES,
                      allow_wait=True)
        self.assertEqual(result["path"], WAIT_PATH)

    def test_start_equals_goal_stays_single_point(self):
        result = plan(3, 1, set(), (1, 0), (1, 0), dynamic_blocked=FRAMES,
                      allow_wait=True)
        self.assertEqual(result, {"path": [(1, 0)], "cost": 0,
                                  "expanded": 1})
        zero = plan(3, 1, set(), (1, 0), (1, 0), dynamic_blocked=FRAMES,
                    allow_wait=True, max_expanded=0)
        self.assertEqual(zero["status"], "budget_exhausted")
        self.assertEqual(zero["expanded"], 0)

    def test_result_is_input_order_independent(self):
        base = plan(3, 1, set(), (0, 0), (2, 0), dynamic_blocked=FRAMES,
                    allow_wait=True, trace=True)
        shuffled = plan(3, 1, set(), (0, 0), (2, 0),
                        dynamic_blocked=[[], [(1, 0), (1, 0)], []],
                        allow_wait=True, trace=True)
        self.assertEqual(shuffled, base)


class AllowWaitFamilyTest(unittest.TestCase):
    def test_plan_any_follows_full_path_lexicographic_tie_break(self):
        # plan_any's goal key is (g, endpoint, path): the double-wait
        # route is lexicographically smaller and wins at equal cost.
        result = plan_any(3, 1, set(), (0, 0), [(2, 0)],
                          dynamic_blocked=FRAMES, allow_wait=True)
        self.assertEqual(result["path"], DOUBLE_WAIT_PATH)
        self.assertEqual(result["cost"], 2)
        checked = replay(3, 1, set(), (0, 0), (2, 0), result["path"],
                         dynamic_blocked=FRAMES, allow_wait=True)
        self.assertEqual(checked, {"valid": True, "cost": 2,
                                   "steps": len(result["path"]) - 1})

    def test_plan_multi_start(self):
        result = plan_multi_start(3, 1, set(), [(0, 0)], (2, 0),
                                  dynamic_blocked=FRAMES, allow_wait=True)
        self.assertEqual(result["path"], DOUBLE_WAIT_PATH)
        self.assertEqual(result["cost"], 2)
        trivial = plan_multi_start(3, 1, set(), [(0, 0), (2, 0)], (2, 0),
                                   dynamic_blocked=FRAMES, allow_wait=True)
        self.assertEqual(trivial, {"path": [(2, 0)], "cost": 0,
                                   "expanded": 1})

    def test_plan_batch_matches_plan_per_request(self):
        single = plan(3, 1, set(), (0, 0), (2, 0), dynamic_blocked=FRAMES,
                      allow_wait=True, trace=True)
        batch = plan_batch(3, 1, set(), [[(0, 0), (2, 0)], [(2, 0), (2, 0)]],
                           dynamic_blocked=FRAMES, allow_wait=True,
                           trace=True)
        self.assertEqual(batch["results"][0], single)
        self.assertEqual(batch["results"][1],
                         {"path": [(2, 0)], "cost": 0, "expanded": 1,
                          "expanded_nodes": [(2, 0, 0)]})

    def test_plan_k_ranks_by_cost_then_complete_path(self):
        result = plan_k(3, 1, set(), (0, 0), (2, 0), 5,
                        dynamic_blocked=FRAMES, allow_wait=True)
        self.assertEqual(result["paths"], [DOUBLE_WAIT_PATH, WAIT_PATH])
        self.assertEqual(result["costs"], [2, 2])
        for path, cost in zip(result["paths"], result["costs"]):
            checked = replay(3, 1, set(), (0, 0), (2, 0), path,
                             dynamic_blocked=FRAMES, allow_wait=True)
            self.assertEqual(checked, {"valid": True, "cost": cost,
                                       "steps": len(path) - 1})

    def test_plan_k_limits_with_waits(self):
        result = plan_k(3, 1, set(), (0, 0), (2, 0), 2,
                        dynamic_blocked=FRAMES, allow_wait=True,
                        max_expanded=0)
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(result["paths"], [])
        self.assertEqual(result["expanded"], 0)
        capped = plan_k(3, 1, set(), (0, 0), (2, 0), 2, costs=[[5, 7, 3]],
                        dynamic_blocked=FRAMES, allow_wait=True, max_cost=9)
        self.assertEqual(capped["status"], "cost_exhausted")
        self.assertEqual(capped["paths"], [])


class AllowWaitReplayTest(unittest.TestCase):
    def test_valid_wait_path(self):
        result = replay(3, 1, set(), (0, 0), (2, 0), WAIT_PATH,
                        dynamic_blocked=FRAMES, allow_wait=True)
        self.assertEqual(result, {"valid": True, "cost": 2, "steps": 3})
        double = replay(3, 1, set(), (0, 0), (2, 0), DOUBLE_WAIT_PATH,
                        dynamic_blocked=FRAMES, allow_wait=True)
        self.assertEqual(double, {"valid": True, "cost": 2, "steps": 4})

    def test_wait_without_allow_wait_is_invalid(self):
        result = replay(3, 1, set(), (0, 0), (2, 0), WAIT_PATH,
                        dynamic_blocked=FRAMES)
        self.assertEqual(result, {"valid": False, "cost": None,
                                  "steps": None})
        diagnosed = replay(3, 1, set(), (0, 0), (2, 0), WAIT_PATH,
                           dynamic_blocked=FRAMES, diagnose=True)
        self.assertEqual(diagnosed["error"], "repeated_coordinate")
        self.assertEqual(diagnosed["error_index"], 1)

    def test_static_stay_is_invalid_even_with_allow_wait(self):
        result = replay(3, 1, set(), (0, 0), (2, 0), WAIT_PATH,
                        allow_wait=True, diagnose=True)
        self.assertEqual(result["error"], "repeated_coordinate")
        self.assertEqual(result["error_index"], 1)

    def test_wait_after_final_frame(self):
        path = [(0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (2, 0)]
        result = replay(3, 1, set(), (0, 0), (2, 0), path,
                        dynamic_blocked=FRAMES, allow_wait=True)
        self.assertEqual(result, {"valid": False, "cost": None,
                                  "steps": None})
        diagnosed = replay(3, 1, set(), (0, 0), (2, 0), path,
                           dynamic_blocked=FRAMES, diagnose=True,
                           allow_wait=True)
        self.assertEqual(diagnosed["error"], "wait_after_final_frame")
        self.assertEqual(diagnosed["error_index"], 3)

    def test_wait_into_a_blocked_cell_is_dynamic_blocked(self):
        frames = [[], [(0, 0)], []]
        result = replay(3, 1, set(), (0, 0), (2, 0), WAIT_PATH,
                        dynamic_blocked=frames, diagnose=True,
                        allow_wait=True)
        self.assertEqual(result["error"], "dynamic_blocked")
        self.assertEqual(result["error_index"], 1)

    def test_non_consecutive_revisit_stays_invalid(self):
        path = [(0, 0), (1, 0), (0, 0), (1, 0), (2, 0)]
        result = replay(3, 1, set(), (0, 0), (2, 0), path,
                        dynamic_blocked=[[], [], []], diagnose=True,
                        allow_wait=True)
        self.assertEqual(result["error"], "repeated_coordinate")
        self.assertEqual(result["error_index"], 2)

    def test_start_equals_goal_still_single_point_only(self):
        result = replay(3, 1, set(), (1, 0), (1, 0), [(1, 0)],
                        dynamic_blocked=FRAMES, allow_wait=True)
        self.assertEqual(result, {"valid": True, "cost": 0, "steps": 0})
        extra = replay(3, 1, set(), (1, 0), (1, 0), [(1, 0), (1, 0)],
                       dynamic_blocked=FRAMES, diagnose=True,
                       allow_wait=True)
        self.assertEqual(extra["error"], "start_goal_extra")
        self.assertEqual(extra["error_index"], 1)


class AllowWaitSnapshotTest(unittest.TestCase):
    def setUp(self):
        self.full = plan(3, 1, set(), (0, 0), (2, 0), dynamic_blocked=FRAMES,
                         allow_wait=True, trace=True)
        first = plan(3, 1, set(), (0, 0), (2, 0), dynamic_blocked=FRAMES,
                     allow_wait=True, max_expanded=2, snapshot=True)
        self.assertEqual(first["status"], "budget_exhausted")
        self.checkpoint = first["checkpoint"]

    def mutated(self, mutate):
        checkpoint = json.loads(json.dumps(self.checkpoint))
        mutate(checkpoint)
        return checkpoint

    def test_checkpoint_records_allow_wait(self):
        self.assertIs(self.checkpoint["allow_wait"], True)
        self.assertEqual(json.loads(json.dumps(self.checkpoint)),
                         self.checkpoint)

    def test_checkpoint_omits_field_without_waiting(self):
        first = plan(3, 1, set(), (0, 0), (2, 0), dynamic_blocked=FRAMES,
                     max_expanded=0, snapshot=True)
        self.assertNotIn("allow_wait", first["checkpoint"])

    def test_resume_equals_uninterrupted_call(self):
        resumed = resume(self.checkpoint)
        self.assertEqual(resumed["path"], self.full["path"])
        self.assertEqual(resumed["cost"], self.full["cost"])
        self.assertEqual(resumed["expanded"], self.full["expanded"])
        self.assertEqual(resumed["expanded_nodes"],
                         self.full["expanded_nodes"])

    def test_chained_resumes_keep_the_flag(self):
        second = resume(self.checkpoint, max_expanded=3)
        self.assertEqual(second["status"], "budget_exhausted")
        self.assertIs(second["checkpoint"]["allow_wait"], True)
        self.assertEqual(resume(second["checkpoint"])["path"],
                         self.full["path"])

    def test_old_checkpoint_without_field_resumes_as_false(self):
        first = plan(3, 1, set(), (0, 0), (2, 0), dynamic_blocked=FRAMES,
                     max_expanded=0, snapshot=True)
        resumed = resume(first["checkpoint"])
        legacy = plan(3, 1, set(), (0, 0), (2, 0), dynamic_blocked=FRAMES,
                      trace=True)
        self.assertEqual(resumed["path"], legacy["path"])
        self.assertEqual(resumed["expanded"], legacy["expanded"])

    def test_field_type_errors(self):
        for bad in (1, 0, "true", None, 1.0, [True]):
            with self.assertRaises(TypeError, msg=f"allow_wait={bad!r}"):
                resume(self.mutated(lambda c, b=bad: c.update(allow_wait=b)))

    def test_wait_state_without_flag_is_value_error(self):
        for mutate in (lambda c: c.update(allow_wait=False),
                       lambda c: c.pop("allow_wait")):
            with self.assertRaises(ValueError, msg=mutate):
                resume(self.mutated(mutate))

    def test_wait_past_final_frame_is_value_error(self):
        def mutate(c):
            c["state"]["open"] = [
                [2, 2, 0, 0, 3, 0, [[0, 0], [0, 0], [0, 0], [0, 0]]]
            ]
        with self.assertRaises(ValueError):
            resume(self.mutated(mutate))

    def test_plan_any_and_multi_start_checkpoints(self):
        full_any = plan_any(3, 1, set(), (0, 0), [(2, 0)],
                            dynamic_blocked=FRAMES, allow_wait=True,
                            trace=True)
        first_any = plan_any(3, 1, set(), (0, 0), [(2, 0)],
                             dynamic_blocked=FRAMES, allow_wait=True,
                             max_expanded=1, snapshot=True)
        self.assertIs(first_any["checkpoint"]["allow_wait"], True)
        resumed_any = resume(first_any["checkpoint"])
        self.assertEqual(resumed_any["path"], full_any["path"])
        self.assertEqual(resumed_any["expanded"], full_any["expanded"])
        full_multi = plan_multi_start(3, 1, set(), [(0, 0)], (2, 0),
                                      dynamic_blocked=FRAMES,
                                      allow_wait=True, trace=True)
        first_multi = plan_multi_start(3, 1, set(), [(0, 0)], (2, 0),
                                       dynamic_blocked=FRAMES,
                                       allow_wait=True, max_expanded=1,
                                       snapshot=True)
        self.assertIs(first_multi["checkpoint"]["allow_wait"], True)
        resumed_multi = resume(first_multi["checkpoint"])
        self.assertEqual(resumed_multi["path"], full_multi["path"])
        self.assertEqual(resumed_multi["expanded"],
                         full_multi["expanded"])

    def test_batch_checkpoint_resumes(self):
        batch = plan_batch(3, 1, set(), [[(0, 0), (2, 0)]],
                           dynamic_blocked=FRAMES, allow_wait=True,
                           max_expanded=1, snapshot=True)
        checkpoint = batch["results"][0]["checkpoint"]
        self.assertIs(checkpoint["allow_wait"], True)
        self.assertEqual(resume(checkpoint)["path"], self.full["path"])


if __name__ == '__main__':
    unittest.main()
