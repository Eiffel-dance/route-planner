import json
import unittest

from app import plan, plan_any, replay, resume


BLOCKED = {(2, 0), (2, 1), (2, 2)}
FRAMES = [
    [(0, 1), (0, 2)],
    [(0, 1), (1, 2), (2, 0), (2, 2)],
    [(2, 1), (2, 2)],
    [(0, 0), (1, 2), (2, 1), (2, 2)],
    [],
    [(0, 1)],
]
# Cheapest route (0, 0) -> (2, 0) runs along the bottom row and back up
# the right column for cost 6; the direct route through the expensive
# middle column costs 11.
COSTS = [[1, 10, 1], [1, 10, 1], [1, 1, 1]]


def without_checkpoint(result):
    return {key: value for key, value in result.items()
            if key != "checkpoint"}


class MaxCostValidationTest(unittest.TestCase):
    def test_type_errors(self):
        for bad in ("5", 1.5, True, [5], (5,), 2.0):
            with self.assertRaises(TypeError, msg=f"max_cost={bad!r}"):
                plan(3, 3, set(), (0, 0), (2, 2), max_cost=bad)
            with self.assertRaises(TypeError, msg=f"max_cost={bad!r}"):
                plan_any(3, 3, set(), (0, 0), [(2, 2)], max_cost=bad)

    def test_negative_is_value_error(self):
        for bad in (-1, -100):
            with self.assertRaises(ValueError, msg=f"max_cost={bad!r}"):
                plan(3, 3, set(), (0, 0), (2, 2), max_cost=bad)
            with self.assertRaises(ValueError, msg=f"max_cost={bad!r}"):
                plan_any(3, 3, set(), (0, 0), [(2, 2)], max_cost=bad)

    def test_zero_is_accepted(self):
        result = plan(3, 3, set(), (0, 0), (0, 0), max_cost=0)
        self.assertEqual(result["status"], "found")
        self.assertEqual(result["cost"], 0)

    def test_validated_after_all_existing_checks(self):
        # Grid ValueError precedes the max_cost TypeError.
        with self.assertRaises(ValueError):
            plan(0, 3, set(), (0, 0), (1, 1), max_cost="x")
        # costs ValueError precedes the max_cost TypeError.
        with self.assertRaises(ValueError):
            plan(2, 2, set(), (0, 0), (1, 1), costs=[[1]], max_cost="x")
        # trace TypeError precedes the max_cost ValueError.
        with self.assertRaises(TypeError):
            plan(3, 3, set(), (0, 0), (2, 2), trace=1, max_cost=-1)
        # dynamic_blocked TypeError precedes the max_cost TypeError.
        with self.assertRaises(TypeError):
            plan(3, 3, set(), (0, 0), (2, 2), dynamic_blocked="x",
                 max_cost="x")
        # budget ValueError precedes the max_cost TypeError.
        with self.assertRaises(ValueError):
            plan(3, 3, set(), (0, 0), (2, 2), max_expanded=-1,
                 max_cost="x")
        # snapshot TypeError precedes the max_cost ValueError.
        with self.assertRaises(TypeError):
            plan(3, 3, set(), (0, 0), (2, 2), snapshot=1, max_cost=-1)
        with self.assertRaises(TypeError):
            plan_any(3, 3, set(), (0, 0), [(2, 2)], snapshot=1, max_cost=-1)

    def test_omitted_or_none_keeps_legacy_shape(self):
        legacy = plan(6, 4, BLOCKED, (0, 0), (5, 3), trace=True)
        explicit = plan(6, 4, BLOCKED, (0, 0), (5, 3), trace=True,
                        max_cost=None)
        self.assertEqual(explicit, legacy)
        self.assertNotIn("status", legacy)
        legacy_any = plan_any(6, 4, BLOCKED, (0, 0), [(5, 3)], trace=True)
        explicit_any = plan_any(6, 4, BLOCKED, (0, 0), [(5, 3)], trace=True,
                                max_cost=None)
        self.assertEqual(explicit_any, legacy_any)
        self.assertNotIn("status", legacy_any)


class MaxCostSearchTest(unittest.TestCase):
    def test_found_within_limit_carries_status(self):
        result = plan(3, 3, set(), (0, 0), (2, 0), costs=COSTS, max_cost=6)
        self.assertEqual(result["status"], "found")
        self.assertEqual(result["cost"], 6)
        self.assertEqual(result["path"][0], (0, 0))
        self.assertEqual(result["path"][-1], (2, 0))

    def test_limit_does_not_change_path_choice(self):
        limited = plan(6, 4, BLOCKED, (0, 0), (5, 3), trace=True,
                       max_cost=1000)
        legacy = plan(6, 4, BLOCKED, (0, 0), (5, 3), trace=True)
        self.assertEqual(limited["path"], legacy["path"])
        self.assertEqual(limited["cost"], legacy["cost"])
        self.assertEqual(limited["expanded"], legacy["expanded"])
        self.assertEqual(limited["expanded_nodes"],
                         legacy["expanded_nodes"])
        self.assertEqual(limited["status"], "found")

    def test_cost_exhausted(self):
        result = plan(3, 3, set(), (0, 0), (2, 0), costs=COSTS, max_cost=4,
                      trace=True)
        self.assertEqual(result["status"], "cost_exhausted")
        self.assertIsNone(result["path"])
        self.assertIsNone(result["cost"])
        # Only the cheap cells within the limit are closed.
        self.assertEqual(result["expanded"], 5)
        self.assertEqual(result["expanded_nodes"],
                         [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)])
        self.assertEqual(result["expanded"],
                         len(result["expanded_nodes"]))

    def test_unreachable_without_discards_stays_unreachable(self):
        # The goal is walled off; nothing is ever discarded by the limit.
        wall = {(1, y) for y in range(3)}
        result = plan(3, 3, wall, (0, 0), (2, 2), max_cost=100)
        self.assertEqual(result["status"], "unreachable")
        self.assertIsNone(result["path"])
        self.assertEqual(result["expanded"], 3)

    def test_zero_limit_discards_every_step(self):
        result = plan(2, 1, set(), (0, 0), (1, 0), max_cost=0, trace=True)
        self.assertEqual(result["status"], "cost_exhausted")
        self.assertEqual(result["expanded"], 1)
        self.assertEqual(result["expanded_nodes"], [(0, 0)])

    def test_start_equals_goal_zero_limit(self):
        result = plan(3, 3, set(), (1, 1), (1, 1), max_cost=0)
        self.assertEqual(result, {"path": [(1, 1)], "cost": 0,
                                  "expanded": 1, "status": "found"})

    def test_zero_budget_rule_still_wins(self):
        result = plan(3, 3, set(), (1, 1), (1, 1), max_expanded=0,
                      max_cost=0)
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(result["expanded"], 0)

    def test_budget_exhausted_has_priority_over_cost_exhausted(self):
        result = plan(6, 4, BLOCKED, (0, 0), (5, 3), max_expanded=3,
                      max_cost=2)
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(result["expanded"], 3)

    def test_trace_records_only_closed_nodes(self):
        result = plan(3, 3, set(), (0, 0), (2, 0), costs=COSTS, max_cost=6,
                      trace=True)
        self.assertEqual(result["status"], "found")
        self.assertEqual(result["expanded"], len(result["expanded_nodes"]))
        self.assertEqual(result["expanded_nodes"][0], (0, 0))
        self.assertEqual(result["expanded_nodes"][-1], (2, 0))

    def test_dynamic_mode(self):
        full = plan(3, 3, set(), (1, 0), (2, 2), dynamic_blocked=FRAMES)
        limited = plan(3, 3, set(), (1, 0), (2, 2), dynamic_blocked=FRAMES,
                       max_cost=full["cost"])
        self.assertEqual(limited["status"], "found")
        self.assertEqual(limited["path"], full["path"])
        self.assertEqual(limited["cost"], full["cost"])
        exhausted = plan(3, 3, set(), (1, 0), (2, 2),
                         dynamic_blocked=FRAMES,
                         max_cost=full["cost"] - 1)
        self.assertEqual(exhausted["status"], "cost_exhausted")
        self.assertIsNone(exhausted["path"])

    def test_plan_any(self):
        goals = [(5, 3), (5, 2)]
        full = plan_any(6, 4, BLOCKED, (0, 0), goals)
        limited = plan_any(6, 4, BLOCKED, (0, 0), goals,
                           max_cost=full["cost"])
        self.assertEqual(limited["status"], "found")
        self.assertEqual(limited["path"], full["path"])
        self.assertEqual(limited["path"][-1], full["path"][-1])
        exhausted = plan_any(6, 4, BLOCKED, (0, 0), goals, max_cost=1)
        self.assertEqual(exhausted["status"], "cost_exhausted")
        self.assertIsNone(exhausted["path"])
        self.assertIsNone(exhausted["cost"])

    def test_plan_any_start_in_goals_zero_limit(self):
        result = plan_any(3, 3, set(), (1, 1), [(1, 1), (2, 2)], max_cost=0)
        self.assertEqual(result, {"path": [(1, 1)], "cost": 0,
                                  "expanded": 1, "status": "found"})

    def test_found_path_replays(self):
        result = plan(3, 3, set(), (0, 0), (2, 0), costs=COSTS, max_cost=6)
        checked = replay(3, 3, set(), (0, 0), (2, 0), result["path"],
                         costs=COSTS)
        self.assertEqual(checked, {"valid": True, "cost": result["cost"],
                                   "steps": len(result["path"]) - 1})


class MaxCostSnapshotTest(unittest.TestCase):
    def test_checkpoint_records_max_cost(self):
        result = plan(3, 3, set(), (0, 0), (2, 0), costs=COSTS, max_cost=6,
                      max_expanded=2, snapshot=True)
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(result["checkpoint"]["max_cost"], 6)
        checkpoint = json.loads(json.dumps(result["checkpoint"]))
        self.assertEqual(checkpoint, result["checkpoint"])

    def test_checkpoint_omits_field_without_max_cost(self):
        result = plan(6, 4, BLOCKED, (0, 0), (5, 3), max_expanded=3,
                      snapshot=True)
        self.assertNotIn("max_cost", result["checkpoint"])
        self.assertNotIn("cost_limited", result["checkpoint"]["state"])

    def test_cost_exhausted_carries_no_checkpoint(self):
        result = plan(3, 3, set(), (0, 0), (2, 0), costs=COSTS, max_cost=4,
                      snapshot=True)
        self.assertEqual(result["status"], "cost_exhausted")
        self.assertNotIn("checkpoint", result)

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
            # An unbudgeted resume equals the unpaused traced call.
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

    def test_static_plan_with_cost_limit(self):
        self.assert_resume_matches(
            plan, (3, 3, set(), (0, 0), (2, 0)),
            {"costs": COSTS, "max_cost": 6})

    def test_static_plan_cost_exhausted(self):
        self.assert_resume_matches(
            plan, (3, 3, set(), (0, 0), (2, 0)),
            {"costs": COSTS, "max_cost": 4})

    def test_dynamic_plan_with_cost_limit(self):
        self.assert_resume_matches(
            plan, (3, 3, set(), (1, 0), (2, 2)),
            {"dynamic_blocked": FRAMES, "max_cost": 6})

    def test_plan_any_with_cost_limit(self):
        self.assert_resume_matches(
            plan_any, (6, 4, BLOCKED, (0, 0), [(5, 3), (5, 2)]),
            {"max_cost": 8})

    def test_plan_any_dynamic_with_cost_limit(self):
        self.assert_resume_matches(
            plan_any, (3, 3, set(), (1, 0), [(2, 2), (0, 2)]),
            {"dynamic_blocked": FRAMES, "max_cost": 6})

    def test_resume_status_present_from_checkpoint_limit(self):
        # The checkpoint's max_cost makes the resumed result carry a
        # status even when resume itself gets no budget.
        first = plan(3, 3, set(), (0, 0), (2, 0), costs=COSTS, max_cost=6,
                     max_expanded=2, snapshot=True)
        resumed = resume(first["checkpoint"])
        self.assertEqual(resumed["status"], "found")
        self.assertEqual(resumed["cost"], 6)

    def test_old_checkpoint_without_field_resumes_as_none(self):
        first = plan(6, 4, BLOCKED, (0, 0), (5, 3), max_expanded=3,
                     snapshot=True)
        self.assertNotIn("max_cost", first["checkpoint"])
        resumed = resume(first["checkpoint"])
        self.assertNotIn("status", resumed)
        self.assertEqual(resumed["path"],
                         plan(6, 4, BLOCKED, (0, 0), (5, 3))["path"])

    def test_checkpoint_max_cost_type_errors(self):
        first = plan(3, 3, set(), (0, 0), (2, 0), costs=COSTS, max_cost=6,
                     max_expanded=2, snapshot=True)
        for bad in ("5", 1.5, True, [5]):
            checkpoint = json.loads(json.dumps(first["checkpoint"]))
            checkpoint["max_cost"] = bad
            with self.assertRaises(TypeError, msg=f"max_cost={bad!r}"):
                resume(checkpoint)

    def test_checkpoint_max_cost_negative_is_value_error(self):
        first = plan(3, 3, set(), (0, 0), (2, 0), costs=COSTS, max_cost=6,
                     max_expanded=2, snapshot=True)
        checkpoint = json.loads(json.dumps(first["checkpoint"]))
        checkpoint["max_cost"] = -1
        with self.assertRaises(ValueError):
            resume(checkpoint)

    def test_checkpoint_state_inconsistent_with_limit(self):
        first = plan(3, 3, set(), (0, 0), (2, 0), costs=COSTS, max_cost=6,
                     max_expanded=2, snapshot=True)
        # A recorded candidate cost above the limit cannot have survived.
        checkpoint = json.loads(json.dumps(first["checkpoint"]))
        checkpoint["max_cost"] = 0
        with self.assertRaises(ValueError):
            resume(checkpoint)

    def test_cost_limited_flag_without_limit_is_value_error(self):
        first = plan(3, 3, set(), (0, 0), (2, 0), costs=COSTS, max_cost=6,
                     max_expanded=2, snapshot=True)
        checkpoint = json.loads(json.dumps(first["checkpoint"]))
        del checkpoint["max_cost"]
        checkpoint["state"]["cost_limited"] = True
        with self.assertRaises(ValueError):
            resume(checkpoint)

    def test_cost_limited_flag_type_error(self):
        first = plan(3, 3, set(), (0, 0), (2, 0), costs=COSTS, max_cost=6,
                     max_expanded=2, snapshot=True)
        checkpoint = json.loads(json.dumps(first["checkpoint"]))
        checkpoint["state"]["cost_limited"] = 1
        with self.assertRaises(TypeError):
            resume(checkpoint)

    def test_repeated_resumes_with_cost_limit(self):
        kwargs = {"costs": COSTS, "max_cost": 6}
        first = plan(3, 3, set(), (0, 0), (2, 0), max_expanded=1,
                     snapshot=True, **kwargs)
        second = resume(first["checkpoint"], max_expanded=3)
        third = resume(second["checkpoint"], max_expanded=100)
        direct = plan(3, 3, set(), (0, 0), (2, 0), trace=True,
                      max_expanded=100, **kwargs)
        self.assertEqual(without_checkpoint(third),
                         without_checkpoint(direct))
        self.assertEqual(third["status"], "found")


class MaxCostReplayTest(unittest.TestCase):
    def test_replay_accepts_no_max_cost(self):
        with self.assertRaises(TypeError):
            replay(3, 3, set(), (0, 0), (2, 2),
                   [(0, 0), (1, 1), (2, 2)], max_cost=5)

    def test_replay_behavior_unchanged(self):
        checked = replay(3, 3, set(), (0, 0), (2, 0),
                         [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1),
                          (2, 0)],
                         costs=COSTS)
        self.assertEqual(checked, {"valid": True, "cost": 6, "steps": 6})


if __name__ == '__main__':
    unittest.main()
