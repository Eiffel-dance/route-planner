import json
import unittest

from app import plan, plan_any, replay, resume


FRAMES = [
    [(0, 1), (0, 2)],
    [(0, 1), (1, 2), (2, 0), (2, 2)],
    [(2, 1), (2, 2)],
    [(0, 0), (1, 2), (2, 1), (2, 2)],
    [],
    [(0, 1)],
]
BLOCKED = {(2, 0), (2, 1), (2, 2)}
WALL = {(1, y) for y in range(3)}


def without_checkpoint(result):
    return {key: value for key, value in result.items()
            if key != "checkpoint"}


class MaxCostValidationTest(unittest.TestCase):
    def test_omitted_or_none_keeps_legacy_shape(self):
        omitted = plan(3, 3, set(), (0, 0), (2, 2), trace=True)
        explicit = plan(3, 3, set(), (0, 0), (2, 2), trace=True,
                        max_cost=None)
        self.assertEqual(explicit, omitted)
        self.assertNotIn("status", omitted)
        any_omitted = plan_any(3, 3, set(), (0, 0), [(2, 2)], trace=True)
        any_explicit = plan_any(3, 3, set(), (0, 0), [(2, 2)], trace=True,
                                max_cost=None)
        self.assertEqual(any_explicit, any_omitted)
        self.assertNotIn("status", any_omitted)

    def test_type_errors(self):
        for bad in ("5", 1.5, True, [5], (5,)):
            with self.assertRaises(TypeError, msg=f"max_cost={bad!r}"):
                plan(3, 3, set(), (0, 0), (2, 2), max_cost=bad)
            with self.assertRaises(TypeError, msg=f"max_cost={bad!r}"):
                plan_any(3, 3, set(), (0, 0), [(2, 2)], max_cost=bad)

    def test_value_errors(self):
        for bad in (-1, -100):
            with self.assertRaises(ValueError, msg=f"max_cost={bad!r}"):
                plan(3, 3, set(), (0, 0), (2, 2), max_cost=bad)
            with self.assertRaises(ValueError, msg=f"max_cost={bad!r}"):
                plan_any(3, 3, set(), (0, 0), [(2, 2)], max_cost=bad)

    def test_validated_after_all_existing_checks(self):
        # Grid ValueError precedes the max_cost TypeError.
        with self.assertRaises(ValueError):
            plan(0, 3, set(), (0, 0), (1, 1), max_cost="x")
        # Obstacle ValueError precedes the max_cost TypeError.
        with self.assertRaises(ValueError):
            plan(3, 3, {(0, 0)}, (0, 0), (2, 2), max_cost="x")
        # Goal bounds ValueError precedes the max_cost TypeError.
        with self.assertRaises(ValueError):
            plan_any(3, 3, set(), (0, 0), [(9, 9)], max_cost="x")
        # Costs ValueError precedes the max_cost TypeError.
        with self.assertRaises(ValueError):
            plan(3, 3, set(), (0, 0), (2, 2),
                 costs=[[1, 1], [1, 1], [1, 1]], max_cost="x")
        # Trace TypeError precedes the max_cost ValueError.
        with self.assertRaises(TypeError):
            plan(3, 3, set(), (0, 0), (2, 2), trace=1, max_cost=-1)
        # Frame-0 ValueError precedes the max_cost TypeError.
        with self.assertRaises(ValueError):
            plan(3, 3, set(), (0, 0), (2, 2),
                 dynamic_blocked=[[(0, 0)]], max_cost="x")
        # Dynamic structure TypeError precedes the max_cost ValueError.
        with self.assertRaises(TypeError):
            plan(3, 3, set(), (0, 0), (2, 2), dynamic_blocked=42,
                 max_cost=-1)
        # The budget checks precede the max_cost checks.
        with self.assertRaises(TypeError) as cm:
            plan(3, 3, set(), (0, 0), (2, 2), max_expanded="x",
                 max_cost="y")
        self.assertIn("max_expanded", str(cm.exception))
        with self.assertRaises(ValueError) as cm:
            plan(3, 3, set(), (0, 0), (2, 2), max_expanded=-1, max_cost=-1)
        self.assertIn("max_expanded", str(cm.exception))
        # The snapshot check precedes the max_cost checks.
        with self.assertRaises(TypeError) as cm:
            plan(3, 3, set(), (0, 0), (2, 2), snapshot=1, max_cost="x")
        self.assertIn("snapshot", str(cm.exception))
        with self.assertRaises(TypeError) as cm:
            plan_any(3, 3, set(), (0, 0), [(2, 2)], snapshot=1, max_cost=-1)
        self.assertIn("snapshot", str(cm.exception))


class MaxCostStaticTest(unittest.TestCase):
    def test_generous_limit_matches_baseline_plus_status(self):
        baseline = plan(6, 4, BLOCKED, (0, 0), (5, 3), trace=True)
        limited = plan(6, 4, BLOCKED, (0, 0), (5, 3), trace=True,
                       max_cost=1000)
        self.assertEqual(limited, {**baseline, "status": "found"})

    def test_found_at_exact_limit(self):
        baseline = plan(3, 3, set(), (0, 0), (2, 2), trace=True)
        self.assertEqual(baseline["cost"], 4)
        limited = plan(3, 3, set(), (0, 0), (2, 2), trace=True, max_cost=4)
        self.assertEqual(limited, {**baseline, "status": "found"})

    def test_cost_exhausted(self):
        result = plan(3, 3, set(), (0, 0), (2, 2), trace=True, max_cost=3)
        self.assertIsNone(result["path"])
        self.assertIsNone(result["cost"])
        self.assertEqual(result["status"], "cost_exhausted")
        # Every cell but the goal is reachable within the limit.
        self.assertEqual(result["expanded"], 8)
        self.assertEqual(result["expanded"],
                         len(result["expanded_nodes"]))
        self.assertNotIn((2, 2), result["expanded_nodes"])

    def test_unreachable_vs_cost_exhausted(self):
        # No candidate ever exceeds the limit: plain unreachable.
        result = plan(3, 3, WALL, (0, 0), (2, 2), trace=True, max_cost=100)
        self.assertEqual(result["status"], "unreachable")
        self.assertEqual(result["expanded"], 3)
        # The only onward candidate exceeds the limit: cost_exhausted.
        costs = [[1, 1, 1], [5, 1, 1], [1, 1, 1]]
        pruned = plan(3, 3, WALL, (0, 0), (2, 2), costs=costs, max_cost=3)
        self.assertEqual(pruned, {"path": None, "cost": None,
                                  "expanded": 1, "status": "cost_exhausted"})

    def test_zero_limit(self):
        result = plan(3, 3, set(), (0, 0), (2, 2), trace=True, max_cost=0)
        self.assertEqual(result, {"path": None, "cost": None, "expanded": 1,
                                  "status": "cost_exhausted",
                                  "expanded_nodes": [(0, 0)]})

    def test_start_equals_goal(self):
        result = plan(3, 3, set(), (1, 1), (1, 1), trace=True, max_cost=0)
        self.assertEqual(result, {"path": [(1, 1)], "cost": 0,
                                  "expanded": 1, "status": "found",
                                  "expanded_nodes": [(1, 1)]})

    def test_budget_takes_priority(self):
        # Candidates are cost-discarded here, but the budget stop wins.
        result = plan(3, 3, set(), (0, 0), (2, 2), max_expanded=2,
                      max_cost=1)
        self.assertEqual(result, {"path": None, "cost": None, "expanded": 2,
                                  "status": "budget_exhausted"})
        # The zero-budget rule keeps priority even at start == goal.
        stopped = plan(3, 3, set(), (1, 1), (1, 1), max_expanded=0,
                       max_cost=0, trace=True)
        self.assertEqual(stopped, {"path": None, "cost": None,
                                   "expanded": 0,
                                   "status": "budget_exhausted",
                                   "expanded_nodes": []})

    def test_costs_matrix_limit(self):
        costs = [[1, 10, 1], [1, 10, 1], [1, 1, 1]]
        baseline = plan(3, 3, set(), (0, 0), (2, 0), costs=costs)
        self.assertEqual(baseline["cost"], 6)  # the detour around the 10s
        limited = plan(3, 3, set(), (0, 0), (2, 0), costs=costs, max_cost=6)
        self.assertEqual(limited, {**baseline, "status": "found"})
        tight = plan(3, 3, set(), (0, 0), (2, 0), costs=costs, max_cost=5)
        self.assertEqual(tight["status"], "cost_exhausted")
        self.assertIsNone(tight["path"])

    def test_found_path_replays(self):
        costs = [[1, 10, 1], [1, 10, 1], [1, 1, 1]]
        result = plan(3, 3, set(), (0, 0), (2, 0), costs=costs, max_cost=6)
        checked = replay(3, 3, set(), (0, 0), (2, 0), result["path"],
                         costs=costs)
        self.assertEqual(checked, {"valid": True, "cost": result["cost"],
                                   "steps": len(result["path"]) - 1})


class MaxCostDynamicTest(unittest.TestCase):
    def test_dynamic_found_at_limit_matches_baseline(self):
        baseline = plan(3, 3, set(), (1, 0), (2, 2),
                        dynamic_blocked=FRAMES, trace=True)
        limited = plan(3, 3, set(), (1, 0), (2, 2),
                       dynamic_blocked=FRAMES, trace=True,
                       max_cost=baseline["cost"])
        self.assertEqual(limited, {**baseline, "status": "found"})
        self.assertTrue(all(len(s) == 3
                            for s in limited["expanded_nodes"]))

    def test_dynamic_cost_exhausted(self):
        baseline = plan(3, 3, set(), (1, 0), (2, 2),
                        dynamic_blocked=FRAMES)
        tight = plan(3, 3, set(), (1, 0), (2, 2),
                     dynamic_blocked=FRAMES, trace=True,
                     max_cost=baseline["cost"] - 1)
        self.assertEqual(tight["status"], "cost_exhausted")
        self.assertIsNone(tight["path"])
        self.assertIsNone(tight["cost"])
        self.assertEqual(tight["expanded"], len(tight["expanded_nodes"]))
        # The dynamic rules still hold for every closed node.
        for x, y, t in tight["expanded_nodes"]:
            frame = FRAMES[t] if t < len(FRAMES) else FRAMES[-1]
            self.assertNotIn((x, y), frame)


class MaxCostPlanAnyTest(unittest.TestCase):
    COSTS = [[1, 9, 1], [1, 1, 1]]

    def test_limit_applies_to_plan_any(self):
        # (2, 0) costs 4 through the cheap row; (1, 0) costs 9 to enter.
        result = plan_any(3, 2, set(), (0, 0), [(1, 0), (2, 0)],
                          costs=self.COSTS, max_cost=5)
        self.assertEqual(result["status"], "found")
        self.assertEqual(result["path"][-1], (2, 0))
        self.assertEqual(result["cost"], 4)
        tight = plan_any(3, 2, set(), (0, 0), [(1, 0), (2, 0)],
                         costs=self.COSTS, max_cost=3)
        self.assertEqual(tight["status"], "cost_exhausted")
        self.assertIsNone(tight["path"])

    def test_minimum_cost_still_wins_under_limit(self):
        # The limit admits both endpoints; the cheaper one still wins.
        result = plan_any(3, 2, set(), (0, 0), [(1, 0), (2, 0)],
                          costs=self.COSTS, max_cost=9)
        self.assertEqual(result["path"][-1], (2, 0))
        self.assertEqual(result["cost"], 4)

    def test_generous_limit_matches_baseline_plus_status(self):
        baseline = plan_any(6, 4, BLOCKED, (0, 0), [(5, 3), (5, 2)],
                            trace=True)
        limited = plan_any(6, 4, BLOCKED, (0, 0), [(5, 3), (5, 2)],
                           trace=True, max_cost=1000)
        self.assertEqual(limited, {**baseline, "status": "found"})

    def test_start_in_goals_zero_limit(self):
        result = plan_any(3, 3, set(), (1, 1), [(2, 2), (1, 1)],
                          trace=True, max_cost=0)
        self.assertEqual(result, {"path": [(1, 1)], "cost": 0,
                                  "expanded": 1, "status": "found",
                                  "expanded_nodes": [(1, 1)]})

    def test_dynamic_cost_exhausted(self):
        baseline = plan_any(3, 3, set(), (1, 0), [(2, 2), (0, 2)],
                            dynamic_blocked=FRAMES)
        tight = plan_any(3, 3, set(), (1, 0), [(2, 2), (0, 2)],
                         dynamic_blocked=FRAMES,
                         max_cost=baseline["cost"] - 1)
        self.assertEqual(tight["status"], "cost_exhausted")
        self.assertIsNone(tight["path"])


class MaxCostSnapshotResumeTest(unittest.TestCase):
    def test_checkpoint_records_max_cost(self):
        result = plan(3, 3, set(), (0, 0), (2, 2), max_expanded=2,
                      snapshot=True, max_cost=3)
        self.assertEqual(result["status"], "budget_exhausted")
        checkpoint = result["checkpoint"]
        self.assertEqual(checkpoint["max_cost"], 3)
        self.assertIs(checkpoint["cost_pruned"], False)
        # The whole checkpoint is JSON-serializable as-is.
        self.assertEqual(json.loads(json.dumps(checkpoint)), checkpoint)

    def test_checkpoint_omits_field_without_limit(self):
        checkpoint = plan(6, 4, BLOCKED, (0, 0), (5, 3), max_expanded=3,
                          snapshot=True)["checkpoint"]
        self.assertNotIn("max_cost", checkpoint)
        self.assertNotIn("cost_pruned", checkpoint)

    def assert_resume_matches(self, call, args, kwargs):
        full = call(*args, trace=True, **kwargs)
        n = full["expanded"]
        for b1 in range(0, n + 2):
            first = call(*args, trace=True, max_expanded=b1, snapshot=True,
                         **kwargs)
            # snapshot=True never alters the legacy result itself.
            self.assertEqual(
                without_checkpoint(first),
                call(*args, trace=True, max_expanded=b1, **kwargs),
                f"b1={b1}",
            )
            if first["status"] != "budget_exhausted":
                self.assertNotIn("checkpoint", first)
                continue
            checkpoint = json.loads(json.dumps(first["checkpoint"]))
            self.assertEqual(checkpoint["max_cost"], kwargs["max_cost"])
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
                    # The chained checkpoint again resumes to the full
                    # unpaused result.
                    self.assertEqual(resume(resumed["checkpoint"]), full)
                else:
                    self.assertNotIn("checkpoint", resumed)

    def test_static_plan(self):
        self.assert_resume_matches(
            plan, (3, 3, set(), (0, 0), (2, 2)), {"max_cost": 3})

    def test_static_plan_with_costs(self):
        costs = [[1, 10, 1], [1, 10, 1], [1, 1, 1]]
        self.assert_resume_matches(
            plan, (3, 3, set(), (0, 0), (2, 0)),
            {"costs": costs, "max_cost": 5})

    def test_dynamic_plan(self):
        self.assert_resume_matches(
            plan, (3, 3, set(), (1, 0), (2, 2)),
            {"dynamic_blocked": FRAMES, "max_cost": 4})

    def test_plan_any_static(self):
        self.assert_resume_matches(
            plan_any, (6, 4, BLOCKED, (0, 0), [(5, 3), (5, 2)]),
            {"max_cost": 8})

    def test_plan_any_dynamic(self):
        self.assert_resume_matches(
            plan_any, (3, 3, set(), (1, 0), [(2, 2), (0, 2)]),
            {"dynamic_blocked": FRAMES, "max_cost": 2})

    def test_cost_pruned_persists_across_resume(self):
        # (1, 0) is a 9-cost dead end pruned while closing the start;
        # the goal is statically unreachable, so nothing is pruned after
        # the resume and only the checkpointed flag can report
        # cost_exhausted.
        blocked = {(2, 0), (1, 1)}
        costs = [[1, 9, 1], [1, 1, 1]]
        full = plan(3, 2, blocked, (0, 0), (2, 1), costs=costs,
                    max_cost=3, trace=True)
        self.assertEqual(full["status"], "cost_exhausted")
        first = plan(3, 2, blocked, (0, 0), (2, 1), costs=costs,
                     max_cost=3, max_expanded=1, snapshot=True)
        self.assertEqual(first["status"], "budget_exhausted")
        self.assertIs(first["checkpoint"]["cost_pruned"], True)
        checkpoint = json.loads(json.dumps(first["checkpoint"]))
        self.assertEqual(resume(checkpoint), full)

    def test_resume_missing_max_cost_field_is_compatible(self):
        checkpoint = plan(3, 3, set(), (0, 0), (2, 2), max_expanded=2,
                          snapshot=True, max_cost=3)["checkpoint"]
        legacy = json.loads(json.dumps(checkpoint))
        del legacy["max_cost"]
        del legacy["cost_pruned"]
        # Without the field the resume is an unlimited search.
        self.assertEqual(resume(legacy),
                         plan(3, 3, set(), (0, 0), (2, 2), trace=True))

    def test_resume_max_cost_validation(self):
        checkpoint = plan(3, 3, set(), (0, 0), (2, 2), max_expanded=2,
                          snapshot=True, max_cost=10)["checkpoint"]

        def mutated(mutate):
            raw = json.loads(json.dumps(checkpoint))
            mutate(raw)
            return raw

        for bad in ("5", 1.5, True, [5], (5,)):
            with self.assertRaises(TypeError, msg=f"max_cost={bad!r}"):
                resume(mutated(lambda c, b=bad: c.update(max_cost=b)))
        for bad in (-1, -100):
            with self.assertRaises(ValueError, msg=f"max_cost={bad!r}"):
                resume(mutated(lambda c, b=bad: c.update(max_cost=b)))
        # A limit the recorded state already exceeds is inconsistent.
        with self.assertRaises(ValueError):
            resume(mutated(lambda c: c.update(max_cost=0)))
        # cost_pruned must be a bool and requires a max_cost.
        with self.assertRaises(TypeError):
            resume(mutated(lambda c: c.update(cost_pruned=1)))
        with self.assertRaises(ValueError):
            resume(mutated(
                lambda c: (c.pop("max_cost"), c.update(cost_pruned=True))
            ))

    def test_resumed_limited_path_replays(self):
        first = plan(3, 3, set(), (1, 0), (2, 2), dynamic_blocked=FRAMES,
                     max_cost=5, max_expanded=2, snapshot=True)
        resumed = resume(first["checkpoint"])
        self.assertEqual(resumed["status"], "found")
        checked = replay(3, 3, set(), (1, 0), (2, 2), resumed["path"],
                         dynamic_blocked=FRAMES)
        self.assertEqual(checked, {
            "valid": True,
            "cost": resumed["cost"],
            "steps": len(resumed["path"]) - 1,
        })


class MaxCostReplayTest(unittest.TestCase):
    def test_replay_accepts_no_max_cost(self):
        with self.assertRaises(TypeError):
            replay(3, 3, set(), (0, 0), (2, 2),
                   [(0, 0), (1, 1), (2, 2)], max_cost=3)

    def test_replay_behavior_unchanged(self):
        path = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]
        self.assertEqual(replay(3, 3, set(), (0, 0), (2, 2), path),
                         {"valid": True, "cost": 4, "steps": 4})


if __name__ == '__main__':
    unittest.main()
