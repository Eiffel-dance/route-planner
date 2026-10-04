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


def without_checkpoint(result):
    return {key: value for key, value in result.items()
            if key != "checkpoint"}


class SnapshotOptionTest(unittest.TestCase):
    def test_omitted_or_false_keeps_legacy_shape(self):
        omitted = plan(6, 4, BLOCKED, (0, 0), (5, 3), trace=True,
                       max_expanded=3)
        explicit = plan(6, 4, BLOCKED, (0, 0), (5, 3), trace=True,
                        max_expanded=3, snapshot=False)
        self.assertEqual(explicit, omitted)
        self.assertEqual(omitted["status"], "budget_exhausted")
        self.assertNotIn("checkpoint", omitted)

    def test_snapshot_without_budget_never_snapshots(self):
        legacy = plan(6, 4, BLOCKED, (0, 0), (5, 3), trace=True)
        flagged = plan(6, 4, BLOCKED, (0, 0), (5, 3), trace=True,
                       snapshot=True)
        self.assertEqual(flagged, legacy)
        self.assertNotIn("checkpoint", flagged)

    def test_budget_exhausted_carries_checkpoint(self):
        result = plan(6, 4, BLOCKED, (0, 0), (5, 3), trace=True,
                      max_expanded=3, snapshot=True)
        self.assertEqual(result["status"], "budget_exhausted")
        checkpoint = result["checkpoint"]
        self.assertEqual(checkpoint["planner"], "plan")
        self.assertEqual(checkpoint["closed"], result["expanded"])
        self.assertEqual(checkpoint["trace"],
                         [list(p) for p in result["expanded_nodes"]])
        # The whole checkpoint is JSON-serializable as-is.
        self.assertEqual(json.loads(json.dumps(checkpoint)), checkpoint)

    def test_found_and_unreachable_carry_no_checkpoint(self):
        found = plan(6, 4, BLOCKED, (0, 0), (5, 3), max_expanded=1000,
                     snapshot=True)
        self.assertEqual(found["status"], "found")
        self.assertNotIn("checkpoint", found)
        wall = {(1, y) for y in range(3)}
        unreachable = plan(3, 3, wall, (0, 0), (2, 2), max_expanded=10,
                           snapshot=True)
        self.assertEqual(unreachable["status"], "unreachable")
        self.assertNotIn("checkpoint", unreachable)

    def test_checkpoint_omitted_without_trace_in_result(self):
        # trace=False keeps expanded_nodes out of the result, but the
        # checkpoint still records the trace internally.
        result = plan(6, 4, BLOCKED, (0, 0), (5, 3), max_expanded=3,
                      snapshot=True)
        self.assertNotIn("expanded_nodes", result)
        self.assertEqual(len(result["checkpoint"]["trace"]), 3)

    def test_snapshot_type_errors(self):
        for bad in (1, 0, "true", None, 1.0, [True]):
            with self.assertRaises(TypeError, msg=f"snapshot={bad!r}"):
                plan(3, 3, set(), (0, 0), (2, 2), snapshot=bad)
            with self.assertRaises(TypeError, msg=f"snapshot={bad!r}"):
                plan_any(3, 3, set(), (0, 0), [(2, 2)], snapshot=bad)

    def test_snapshot_validated_after_all_existing_checks(self):
        # Grid ValueError precedes the snapshot TypeError.
        with self.assertRaises(ValueError):
            plan(0, 3, set(), (0, 0), (1, 1), snapshot=1)
        # The budget TypeError precedes the snapshot TypeError.
        with self.assertRaises(TypeError):
            plan(3, 3, set(), (0, 0), (2, 2), max_expanded="x", snapshot=1)
        # The budget ValueError precedes the snapshot TypeError.
        with self.assertRaises(ValueError):
            plan_any(3, 3, set(), (0, 0), [(2, 2)], max_expanded=-1,
                     snapshot=1)

    def test_checkpoint_ordering_determinism(self):
        cells = [(x, 1) for x in range(5) if x != 2]
        checkpoints = [
            plan(5, 3, order, (0, 0), (4, 2), max_expanded=6,
                 snapshot=True)["checkpoint"]
            for order in (cells, list(reversed(cells)),
                          [cells[2], cells[0], cells[3], cells[1]])
        ]
        self.assertEqual(checkpoints[0], checkpoints[1])
        self.assertEqual(checkpoints[0], checkpoints[2])
        # Frame-internal ordering does not affect the checkpoint either.
        frames_a = [[(0, 1), (0, 2)], [(2, 0)]]
        frames_b = [[(0, 2), (0, 1)], [(2, 0)]]
        cp_a = plan(3, 3, set(), (1, 0), (2, 2), dynamic_blocked=frames_a,
                    max_expanded=2, snapshot=True)["checkpoint"]
        cp_b = plan(3, 3, set(), (1, 0), (2, 2), dynamic_blocked=frames_b,
                    max_expanded=2, snapshot=True)["checkpoint"]
        self.assertEqual(cp_a, cp_b)


class ResumeEquivalenceTest(unittest.TestCase):
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
            plan, (6, 4, BLOCKED, (0, 0), (5, 3)), {})

    def test_static_plan_with_costs(self):
        costs = [[1, 10, 1], [1, 10, 1], [1, 1, 1]]
        self.assert_resume_matches(
            plan, (3, 3, set(), (0, 0), (2, 0)), {"costs": costs})

    def test_dynamic_plan(self):
        self.assert_resume_matches(
            plan, (3, 3, set(), (1, 0), (2, 2)),
            {"dynamic_blocked": FRAMES})

    def test_plan_any_static(self):
        self.assert_resume_matches(
            plan_any, (6, 4, BLOCKED, (0, 0), [(5, 3), (5, 2)]), {})

    def test_plan_any_dynamic(self):
        self.assert_resume_matches(
            plan_any, (3, 3, set(), (1, 0), [(2, 2), (0, 2)]),
            {"dynamic_blocked": FRAMES})

    def test_resume_result_always_carries_trace(self):
        first = plan(6, 4, BLOCKED, (0, 0), (5, 3), max_expanded=3,
                     snapshot=True)
        resumed = resume(first["checkpoint"])
        self.assertIn("expanded_nodes", resumed)
        self.assertEqual(resumed["expanded_nodes"],
                         plan(6, 4, BLOCKED, (0, 0), (5, 3),
                              trace=True)["expanded_nodes"])

    def test_resume_without_budget_has_no_status(self):
        first = plan(6, 4, BLOCKED, (0, 0), (5, 3), max_expanded=3,
                     snapshot=True)
        resumed = resume(first["checkpoint"])
        self.assertNotIn("status", resumed)
        self.assertNotIn("checkpoint", resumed)

    def test_limit_not_above_closed_count_closes_nothing(self):
        first = plan(6, 4, BLOCKED, (0, 0), (5, 3), trace=True,
                     max_expanded=3, snapshot=True)
        checkpoint = first["checkpoint"]
        for budget in (0, 2, 3):
            resumed = resume(checkpoint, max_expanded=budget)
            self.assertEqual(resumed["status"], "budget_exhausted")
            self.assertEqual(resumed["expanded"], 3)
            self.assertEqual(resumed["expanded_nodes"],
                             first["expanded_nodes"])
            self.assertIn("checkpoint", resumed)

    def test_repeated_resumes_do_not_double_count(self):
        first = plan(6, 4, BLOCKED, (0, 0), (5, 3), max_expanded=2,
                     snapshot=True)
        second = resume(first["checkpoint"], max_expanded=5)
        third = resume(second["checkpoint"], max_expanded=9)
        direct = plan(6, 4, BLOCKED, (0, 0), (5, 3), trace=True,
                      max_expanded=9, snapshot=True)
        self.assertEqual(without_checkpoint(third),
                         without_checkpoint(direct))
        self.assertEqual(third["expanded"], direct["expanded"])

    def test_zero_budget_start_equals_goal_snapshots(self):
        first = plan(3, 3, set(), (1, 1), (1, 1), max_expanded=0,
                     snapshot=True)
        self.assertEqual(first["status"], "budget_exhausted")
        self.assertIn("checkpoint", first)
        resumed = resume(first["checkpoint"], max_expanded=1)
        self.assertEqual(resumed, {"path": [(1, 1)], "cost": 0,
                                   "expanded": 1, "status": "found",
                                   "expanded_nodes": [(1, 1)]})

    def test_resumed_path_replays(self):
        first = plan(3, 3, set(), (1, 0), (2, 2), dynamic_blocked=FRAMES,
                     max_expanded=2, snapshot=True)
        resumed = resume(first["checkpoint"])
        self.assertIsNotNone(resumed["path"])
        checked = replay(3, 3, set(), (1, 0), (2, 2), resumed["path"],
                         dynamic_blocked=FRAMES)
        self.assertEqual(checked, {
            "valid": True,
            "cost": resumed["cost"],
            "steps": len(resumed["path"]) - 1,
        })

    def test_resumed_any_path_replays_via_endpoint(self):
        first = plan_any(6, 4, BLOCKED, (0, 0), [(5, 3), (5, 2)],
                         max_expanded=4, snapshot=True)
        resumed = resume(first["checkpoint"])
        goal = resumed["path"][-1]
        checked = replay(6, 4, BLOCKED, (0, 0), goal, resumed["path"])
        self.assertEqual(checked, {
            "valid": True,
            "cost": resumed["cost"],
            "steps": len(resumed["path"]) - 1,
        })


class ResumeValidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checkpoint = plan(6, 4, BLOCKED, (0, 0), (5, 3), max_expanded=3,
                              snapshot=True)["checkpoint"]

    def mutated(self, mutate):
        checkpoint = json.loads(json.dumps(self.checkpoint))
        mutate(checkpoint)
        return checkpoint

    def test_non_object_checkpoint_is_type_error(self):
        for bad in (None, 42, 1.5, "checkpoint", b"checkpoint", [1], (1,)):
            with self.assertRaises(TypeError, msg=f"checkpoint={bad!r}"):
                resume(bad)

    def test_missing_required_fields_are_type_errors(self):
        for key in ("version", "planner", "width", "height", "blocked",
                    "start", "goal", "costs", "dynamic_blocked", "closed",
                    "trace", "state"):
            with self.assertRaises(TypeError, msg=f"missing {key}"):
                resume(self.mutated(lambda c, k=key: c.pop(k)))

    def test_field_type_errors(self):
        cases = [
            lambda c: c.update(version="1"),
            lambda c: c.update(version=True),
            lambda c: c.update(planner=1),
            lambda c: c.update(width="6"),
            lambda c: c.update(height=4.0),
            lambda c: c.update(blocked="x"),
            lambda c: c.update(start=[0]),
            lambda c: c.update(goal=(0, 0, 0)),
            lambda c: c.update(costs=[[1.5] * 6] * 4),
            lambda c: c.update(dynamic_blocked=42),
            lambda c: c.update(closed=True),
            lambda c: c.update(closed="3"),
            lambda c: c.update(trace="x"),
            lambda c: c.update(trace=[[0, 0, 0]]),
            lambda c: c.update(trace=[[0, "x"]]),
            lambda c: c.update(state=[]),
            lambda c: c["state"].update(open=None),
            lambda c: c["state"].update(closed=[[0]]),
        ]
        for mutate in cases:
            with self.assertRaises(TypeError, msg=mutate):
                resume(self.mutated(mutate))

    def test_semantic_errors_are_value_errors(self):
        cases = [
            lambda c: c.update(version=2),
            lambda c: c.update(planner="astar"),
            lambda c: c.update(width=0),
            lambda c: c.update(start=[9, 9]),
            lambda c: c.update(goal=[2, 0]),      # goal on obstacle
            lambda c: c.update(blocked=[[0, 0]]),  # start on obstacle
            lambda c: c.update(costs=[[1] * 6] * 3),  # wrong row count
            lambda c: c.update(closed=-1),
            lambda c: c.update(closed=2),          # trace length mismatch
            lambda c: c.update(trace=[]),
            lambda c: c["state"].update(open=[]),  # no pending candidates
            lambda c: c["state"].update(closed=[]),
            lambda c: c["state"]["g_score"].append([[9, 9], 3]),
        ]
        for mutate in cases:
            with self.assertRaises(ValueError, msg=mutate):
                resume(self.mutated(mutate))

    def test_errors_are_decided_before_any_search(self):
        # A checkpoint with an inconsistent state raises even when the
        # budget would allow zero further closings anyway.
        bad = self.mutated(lambda c: c["state"].update(open=[]))
        with self.assertRaises(ValueError):
            resume(bad, max_expanded=0)

    def test_budget_rules_match_plan(self):
        for bad in ("5", 1.5, True, [5], (5,)):
            with self.assertRaises(TypeError, msg=f"max_expanded={bad!r}"):
                resume(self.checkpoint, max_expanded=bad)
        for bad in (-1, -100):
            with self.assertRaises(ValueError, msg=f"max_expanded={bad!r}"):
                resume(self.checkpoint, max_expanded=bad)

    def test_plan_any_checkpoint_round_trip(self):
        first = plan_any(3, 3, set(), (1, 0), [(2, 2), (0, 2)],
                         dynamic_blocked=FRAMES, max_expanded=2,
                         snapshot=True)
        checkpoint = first["checkpoint"]
        self.assertEqual(checkpoint["planner"], "plan_any")
        self.assertEqual(checkpoint["goals"], [[0, 2], [2, 2]])
        resumed = resume(json.loads(json.dumps(checkpoint)))
        self.assertEqual(resumed["path"],
                         plan_any(3, 3, set(), (1, 0), [(2, 2), (0, 2)],
                                  dynamic_blocked=FRAMES)["path"])


if __name__ == '__main__':
    unittest.main()
