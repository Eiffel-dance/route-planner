import json
import unittest

from app import plan_any, replay, resume, verify_any_trace


BLOCKED = {(2, 0), (2, 1), (2, 2)}
WALL = {(1, y) for y in range(3)}
FRAMES = [
    [(0, 1), (0, 2)],
    [(0, 1), (1, 2), (2, 0), (2, 2)],
    [(2, 1), (2, 2)],
    [(0, 0), (1, 2), (2, 1), (2, 2)],
    [],
    [(0, 1)],
]
GOALS = [(5, 3), (5, 2)]


def round_trip(result):
    return json.loads(json.dumps(result))


def audit(goals, record, **kwargs):
    return verify_any_trace(6, 4, BLOCKED, (0, 0), goals, record, **kwargs)


class VerifyAnyTraceStaticTest(unittest.TestCase):
    def test_valid_record_direct(self):
        record = plan_any(6, 4, BLOCKED, (0, 0), GOALS, trace=True)
        self.assertEqual(
            audit(GOALS, record),
            {"valid": True, "mismatch": None, "index": None},
        )

    def test_valid_record_json_round_trip(self):
        record = round_trip(plan_any(6, 4, BLOCKED, (0, 0), GOALS, trace=True))
        self.assertTrue(audit(GOALS, record)["valid"])

    def test_mixed_sequences_and_extra_keys(self):
        record = plan_any(6, 4, BLOCKED, (0, 0), GOALS, trace=True)
        record["path"] = [list(point) for point in record["path"]]
        record["expanded_nodes"] = [
            list(entry) for entry in record["expanded_nodes"]
        ]
        record["something_else"] = {"ignored": True}
        self.assertTrue(audit(GOALS, record)["valid"])

    def test_duplicate_goals_and_input_order_do_not_matter(self):
        record = round_trip(plan_any(6, 4, BLOCKED, (0, 0), GOALS, trace=True))
        shuffled = [GOALS[1], GOALS[0], GOALS[0], GOALS[1]]
        self.assertTrue(audit(shuffled, record)["valid"])

    def test_unreachable_record(self):
        record = plan_any(3, 3, WALL, (0, 0), [(2, 2), (2, 0)], trace=True)
        self.assertIsNone(record["path"])
        self.assertIsNone(record["cost"])
        self.assertTrue(
            verify_any_trace(3, 3, WALL, (0, 0), [(2, 0), (2, 2)], record)
            ["valid"]
        )

    def test_start_in_goals_single_point_record(self):
        record = plan_any(3, 3, set(), (1, 1), [(2, 2), (1, 1), (0, 0)],
                          trace=True)
        self.assertEqual(record, {"path": [(1, 1)], "cost": 0, "expanded": 1,
                                  "expanded_nodes": [(1, 1)]})
        self.assertTrue(
            verify_any_trace(3, 3, set(), (1, 1), [(0, 0), (1, 1), (2, 2)],
                             record)["valid"]
        )

    def test_costs_matrix_record(self):
        costs = [[1, 2, 1], [3, 1, 4], [1, 1, 1]]
        record = plan_any(3, 3, set(), (0, 0), [(2, 2), (2, 0)],
                          costs=costs, trace=True)
        self.assertTrue(
            verify_any_trace(3, 3, set(), (0, 0), [(2, 0), (2, 2)], record,
                             costs=costs)["valid"]
        )
        # The same record under different costs no longer matches.
        other = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        report = verify_any_trace(3, 3, set(), (0, 0), [(2, 2)], record,
                                  costs=other)
        self.assertFalse(report["valid"])

    def test_static_trace_may_repeat_coordinates(self):
        # plan_any's route-tree search closes a coordinate through distinct
        # histories even in static mode, so repeated pairs in
        # expanded_nodes are legitimate (unlike verify_trace's plan audit).
        record = plan_any(3, 3, set(), (0, 0), [(2, 2)], trace=True)
        nodes = record["expanded_nodes"]
        self.assertNotEqual(len(nodes), len(set(nodes)))
        self.assertEqual(record["expanded"], len(nodes))
        self.assertTrue(
            verify_any_trace(3, 3, set(), (0, 0), [(2, 2)],
                             round_trip(record))["valid"]
        )

    def test_result_is_json_serializable(self):
        record = round_trip(plan_any(6, 4, BLOCKED, (0, 0), GOALS, trace=True))
        report = audit(GOALS, record)
        self.assertEqual(json.loads(json.dumps(report)), report)
        record["cost"] += 1
        report = audit(GOALS, record)
        self.assertEqual(json.loads(json.dumps(report)), report)

    def test_record_is_not_modified(self):
        record = round_trip(plan_any(6, 4, BLOCKED, (0, 0), GOALS, trace=True))
        before = json.dumps(record, sort_keys=True)
        audit(GOALS, record)
        self.assertEqual(json.dumps(record, sort_keys=True), before)

    def test_success_record_replays_with_its_endpoint(self):
        record = plan_any(6, 4, BLOCKED, (0, 0), GOALS, trace=True)
        goal = record["path"][-1]
        checked = replay(6, 4, BLOCKED, (0, 0), goal, record["path"])
        self.assertTrue(checked["valid"])
        self.assertEqual(checked["cost"], record["cost"])
        self.assertEqual(checked["steps"], len(record["path"]) - 1)


class VerifyAnyTraceMismatchTest(unittest.TestCase):
    def setUp(self):
        self.record = round_trip(
            plan_any(6, 4, BLOCKED, (0, 0), GOALS, trace=True)
        )

    def test_path_mismatch_index(self):
        self.record["path"][2] = list(self.record["path"][1])
        self.assertEqual(
            audit(GOALS, self.record),
            {"valid": False, "mismatch": "path", "index": 2},
        )

    def test_path_length_mismatch_index(self):
        del self.record["path"][2:]
        self.assertEqual(
            audit(GOALS, self.record),
            {"valid": False, "mismatch": "path", "index": 2},
        )

    def test_path_null_against_route(self):
        self.record["path"] = None
        self.assertEqual(
            audit(GOALS, self.record),
            {"valid": False, "mismatch": "path", "index": 0},
        )

    def test_cost_mismatch(self):
        self.record["cost"] += 1
        self.assertEqual(
            audit(GOALS, self.record),
            {"valid": False, "mismatch": "cost", "index": None},
        )

    def test_expanded_mismatch(self):
        self.record["expanded"] += 1
        self.assertEqual(
            audit(GOALS, self.record),
            {"valid": False, "mismatch": "expanded", "index": None},
        )

    def test_expanded_nodes_mismatch_index(self):
        nodes = self.record["expanded_nodes"]
        self.assertNotEqual(nodes[1], nodes[2])
        nodes[1], nodes[2] = nodes[2], nodes[1]
        self.assertEqual(
            audit(GOALS, self.record),
            {"valid": False, "mismatch": "expanded_nodes", "index": 1},
        )

    def test_expanded_nodes_prefix_mismatch(self):
        del self.record["expanded_nodes"][-1]
        report = audit(GOALS, self.record)
        self.assertEqual(report["mismatch"], "expanded_nodes")
        self.assertEqual(report["index"],
                         len(self.record["expanded_nodes"]))

    def test_mismatch_order_path_before_cost(self):
        self.record["path"][1] = list(self.record["path"][0])
        self.record["cost"] += 5
        self.assertEqual(audit(GOALS, self.record)["mismatch"], "path")

    def test_mismatch_order_expanded_before_nodes(self):
        self.record["expanded"] += 1
        self.record["expanded_nodes"][0] = [1, 0]
        self.assertEqual(audit(GOALS, self.record)["mismatch"], "expanded")

    def test_status_mismatch(self):
        record = round_trip(plan_any(3, 3, WALL, (0, 0), [(2, 2)],
                                    max_expanded=2, trace=True))
        self.assertEqual(record["status"], "budget_exhausted")
        record["status"] = "found"
        report = verify_any_trace(3, 3, WALL, (0, 0), [(2, 2)], record,
                                  max_expanded=2)
        self.assertEqual(
            report, {"valid": False, "mismatch": "status", "index": None}
        )

    def test_status_missing_from_record(self):
        record = round_trip(plan_any(3, 3, WALL, (0, 0), [(2, 2)],
                                    max_expanded=2, trace=True))
        del record["status"]
        report = verify_any_trace(3, 3, WALL, (0, 0), [(2, 2)], record,
                                  max_expanded=2)
        self.assertEqual(report["mismatch"], "status")

    def test_extra_status_ignored_without_budget(self):
        self.record["status"] = "found"
        report = audit(GOALS, self.record)
        self.assertTrue(report["valid"])

    def test_checkpoint_mismatch(self):
        record = round_trip(plan_any(3, 3, WALL, (0, 0), [(2, 2)],
                                    max_expanded=2, snapshot=True, trace=True))
        self.assertIn("checkpoint", record)
        # A structurally valid but different snapshot: unit costs spelled
        # out as a matrix instead of the recorded null.
        record["checkpoint"]["costs"] = [[1] * 3 for _ in range(3)]
        report = verify_any_trace(3, 3, WALL, (0, 0), [(2, 2)], record,
                                  max_expanded=2, snapshot=True)
        self.assertEqual(
            report,
            {"valid": False, "mismatch": "checkpoint", "index": None},
        )

    def test_checkpoint_missing_from_record(self):
        record = round_trip(plan_any(3, 3, WALL, (0, 0), [(2, 2)],
                                    max_expanded=2, snapshot=True, trace=True))
        del record["checkpoint"]
        report = verify_any_trace(3, 3, WALL, (0, 0), [(2, 2)], record,
                                  max_expanded=2, snapshot=True)
        self.assertEqual(report["mismatch"], "checkpoint")

    def test_checkpoint_valid(self):
        record = round_trip(plan_any(3, 3, WALL, (0, 0), [(2, 2)],
                                    max_expanded=2, snapshot=True, trace=True))
        report = verify_any_trace(3, 3, WALL, (0, 0), [(2, 2)], record,
                                  max_expanded=2, snapshot=True)
        self.assertTrue(report["valid"])

    def test_extra_checkpoint_ignored_without_snapshot(self):
        record = round_trip(plan_any(3, 3, WALL, (0, 0), [(2, 2)],
                                    max_expanded=2, snapshot=True, trace=True))
        report = verify_any_trace(3, 3, WALL, (0, 0), [(2, 2)], record,
                                  max_expanded=2)
        self.assertTrue(report["valid"])


class VerifyAnyTraceValidationTest(unittest.TestCase):
    def setUp(self):
        self.record = round_trip(
            plan_any(6, 4, BLOCKED, (0, 0), GOALS, trace=True)
        )

    def test_grid_errors_precede_record_errors(self):
        # A bad width raises before the missing-field record error.
        with self.assertRaises(TypeError):
            verify_any_trace("6", 4, BLOCKED, (0, 0), GOALS, {})
        # A bad start raises before the record is inspected.
        with self.assertRaises(ValueError):
            verify_any_trace(6, 4, BLOCKED, (0, 0), [(9, 9)], None)
        # A blocked goal raises before the record is inspected.
        with self.assertRaises(ValueError):
            verify_any_trace(6, 4, {(2, 0)}, (0, 0), [(2, 0)], "record")
        # A bad budget raises before the record is inspected.
        with self.assertRaises(ValueError):
            verify_any_trace(6, 4, BLOCKED, (0, 0), GOALS, [],
                             max_expanded=-1)
        # A bad reservation raises before the record is inspected.
        with self.assertRaises(ValueError):
            verify_any_trace(6, 4, BLOCKED, (0, 0), GOALS, [],
                             reservations=[[(2, 0)]])

    def test_goals_errors_precede_record_errors(self):
        # Non-sequence / empty goals fail before the (also bad) record.
        with self.assertRaises(TypeError):
            verify_any_trace(6, 4, BLOCKED, (0, 0), None, {})
        with self.assertRaises(ValueError):
            verify_any_trace(6, 4, BLOCKED, (0, 0), [], None)
        # An out-of-range or blocked goal fails before the record.
        with self.assertRaises(ValueError):
            verify_any_trace(6, 4, BLOCKED, (0, 0), [(9, 9)], None)
        with self.assertRaises(ValueError):
            verify_any_trace(6, 4, {(2, 2)}, (0, 0), [(2, 2)], None)
        # A malformed goal element fails before the record.
        with self.assertRaises(TypeError):
            verify_any_trace(6, 4, BLOCKED, (0, 0), [[1]], None)
        # A start blocked at frame 0 fails before the record.
        with self.assertRaises(ValueError):
            verify_any_trace(6, 4, BLOCKED, (0, 0), GOALS, None,
                             dynamic_blocked=[[(0, 0)]])

    def test_failed_validation_does_not_start_a_search(self):
        # A valid record cannot mask a preceding validation failure.
        record = self.record
        with self.assertRaises(ValueError):
            verify_any_trace(6, 4, BLOCKED, (0, 0), [(9, 9)], record)
        with self.assertRaises(TypeError):
            verify_any_trace(6, 4, BLOCKED, (0, 0), "goals", record)

    def test_record_not_an_object(self):
        for bad in (None, [], "record", 42, True):
            with self.assertRaises(TypeError):
                audit(GOALS, bad)

    def test_record_missing_fields(self):
        for key in ("path", "cost", "expanded", "expanded_nodes"):
            record = dict(self.record)
            del record[key]
            with self.assertRaises(TypeError):
                audit(GOALS, record)

    def test_record_field_type_errors(self):
        for key, bad in (("path", "path"), ("cost", "cost"),
                         ("expanded", True), ("expanded_nodes", 7)):
            record = dict(self.record)
            record[key] = bad
            with self.assertRaises(TypeError):
                audit(GOALS, record)

    def test_record_path_shape_errors(self):
        for bad_entry in ("xy", [1], [1, 2, 3], [1, "y"], [True, 0]):
            record = dict(self.record)
            record["path"] = [bad_entry]
            with self.assertRaises(TypeError):
                audit(GOALS, record)

    def test_record_trajectory_shape_errors(self):
        for bad_entry in ("xyt", [1], [1, 2, 3, 4], [1, 2, "t"],
                          [True, 0, 0]):
            record = dict(self.record)
            record["expanded_nodes"] = [bad_entry]
            with self.assertRaises(TypeError):
                audit(GOALS, record)

    def test_record_coordinate_out_of_bounds(self):
        record = dict(self.record)
        record["path"] = [[9, 0]]
        with self.assertRaises(ValueError):
            audit(GOALS, record)
        record = dict(self.record)
        record["expanded_nodes"] = [[0, 9]]
        with self.assertRaises(ValueError):
            audit(GOALS, record)

    def test_record_trajectory_mode_mismatch(self):
        # A triple in static mode is a ValueError, not a mismatch.
        record = dict(self.record)
        record["expanded_nodes"] = [[0, 0, 0]]
        with self.assertRaises(ValueError):
            audit(GOALS, record)
        # A pair in dynamic mode is a ValueError too.
        dynamic = round_trip(plan_any(6, 4, BLOCKED, (0, 0), GOALS,
                                     dynamic_blocked=FRAMES, trace=True))
        dynamic["expanded_nodes"] = [[0, 0]]
        with self.assertRaises(ValueError):
            audit(GOALS, dynamic, dynamic_blocked=FRAMES)

    def test_record_dynamic_negative_time(self):
        record = round_trip(plan_any(6, 4, BLOCKED, (0, 0), GOALS,
                                    dynamic_blocked=FRAMES, trace=True))
        record["expanded_nodes"][0] = [0, 0, -1]
        with self.assertRaises(ValueError):
            audit(GOALS, record, dynamic_blocked=FRAMES)

    def test_record_static_duplicate_state_is_allowed(self):
        # Distinct histories close the same cell: appending another
        # occurrence stays structurally valid and is reported as a content
        # mismatch rather than raised.
        record = round_trip(plan_any(3, 3, set(), (0, 0), [(2, 2)],
                                    trace=True))
        record["expanded_nodes"] = (record["expanded_nodes"]
                                    + [record["expanded_nodes"][0]])
        report = verify_any_trace(3, 3, set(), (0, 0), [(2, 2)], record)
        self.assertFalse(report["valid"])
        self.assertEqual(report["mismatch"], "expanded_nodes")

    def test_record_bad_checkpoint(self):
        record = dict(self.record)
        record["checkpoint"] = {"version": 1}
        with self.assertRaises(ValueError):
            audit(GOALS, record)
        record = dict(self.record)
        record["checkpoint"] = 42
        with self.assertRaises(ValueError):
            audit(GOALS, record)

    def test_record_bad_status_type(self):
        record = dict(self.record)
        record["status"] = 7
        with self.assertRaises(TypeError):
            audit(GOALS, record)


class VerifyAnyTraceModesTest(unittest.TestCase):
    def test_dynamic_blocked(self):
        record = plan_any(3, 3, set(), (1, 0), [(2, 2), (0, 2)],
                          dynamic_blocked=FRAMES, trace=True)
        goals = [(0, 2), (2, 2)]
        self.assertTrue(
            verify_any_trace(3, 3, set(), (1, 0), goals, record,
                             dynamic_blocked=FRAMES)["valid"]
        )
        self.assertTrue(
            verify_any_trace(3, 3, set(), (1, 0), goals, round_trip(record),
                             dynamic_blocked=FRAMES)["valid"]
        )

    def test_dynamic_last_frame_persists(self):
        # The candidate is blocked at frame 1 only; the detour reaches it
        # later, when the last supplied frame (the empty frame) persists.
        frames = [[], [(1, 0)], []]
        record = plan_any(3, 2, set(), (0, 0), [(1, 0), (2, 0)],
                          dynamic_blocked=frames, trace=True)
        self.assertEqual(record["path"],
                         [(0, 0), (0, 1), (1, 1), (1, 0)])
        self.assertTrue(
            verify_any_trace(3, 2, set(), (0, 0), [(2, 0), (1, 0)], record,
                             dynamic_blocked=frames)["valid"]
        )
        # A candidate only blocked in the persistent last frame is
        # unreachable; the null result still audits.
        persistent = [[], [(2, 0)]]
        none = plan_any(3, 1, set(), (0, 0), [(2, 0)],
                        dynamic_blocked=persistent, trace=True)
        self.assertIsNone(none["path"])
        self.assertTrue(
            verify_any_trace(3, 1, set(), (0, 0), [(2, 0)], none,
                             dynamic_blocked=persistent)["valid"]
        )

    def test_dynamic_trace_mismatch_index(self):
        record = round_trip(plan_any(3, 3, set(), (1, 0), [(2, 2), (0, 2)],
                                    dynamic_blocked=FRAMES, trace=True))
        record["expanded_nodes"][0] = [1, 0, 1]
        report = verify_any_trace(3, 3, set(), (1, 0), [(2, 2), (0, 2)],
                                  record, dynamic_blocked=FRAMES)
        self.assertEqual(report["mismatch"], "expanded_nodes")
        self.assertEqual(report["index"], 0)

    def test_allow_wait(self):
        record = plan_any(6, 4, BLOCKED, (0, 0), GOALS,
                          dynamic_blocked=FRAMES, allow_wait=True, trace=True)
        self.assertTrue(
            verify_any_trace(6, 4, BLOCKED, (0, 0), GOALS, record,
                             dynamic_blocked=FRAMES,
                             allow_wait=True)["valid"]
        )
        # Recomputing without waiting may select a different route.
        report = verify_any_trace(6, 4, BLOCKED, (0, 0), GOALS,
                                  round_trip(record),
                                  dynamic_blocked=FRAMES)
        self.assertFalse(report["valid"])

    def test_dynamic_costs(self):
        frames = [
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [[2, 2, 2], [2, 1, 2], [2, 2, 2]],
        ]
        record = plan_any(3, 3, set(), (0, 0), [(2, 2), (2, 0)],
                          dynamic_costs=frames, trace=True)
        self.assertTrue(
            verify_any_trace(3, 3, set(), (0, 0), [(2, 0), (2, 2)], record,
                             dynamic_costs=frames)["valid"]
        )

    def test_reservations(self):
        reservations = [[(1, 0), (1, 1), (1, 2)]]
        record = plan_any(3, 3, set(), (0, 0), [(2, 2), (2, 0)],
                          reservations=reservations, trace=True)
        self.assertTrue(all(len(node) == 3 for node
                            in record["expanded_nodes"]))
        self.assertTrue(
            verify_any_trace(3, 3, set(), (0, 0), [(2, 0), (2, 2)], record,
                             reservations=reservations)["valid"]
        )

    def test_max_cost(self):
        record = plan_any(6, 4, BLOCKED, (0, 0), GOALS, max_cost=4,
                          trace=True)
        self.assertEqual(record["status"], "cost_exhausted")
        self.assertIsNone(record["path"])
        self.assertTrue(
            verify_any_trace(6, 4, BLOCKED, (0, 0), GOALS,
                             round_trip(record), max_cost=4)["valid"]
        )

    def test_zero_budget(self):
        record = plan_any(3, 3, set(), (1, 1), [(2, 2), (1, 1)],
                          max_expanded=0, trace=True)
        self.assertEqual(record, {"path": None, "cost": None, "expanded": 0,
                                  "status": "budget_exhausted",
                                  "expanded_nodes": []})
        self.assertTrue(
            verify_any_trace(3, 3, set(), (1, 1), [(1, 1), (2, 2)],
                             round_trip(record),
                             max_expanded=0)["valid"]
        )

    def test_budget_wins_over_cost(self):
        # With both a budget and a cost limit the budget takes precedence;
        # the recorded budget_exhausted status verifies.
        record = plan_any(6, 4, BLOCKED, (0, 0), GOALS, max_expanded=2,
                          max_cost=4, trace=True)
        self.assertEqual(record["status"], "budget_exhausted")
        self.assertTrue(
            verify_any_trace(6, 4, BLOCKED, (0, 0), GOALS,
                             round_trip(record),
                             max_expanded=2, max_cost=4)["valid"]
        )

    def test_resume_record(self):
        # A resumed plan_any result equals the uninterrupted call with the
        # same cumulative budget, so it verifies against the same args.
        first = plan_any(6, 4, BLOCKED, (0, 0), GOALS, max_expanded=3,
                         snapshot=True, trace=True)
        record = resume(first["checkpoint"], max_expanded=100)
        self.assertEqual(record["status"], "found")
        self.assertTrue(
            verify_any_trace(6, 4, BLOCKED, (0, 0), GOALS, record,
                             max_expanded=100, snapshot=True)["valid"]
        )

    def test_resume_record_with_fresh_checkpoint(self):
        first = plan_any(6, 4, BLOCKED, (0, 0), GOALS, max_expanded=3,
                         snapshot=True, trace=True)
        record = resume(first["checkpoint"], max_expanded=5)
        self.assertEqual(record["status"], "budget_exhausted")
        self.assertIn("checkpoint", record)
        self.assertTrue(
            verify_any_trace(6, 4, BLOCKED, (0, 0), GOALS,
                             round_trip(record),
                             max_expanded=5, snapshot=True)["valid"]
        )


if __name__ == "__main__":
    unittest.main()
