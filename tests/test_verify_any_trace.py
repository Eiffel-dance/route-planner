import json
import unittest

from app import plan_any, replay, resume, verify_any_trace


BLOCKED = {(2, 0), (2, 1), (2, 2)}
GOALS = [(5, 3), (5, 2)]
FRAMES = [
    [(0, 1), (0, 2)],
    [(0, 1), (1, 2), (2, 0), (2, 2)],
    [(2, 1), (2, 2)],
    [(0, 0), (1, 2), (2, 1), (2, 2)],
    [],
    [(0, 1)],
]


def round_trip(result):
    return json.loads(json.dumps(result))


class VerifyAnyTraceStaticTest(unittest.TestCase):
    def test_valid_record_direct(self):
        record = plan_any(6, 4, BLOCKED, (0, 0), GOALS, trace=True)
        self.assertEqual(
            verify_any_trace(6, 4, BLOCKED, (0, 0), GOALS, record),
            {"valid": True, "mismatch": None, "index": None},
        )

    def test_valid_record_json_round_trip(self):
        record = round_trip(plan_any(6, 4, BLOCKED, (0, 0), GOALS,
                                     trace=True))
        self.assertTrue(
            verify_any_trace(6, 4, BLOCKED, (0, 0), GOALS, record)["valid"]
        )

    def test_mixed_sequences_and_extra_keys(self):
        record = plan_any(6, 4, BLOCKED, (0, 0), GOALS, trace=True)
        record["path"] = [list(point) for point in record["path"]]
        record["expanded_nodes"] = [
            list(entry) for entry in record["expanded_nodes"]
        ]
        record["something_else"] = {"ignored": True}
        self.assertTrue(
            verify_any_trace(6, 4, BLOCKED, (0, 0), GOALS, record)["valid"]
        )

    def test_goal_order_and_duplicates_do_not_matter(self):
        record = plan_any(6, 4, BLOCKED, (0, 0), [(5, 2), (5, 3), (5, 2)],
                          trace=True)
        self.assertTrue(
            verify_any_trace(6, 4, BLOCKED, (0, 0),
                             [(5, 3), (5, 2)], record)["valid"]
        )

    def test_unreachable_record(self):
        blocked = {(1, y) for y in range(3)}
        record = plan_any(3, 3, blocked, (0, 0), [(2, 2), (2, 0)],
                          trace=True)
        self.assertIsNone(record["path"])
        self.assertIsNone(record["cost"])
        self.assertTrue(
            verify_any_trace(3, 3, blocked, (0, 0), [(2, 2), (2, 0)],
                             record)["valid"]
        )

    def test_start_in_goals_record(self):
        record = plan_any(3, 3, set(), (1, 1), [(1, 1), (2, 2)], trace=True)
        self.assertEqual(record["path"], [(1, 1)])
        self.assertEqual(record["cost"], 0)
        self.assertTrue(
            verify_any_trace(3, 3, set(), (1, 1), [(2, 2), (1, 1)],
                             record)["valid"]
        )

    def test_costs_matrix_record(self):
        costs = [[1, 2, 1], [3, 1, 4], [1, 1, 1]]
        record = plan_any(3, 3, set(), (0, 0), [(2, 2), (2, 0)],
                          costs=costs, trace=True)
        self.assertTrue(
            verify_any_trace(3, 3, set(), (0, 0), [(2, 2), (2, 0)], record,
                             costs=costs)["valid"]
        )
        # The same record under different costs no longer matches.
        other = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        report = verify_any_trace(3, 3, set(), (0, 0), [(2, 2), (2, 0)],
                                  record, costs=other)
        self.assertFalse(report["valid"])

    def test_result_is_json_serializable(self):
        record = plan_any(6, 4, BLOCKED, (0, 0), GOALS, trace=True)
        report = verify_any_trace(6, 4, BLOCKED, (0, 0), GOALS, record)
        self.assertEqual(json.loads(json.dumps(report)), report)
        record["cost"] += 1
        report = verify_any_trace(6, 4, BLOCKED, (0, 0), GOALS, record)
        self.assertEqual(json.loads(json.dumps(report)), report)

    def test_record_is_not_modified(self):
        record = round_trip(plan_any(6, 4, BLOCKED, (0, 0), GOALS,
                                     trace=True))
        before = json.dumps(record, sort_keys=True)
        verify_any_trace(6, 4, BLOCKED, (0, 0), GOALS, record)
        self.assertEqual(json.dumps(record, sort_keys=True), before)

    def test_winning_path_replays(self):
        record = plan_any(6, 4, BLOCKED, (0, 0), GOALS, trace=True)
        replayed = replay(6, 4, BLOCKED, (0, 0), record["path"][-1],
                          record["path"])
        self.assertEqual(
            replayed,
            {"valid": True, "cost": record["cost"],
             "steps": len(record["path"]) - 1},
        )


class VerifyAnyTraceMismatchTest(unittest.TestCase):
    def setUp(self):
        self.args = (6, 4, BLOCKED, (0, 0), GOALS)
        self.record = round_trip(plan_any(*self.args, trace=True))

    def test_path_mismatch_index(self):
        self.record["path"][2] = list(self.record["path"][1])
        report = verify_any_trace(*self.args, self.record)
        self.assertEqual(
            report, {"valid": False, "mismatch": "path", "index": 2}
        )

    def test_path_length_mismatch_index(self):
        del self.record["path"][2:]
        report = verify_any_trace(*self.args, self.record)
        self.assertEqual(
            report, {"valid": False, "mismatch": "path", "index": 2}
        )

    def test_path_null_against_route(self):
        self.record["path"] = None
        report = verify_any_trace(*self.args, self.record)
        self.assertEqual(
            report, {"valid": False, "mismatch": "path", "index": 0}
        )

    def test_cost_mismatch(self):
        self.record["cost"] += 1
        report = verify_any_trace(*self.args, self.record)
        self.assertEqual(
            report, {"valid": False, "mismatch": "cost", "index": None}
        )

    def test_expanded_mismatch(self):
        self.record["expanded"] += 1
        report = verify_any_trace(*self.args, self.record)
        self.assertEqual(
            report, {"valid": False, "mismatch": "expanded", "index": None}
        )

    def test_expanded_nodes_mismatch_index(self):
        nodes = self.record["expanded_nodes"]
        nodes[1], nodes[2] = nodes[2], nodes[1]
        report = verify_any_trace(*self.args, self.record)
        self.assertEqual(
            report,
            {"valid": False, "mismatch": "expanded_nodes", "index": 1},
        )

    def test_expanded_nodes_prefix_mismatch(self):
        del self.record["expanded_nodes"][-1]
        report = verify_any_trace(*self.args, self.record)
        self.assertEqual(report["mismatch"], "expanded_nodes")
        self.assertEqual(report["index"], len(self.record["expanded_nodes"]))

    def test_mismatch_order_path_before_cost(self):
        self.record["path"][1] = list(self.record["path"][0])
        self.record["cost"] += 5
        report = verify_any_trace(*self.args, self.record)
        self.assertEqual(report["mismatch"], "path")

    def test_mismatch_order_expanded_before_nodes(self):
        self.record["expanded"] += 1
        self.record["expanded_nodes"][0] = [1, 0]
        report = verify_any_trace(*self.args, self.record)
        self.assertEqual(report["mismatch"], "expanded")

    def test_status_mismatch(self):
        record = round_trip(plan_any(6, 4, BLOCKED, (0, 0), GOALS,
                                     max_expanded=3, trace=True))
        self.assertEqual(record["status"], "budget_exhausted")
        record["status"] = "found"
        report = verify_any_trace(6, 4, BLOCKED, (0, 0), GOALS, record,
                                  max_expanded=3)
        self.assertEqual(
            report, {"valid": False, "mismatch": "status", "index": None}
        )

    def test_status_missing_from_record(self):
        record = round_trip(plan_any(6, 4, BLOCKED, (0, 0), GOALS,
                                     max_expanded=3, trace=True))
        del record["status"]
        report = verify_any_trace(6, 4, BLOCKED, (0, 0), GOALS, record,
                                  max_expanded=3)
        self.assertEqual(report["mismatch"], "status")

    def test_extra_status_ignored_without_budget(self):
        self.record["status"] = "found"
        report = verify_any_trace(*self.args, self.record)
        self.assertTrue(report["valid"])

    def test_checkpoint_mismatch(self):
        record = round_trip(plan_any(6, 4, BLOCKED, (0, 0), GOALS,
                                     max_expanded=3, snapshot=True,
                                     trace=True))
        self.assertIn("checkpoint", record)
        # A structurally valid but different snapshot: unit costs spelled
        # out as a matrix instead of the recorded null.
        record["checkpoint"]["costs"] = [[1] * 6 for _ in range(4)]
        report = verify_any_trace(6, 4, BLOCKED, (0, 0), GOALS, record,
                                  max_expanded=3, snapshot=True)
        self.assertEqual(
            report,
            {"valid": False, "mismatch": "checkpoint", "index": None},
        )

    def test_checkpoint_missing_from_record(self):
        record = round_trip(plan_any(6, 4, BLOCKED, (0, 0), GOALS,
                                     max_expanded=3, snapshot=True,
                                     trace=True))
        del record["checkpoint"]
        report = verify_any_trace(6, 4, BLOCKED, (0, 0), GOALS, record,
                                  max_expanded=3, snapshot=True)
        self.assertEqual(report["mismatch"], "checkpoint")

    def test_checkpoint_valid(self):
        record = round_trip(plan_any(6, 4, BLOCKED, (0, 0), GOALS,
                                     max_expanded=3, snapshot=True,
                                     trace=True))
        report = verify_any_trace(6, 4, BLOCKED, (0, 0), GOALS, record,
                                  max_expanded=3, snapshot=True)
        self.assertTrue(report["valid"])

    def test_extra_checkpoint_ignored_without_snapshot(self):
        record = round_trip(plan_any(6, 4, BLOCKED, (0, 0), GOALS,
                                     max_expanded=3, snapshot=True,
                                     trace=True))
        report = verify_any_trace(6, 4, BLOCKED, (0, 0), GOALS, record,
                                  max_expanded=3)
        self.assertTrue(report["valid"])


class VerifyAnyTraceValidationTest(unittest.TestCase):
    def setUp(self):
        self.record = round_trip(plan_any(6, 4, BLOCKED, (0, 0), GOALS,
                                          trace=True))

    def test_grid_errors_precede_record_errors(self):
        # A bad width raises before the missing-field record error.
        with self.assertRaises(TypeError):
            verify_any_trace("6", 4, BLOCKED, (0, 0), GOALS, {})
        # A bad start raises before the record is inspected.
        with self.assertRaises(ValueError):
            verify_any_trace(6, 4, BLOCKED, (9, 9), GOALS, None)
        # A blocked goal candidate raises before the record is inspected.
        with self.assertRaises(ValueError):
            verify_any_trace(6, 4, BLOCKED, (0, 0), [(5, 3), (2, 0)],
                             "record")
        # A bad budget raises before the record is inspected.
        with self.assertRaises(ValueError):
            verify_any_trace(6, 4, BLOCKED, (0, 0), GOALS, [],
                             max_expanded=-1)
        # A bad reservation raises before the record is inspected.
        with self.assertRaises(ValueError):
            verify_any_trace(6, 4, BLOCKED, (0, 0), GOALS, [],
                             reservations=[[(2, 0)]])

    def test_goals_errors_precede_record_errors(self):
        # ``goals`` occupies ``goal``'s position in the validation
        # sequence: every goals error is decided before the record is
        # inspected.
        with self.assertRaises(TypeError):
            verify_any_trace(6, 4, BLOCKED, (0, 0), "goals", {})
        with self.assertRaises(TypeError):
            verify_any_trace(6, 4, BLOCKED, (0, 0), None, {})
        with self.assertRaises(ValueError):
            verify_any_trace(6, 4, BLOCKED, (0, 0), [], {})
        with self.assertRaises(TypeError):
            verify_any_trace(6, 4, BLOCKED, (0, 0), [(5, 3), "xy"], {})
        with self.assertRaises(ValueError):
            verify_any_trace(6, 4, BLOCKED, (0, 0), [(5, 3), (9, 9)], {})

    def test_record_not_an_object(self):
        for bad in (None, [], "record", 42, True):
            with self.assertRaises(TypeError):
                verify_any_trace(6, 4, BLOCKED, (0, 0), GOALS, bad)

    def test_record_missing_fields(self):
        for key in ("path", "cost", "expanded", "expanded_nodes"):
            record = dict(self.record)
            del record[key]
            with self.assertRaises(TypeError):
                verify_any_trace(6, 4, BLOCKED, (0, 0), GOALS, record)

    def test_record_field_type_errors(self):
        for key, bad in (("path", "path"), ("cost", "cost"),
                         ("expanded", True), ("expanded_nodes", 7)):
            record = dict(self.record)
            record[key] = bad
            with self.assertRaises(TypeError):
                verify_any_trace(6, 4, BLOCKED, (0, 0), GOALS, record)

    def test_record_path_shape_errors(self):
        for bad_entry in ("xy", [1], [1, 2, 3], [1, "y"], [True, 0]):
            record = dict(self.record)
            record["path"] = [bad_entry]
            with self.assertRaises(TypeError):
                verify_any_trace(6, 4, BLOCKED, (0, 0), GOALS, record)

    def test_record_trajectory_shape_errors(self):
        for bad_entry in ("xyt", [1], [1, 2, 3, 4], [1, 2, "t"]):
            record = dict(self.record)
            record["expanded_nodes"] = [bad_entry]
            with self.assertRaises(TypeError):
                verify_any_trace(6, 4, BLOCKED, (0, 0), GOALS, record)

    def test_record_coordinate_out_of_bounds(self):
        record = dict(self.record)
        record["path"] = [[9, 0]]
        with self.assertRaises(ValueError):
            verify_any_trace(6, 4, BLOCKED, (0, 0), GOALS, record)
        record = dict(self.record)
        record["expanded_nodes"] = [[0, 9]]
        with self.assertRaises(ValueError):
            verify_any_trace(6, 4, BLOCKED, (0, 0), GOALS, record)

    def test_record_trajectory_mode_mismatch(self):
        # A triple in static mode is a ValueError, not a mismatch.
        record = dict(self.record)
        record["expanded_nodes"] = [[0, 0, 0]]
        with self.assertRaises(ValueError):
            verify_any_trace(6, 4, BLOCKED, (0, 0), GOALS, record)
        # A pair in dynamic mode is a ValueError too.
        dynamic = plan_any(6, 4, BLOCKED, (0, 0), GOALS,
                           dynamic_blocked=FRAMES, trace=True)
        record = round_trip(dynamic)
        record["expanded_nodes"] = [[0, 0]]
        with self.assertRaises(ValueError):
            verify_any_trace(6, 4, BLOCKED, (0, 0), GOALS, record,
                             dynamic_blocked=FRAMES)

    def test_record_negative_time(self):
        record = round_trip(plan_any(6, 4, BLOCKED, (0, 0), GOALS,
                                     dynamic_blocked=FRAMES, trace=True))
        record["expanded_nodes"][0] = [0, 0, -1]
        with self.assertRaises(ValueError):
            verify_any_trace(6, 4, BLOCKED, (0, 0), GOALS, record,
                             dynamic_blocked=FRAMES)

    def test_record_static_duplicate_state_is_not_an_error(self):
        # The multi-goal route-tree search legitimately records the same
        # cell through different histories even in static mode, so a
        # repeated state is legal structure and only ever a mismatch.
        record = dict(self.record)
        record["expanded_nodes"] = (record["expanded_nodes"]
                                    + [record["expanded_nodes"][0]])
        report = verify_any_trace(6, 4, BLOCKED, (0, 0), GOALS, record)
        self.assertEqual(report["mismatch"], "expanded_nodes")
        self.assertEqual(report["index"], len(self.record["expanded_nodes"]))
        # A genuine record with repeated states verifies as valid.
        genuine = plan_any(6, 4, BLOCKED, (0, 0), GOALS, trace=True)
        nodes = genuine["expanded_nodes"]
        self.assertLess(len(set(nodes)), len(nodes))  # repeats present
        self.assertTrue(
            verify_any_trace(6, 4, BLOCKED, (0, 0), GOALS,
                             round_trip(genuine))["valid"]
        )

    def test_record_bad_checkpoint(self):
        record = dict(self.record)
        record["checkpoint"] = {"version": 1}
        with self.assertRaises(ValueError):
            verify_any_trace(6, 4, BLOCKED, (0, 0), GOALS, record)
        record = dict(self.record)
        record["checkpoint"] = 42
        with self.assertRaises(ValueError):
            verify_any_trace(6, 4, BLOCKED, (0, 0), GOALS, record)

    def test_record_bad_status_type(self):
        record = dict(self.record)
        record["status"] = 7
        with self.assertRaises(TypeError):
            verify_any_trace(6, 4, BLOCKED, (0, 0), GOALS, record)


class VerifyAnyTraceModesTest(unittest.TestCase):
    def test_dynamic_blocked(self):
        record = plan_any(6, 4, BLOCKED, (0, 0), GOALS,
                          dynamic_blocked=FRAMES, trace=True)
        self.assertTrue(
            verify_any_trace(6, 4, BLOCKED, (0, 0), GOALS, record,
                             dynamic_blocked=FRAMES)["valid"]
        )
        self.assertTrue(
            verify_any_trace(6, 4, BLOCKED, (0, 0), GOALS,
                             round_trip(record), dynamic_blocked=FRAMES
                             )["valid"]
        )

    def test_dynamic_blocked_mismatch(self):
        record = round_trip(plan_any(6, 4, BLOCKED, (0, 0), GOALS,
                                     dynamic_blocked=FRAMES, trace=True))
        record["expanded_nodes"][0] = [0, 0, 1]
        report = verify_any_trace(6, 4, BLOCKED, (0, 0), GOALS, record,
                                  dynamic_blocked=FRAMES)
        self.assertEqual(report["mismatch"], "expanded_nodes")
        self.assertEqual(report["index"], 0)

    def test_allow_wait(self):
        record = plan_any(6, 4, BLOCKED, (0, 0), GOALS,
                          dynamic_blocked=FRAMES, allow_wait=True,
                          trace=True)
        self.assertTrue(
            verify_any_trace(6, 4, BLOCKED, (0, 0), GOALS, record,
                             dynamic_blocked=FRAMES,
                             allow_wait=True)["valid"]
        )

    def test_dynamic_costs(self):
        frames = [
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [[2, 2, 2], [2, 1, 2], [2, 2, 2]],
        ]
        record = plan_any(3, 3, set(), (0, 0), [(2, 2), (0, 2)],
                          dynamic_costs=frames, trace=True)
        self.assertTrue(
            verify_any_trace(3, 3, set(), (0, 0), [(2, 2), (0, 2)], record,
                             dynamic_costs=frames)["valid"]
        )

    def test_reservations(self):
        reservations = [[(1, 0), (1, 1), (1, 2)]]
        record = plan_any(3, 3, set(), (0, 0), [(2, 2), (2, 0)],
                          reservations=reservations, trace=True)
        self.assertTrue(
            verify_any_trace(3, 3, set(), (0, 0), [(2, 2), (2, 0)], record,
                             reservations=reservations)["valid"]
        )

    def test_max_cost(self):
        record = plan_any(6, 4, BLOCKED, (0, 0), GOALS, max_cost=4,
                          trace=True)
        self.assertEqual(record["status"], "cost_exhausted")
        self.assertTrue(
            verify_any_trace(6, 4, BLOCKED, (0, 0), GOALS, record,
                             max_cost=4)["valid"]
        )

    def test_max_expanded_zero_budget(self):
        record = plan_any(3, 3, set(), (1, 1), [(1, 1), (2, 2)],
                          max_expanded=0, trace=True)
        self.assertEqual(record["status"], "budget_exhausted")
        self.assertEqual(record["expanded"], 0)
        self.assertEqual(record["expanded_nodes"], [])
        self.assertTrue(
            verify_any_trace(3, 3, set(), (1, 1), [(1, 1), (2, 2)], record,
                             max_expanded=0)["valid"]
        )

    def test_resume_record(self):
        # A resumed result equals the uninterrupted call with the same
        # cumulative budget, so it verifies against the same arguments.
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
                             round_trip(record), max_expanded=5,
                             snapshot=True)["valid"]
        )


if __name__ == "__main__":
    unittest.main()
