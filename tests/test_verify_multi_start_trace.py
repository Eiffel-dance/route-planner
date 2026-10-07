import json
import unittest

from app import plan_multi_start, resume, verify_multi_start_trace


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
STARTS = [(0, 0), (5, 0), (0, 3)]


def round_trip(result):
    return json.loads(json.dumps(result))


def audit(starts, record, **kwargs):
    return verify_multi_start_trace(6, 4, BLOCKED, starts, (5, 3), record,
                                    **kwargs)


class VerifyMultiStartStaticTest(unittest.TestCase):
    def test_valid_record_direct(self):
        record = plan_multi_start(6, 4, BLOCKED, STARTS, (5, 3), trace=True)
        self.assertEqual(
            audit(STARTS, record),
            {"valid": True, "mismatch": None, "index": None},
        )

    def test_valid_record_json_round_trip(self):
        record = round_trip(plan_multi_start(6, 4, BLOCKED, STARTS, (5, 3),
                                             trace=True))
        self.assertTrue(audit(STARTS, record)["valid"])

    def test_mixed_sequences_and_extra_keys(self):
        record = plan_multi_start(6, 4, BLOCKED, STARTS, (5, 3), trace=True)
        record["path"] = [list(point) for point in record["path"]]
        record["expanded_nodes"] = [
            list(entry) for entry in record["expanded_nodes"]
        ]
        record["something_else"] = {"ignored": True}
        self.assertTrue(audit(STARTS, record)["valid"])

    def test_start_order_and_duplicates_do_not_matter(self):
        record = round_trip(plan_multi_start(6, 4, BLOCKED, STARTS, (5, 3),
                                             trace=True))
        shuffled = [STARTS[2], STARTS[0], STARTS[0], STARTS[1]]
        self.assertTrue(audit(shuffled, record)["valid"])

    def test_unreachable_record(self):
        record = plan_multi_start(3, 3, WALL, [(0, 0), (0, 2)], (2, 2),
                                  trace=True)
        self.assertIsNone(record["path"])
        self.assertIsNone(record["cost"])
        self.assertTrue(
            verify_multi_start_trace(3, 3, WALL, [(0, 2), (0, 0)], (2, 2),
                                     record)["valid"]
        )

    def test_start_equals_goal_record(self):
        record = plan_multi_start(3, 3, set(), [(0, 0), (1, 1)], (1, 1),
                                  trace=True)
        self.assertEqual(record, {"path": [(1, 1)], "cost": 0, "expanded": 1,
                                  "expanded_nodes": [(1, 1)]})
        self.assertTrue(
            verify_multi_start_trace(3, 3, set(), [(1, 1), (0, 0)], (1, 1),
                                     record)["valid"]
        )

    def test_costs_matrix_record(self):
        costs = [[1, 2, 1], [3, 1, 4], [1, 1, 1]]
        record = plan_multi_start(3, 3, set(), [(0, 0), (0, 2)], (2, 2),
                                  costs=costs, trace=True)
        self.assertTrue(
            verify_multi_start_trace(3, 3, set(), [(0, 2), (0, 0)], (2, 2),
                                     record, costs=costs)["valid"]
        )
        # The same record under different costs no longer matches.
        other = [[2, 2, 2], [2, 2, 2], [2, 2, 2]]
        report = verify_multi_start_trace(3, 3, set(), [(0, 0), (0, 2)],
                                          (2, 2), record, costs=other)
        self.assertFalse(report["valid"])

    def test_result_is_json_serializable(self):
        record = plan_multi_start(6, 4, BLOCKED, STARTS, (5, 3), trace=True)
        report = audit(STARTS, record)
        self.assertEqual(json.loads(json.dumps(report)), report)
        record["cost"] += 1
        report = audit(STARTS, record)
        self.assertEqual(json.loads(json.dumps(report)), report)

    def test_record_is_not_modified(self):
        record = round_trip(plan_multi_start(6, 4, BLOCKED, STARTS, (5, 3),
                                             trace=True))
        before = json.dumps(record, sort_keys=True)
        audit(STARTS, record)
        self.assertEqual(json.dumps(record, sort_keys=True), before)


class VerifyMultiStartMismatchTest(unittest.TestCase):
    def setUp(self):
        self.record = round_trip(plan_multi_start(6, 4, BLOCKED, STARTS,
                                                  (5, 3), trace=True))

    def test_path_mismatch_index(self):
        self.record["path"][2] = list(self.record["path"][1])
        report = audit(STARTS, self.record)
        self.assertEqual(
            report, {"valid": False, "mismatch": "path", "index": 2}
        )

    def test_path_length_mismatch_index(self):
        del self.record["path"][2:]
        report = audit(STARTS, self.record)
        self.assertEqual(
            report, {"valid": False, "mismatch": "path", "index": 2}
        )

    def test_path_null_against_route(self):
        self.record["path"] = None
        report = audit(STARTS, self.record)
        self.assertEqual(
            report, {"valid": False, "mismatch": "path", "index": 0}
        )

    def test_cost_mismatch(self):
        self.record["cost"] += 1
        report = audit(STARTS, self.record)
        self.assertEqual(
            report, {"valid": False, "mismatch": "cost", "index": None}
        )

    def test_expanded_mismatch(self):
        self.record["expanded"] += 1
        report = audit(STARTS, self.record)
        self.assertEqual(
            report, {"valid": False, "mismatch": "expanded", "index": None}
        )

    def test_expanded_nodes_mismatch_index(self):
        nodes = self.record["expanded_nodes"]
        nodes[1], nodes[2] = nodes[2], nodes[1]
        report = audit(STARTS, self.record)
        self.assertEqual(
            report,
            {"valid": False, "mismatch": "expanded_nodes", "index": 1},
        )

    def test_expanded_nodes_prefix_mismatch(self):
        del self.record["expanded_nodes"][-1]
        report = audit(STARTS, self.record)
        self.assertEqual(report["mismatch"], "expanded_nodes")
        self.assertEqual(report["index"], len(self.record["expanded_nodes"]))

    def test_mismatch_order_path_before_cost(self):
        self.record["path"][1] = list(self.record["path"][0])
        self.record["cost"] += 5
        report = audit(STARTS, self.record)
        self.assertEqual(report["mismatch"], "path")

    def test_mismatch_order_expanded_before_nodes(self):
        self.record["expanded"] += 1
        self.record["expanded_nodes"][0] = [1, 0]
        report = audit(STARTS, self.record)
        self.assertEqual(report["mismatch"], "expanded")

    def test_status_mismatch(self):
        record = round_trip(plan_multi_start(6, 4, BLOCKED, STARTS, (5, 3),
                                             max_expanded=3, trace=True))
        self.assertEqual(record["status"], "budget_exhausted")
        record["status"] = "found"
        report = audit(STARTS, record, max_expanded=3)
        self.assertEqual(
            report, {"valid": False, "mismatch": "status", "index": None}
        )

    def test_status_missing_from_record(self):
        record = round_trip(plan_multi_start(6, 4, BLOCKED, STARTS, (5, 3),
                                             max_expanded=3, trace=True))
        del record["status"]
        report = audit(STARTS, record, max_expanded=3)
        self.assertEqual(report["mismatch"], "status")

    def test_extra_status_ignored_without_budget(self):
        self.record["status"] = "found"
        report = audit(STARTS, self.record)
        self.assertTrue(report["valid"])

    def test_checkpoint_mismatch(self):
        record = round_trip(plan_multi_start(6, 4, BLOCKED, STARTS, (5, 3),
                                             max_expanded=3, snapshot=True,
                                             trace=True))
        self.assertIn("checkpoint", record)
        # A structurally valid but different snapshot: unit costs spelled
        # out as a matrix instead of the recorded null.
        record["checkpoint"]["costs"] = [[1] * 6 for _ in range(4)]
        report = audit(STARTS, record, max_expanded=3, snapshot=True)
        self.assertEqual(
            report,
            {"valid": False, "mismatch": "checkpoint", "index": None},
        )

    def test_checkpoint_missing_from_record(self):
        record = round_trip(plan_multi_start(6, 4, BLOCKED, STARTS, (5, 3),
                                             max_expanded=3, snapshot=True,
                                             trace=True))
        del record["checkpoint"]
        report = audit(STARTS, record, max_expanded=3, snapshot=True)
        self.assertEqual(report["mismatch"], "checkpoint")

    def test_checkpoint_valid(self):
        record = round_trip(plan_multi_start(6, 4, BLOCKED, STARTS, (5, 3),
                                             max_expanded=3, snapshot=True,
                                             trace=True))
        report = audit(STARTS, record, max_expanded=3, snapshot=True)
        self.assertTrue(report["valid"])

    def test_extra_checkpoint_ignored_without_snapshot(self):
        record = round_trip(plan_multi_start(6, 4, BLOCKED, STARTS, (5, 3),
                                             max_expanded=3, snapshot=True,
                                             trace=True))
        report = audit(STARTS, record, max_expanded=3)
        self.assertTrue(report["valid"])


class VerifyMultiStartValidationTest(unittest.TestCase):
    def setUp(self):
        self.record = round_trip(plan_multi_start(6, 4, BLOCKED, STARTS,
                                                  (5, 3), trace=True))

    def test_grid_errors_precede_record_errors(self):
        # A bad width raises before the missing-field record error.
        with self.assertRaises(TypeError):
            verify_multi_start_trace("6", 4, BLOCKED, STARTS, (5, 3), {})
        # Bad starts raise before the record is inspected.
        with self.assertRaises(TypeError):
            verify_multi_start_trace(6, 4, BLOCKED, None, (5, 3), {})
        with self.assertRaises(ValueError):
            verify_multi_start_trace(6, 4, BLOCKED, [(9, 9)], (5, 3), None)
        # A blocked start raises before the record is inspected.
        with self.assertRaises(ValueError):
            verify_multi_start_trace(6, 4, BLOCKED, [(2, 0)], (5, 3),
                                     "record")
        # A blocked goal raises before the record is inspected.
        with self.assertRaises(ValueError):
            verify_multi_start_trace(6, 4, BLOCKED, STARTS, (2, 0), "record")
        # A bad budget raises before the record is inspected.
        with self.assertRaises(ValueError):
            verify_multi_start_trace(6, 4, BLOCKED, STARTS, (5, 3), [],
                                     max_expanded=-1)
        # A bad reservation raises before the record is inspected.
        with self.assertRaises(ValueError):
            verify_multi_start_trace(6, 4, BLOCKED, STARTS, (5, 3), [],
                                     reservations=[[(2, 0)]])
        # A bad step window raises before the record is inspected.
        with self.assertRaises(ValueError):
            verify_multi_start_trace(6, 4, BLOCKED, STARTS, (5, 3), [],
                                     max_steps=-1)

    def test_record_not_an_object(self):
        for bad in (None, [], "record", 42, True):
            with self.assertRaises(TypeError):
                audit(STARTS, bad)

    def test_record_missing_fields(self):
        for key in ("path", "cost", "expanded", "expanded_nodes"):
            record = dict(self.record)
            del record[key]
            with self.assertRaises(TypeError):
                audit(STARTS, record)

    def test_record_field_type_errors(self):
        for key, bad in (("path", "path"), ("cost", "cost"),
                         ("expanded", True), ("expanded_nodes", 7)):
            record = dict(self.record)
            record[key] = bad
            with self.assertRaises(TypeError):
                audit(STARTS, record)

    def test_record_path_shape_errors(self):
        for bad_entry in ("xy", [1], [1, 2, 3], [1, "y"], [True, 0]):
            record = dict(self.record)
            record["path"] = [bad_entry]
            with self.assertRaises(TypeError):
                audit(STARTS, record)

    def test_record_trajectory_shape_errors(self):
        for bad_entry in ("xyt", [1], [1, 2, 3, 4], [1, 2, "t"]):
            record = dict(self.record)
            record["expanded_nodes"] = [bad_entry]
            with self.assertRaises(TypeError):
                audit(STARTS, record)

    def test_record_coordinate_out_of_bounds(self):
        record = dict(self.record)
        record["path"] = [[9, 0]]
        with self.assertRaises(ValueError):
            audit(STARTS, record)
        record = dict(self.record)
        record["expanded_nodes"] = [[0, 9]]
        with self.assertRaises(ValueError):
            audit(STARTS, record)

    def test_record_trajectory_mode_mismatch(self):
        # A triple in static mode is a ValueError, not a mismatch.
        record = dict(self.record)
        record["expanded_nodes"] = [[0, 0, 0]]
        with self.assertRaises(ValueError):
            audit(STARTS, record)
        # A pair in dynamic mode is a ValueError too.
        dynamic = plan_multi_start(6, 4, BLOCKED, STARTS, (5, 3),
                                   dynamic_blocked=FRAMES, trace=True)
        record = round_trip(dynamic)
        record["expanded_nodes"] = [[0, 0]]
        with self.assertRaises(ValueError):
            audit(STARTS, record, dynamic_blocked=FRAMES)
        # A negative time in dynamic mode is a ValueError.
        record = round_trip(dynamic)
        record["expanded_nodes"][0] = [0, 0, -1]
        with self.assertRaises(ValueError):
            audit(STARTS, record, dynamic_blocked=FRAMES)

    def test_record_static_duplicate_state(self):
        record = dict(self.record)
        record["expanded_nodes"] = (record["expanded_nodes"]
                                    + [record["expanded_nodes"][0]])
        with self.assertRaises(ValueError):
            audit(STARTS, record)

    def test_record_bad_checkpoint(self):
        record = dict(self.record)
        record["checkpoint"] = {"version": 1}
        with self.assertRaises(ValueError):
            audit(STARTS, record)
        record = dict(self.record)
        record["checkpoint"] = 42
        with self.assertRaises(ValueError):
            audit(STARTS, record)

    def test_record_bad_status_type(self):
        record = dict(self.record)
        record["status"] = 7
        with self.assertRaises(TypeError):
            audit(STARTS, record)


class VerifyMultiStartModesTest(unittest.TestCase):
    def test_dynamic_blocked(self):
        record = plan_multi_start(6, 4, BLOCKED, STARTS, (5, 3),
                                  dynamic_blocked=FRAMES, trace=True)
        self.assertTrue(
            audit(STARTS, record, dynamic_blocked=FRAMES)["valid"]
        )
        self.assertTrue(
            audit(STARTS, round_trip(record), dynamic_blocked=FRAMES)["valid"]
        )

    def test_dynamic_blocked_mismatch(self):
        record = round_trip(plan_multi_start(6, 4, BLOCKED, STARTS, (5, 3),
                                             dynamic_blocked=FRAMES,
                                             trace=True))
        record["expanded_nodes"][0] = [0, 0, 1]
        report = audit(STARTS, record, dynamic_blocked=FRAMES)
        self.assertEqual(report["mismatch"], "expanded_nodes")
        self.assertEqual(report["index"], 0)

    def test_allow_wait(self):
        record = plan_multi_start(6, 4, BLOCKED, STARTS, (5, 3),
                                  dynamic_blocked=FRAMES, allow_wait=True,
                                  trace=True)
        self.assertTrue(
            audit(STARTS, record, dynamic_blocked=FRAMES,
                  allow_wait=True)["valid"]
        )

    def test_dynamic_costs(self):
        frames = [
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [[2, 2, 2], [2, 1, 2], [2, 2, 2]],
        ]
        record = plan_multi_start(3, 3, set(), [(0, 0), (0, 2)], (2, 2),
                                  dynamic_costs=frames, trace=True)
        self.assertTrue(
            verify_multi_start_trace(3, 3, set(), [(0, 0), (0, 2)], (2, 2),
                                     record, dynamic_costs=frames)["valid"]
        )

    def test_reservations(self):
        reservations = [[(1, 0), (1, 1), (1, 2)]]
        record = plan_multi_start(3, 3, set(), [(0, 0), (2, 0)], (2, 2),
                                  reservations=reservations, trace=True)
        self.assertTrue(
            verify_multi_start_trace(3, 3, set(), [(0, 0), (2, 0)], (2, 2),
                                     record,
                                     reservations=reservations)["valid"]
        )

    def test_max_cost(self):
        # Every start is too far to reach the goal within the cost limit.
        starts = [(0, 0), (0, 3)]
        record = plan_multi_start(6, 4, BLOCKED, starts, (5, 3), max_cost=4,
                                  trace=True)
        self.assertEqual(record["status"], "cost_exhausted")
        self.assertTrue(audit(starts, record, max_cost=4)["valid"])

    def test_max_steps(self):
        record = plan_multi_start(6, 4, BLOCKED, STARTS, (5, 3), max_steps=2,
                                  trace=True)
        self.assertEqual(record["status"], "step_exhausted")
        self.assertTrue(audit(STARTS, record, max_steps=2)["valid"])

    def test_max_expanded_zero_budget(self):
        record = plan_multi_start(3, 3, set(), [(0, 0), (1, 1)], (1, 1),
                                  max_expanded=0, trace=True)
        self.assertEqual(record["status"], "budget_exhausted")
        self.assertEqual(record["expanded"], 0)
        self.assertEqual(record["expanded_nodes"], [])
        self.assertTrue(
            verify_multi_start_trace(3, 3, set(), [(1, 1), (0, 0)], (1, 1),
                                     record, max_expanded=0)["valid"]
        )

    def test_resume_record(self):
        # A resumed result equals the uninterrupted call with the same
        # cumulative budget, so it verifies against the same arguments.
        first = plan_multi_start(6, 4, BLOCKED, STARTS, (5, 3),
                                 max_expanded=3, snapshot=True, trace=True)
        record = resume(first["checkpoint"], max_expanded=100)
        self.assertEqual(record["status"], "found")
        self.assertTrue(
            audit(STARTS, record, max_expanded=100, snapshot=True)["valid"]
        )

    def test_resume_record_with_fresh_checkpoint(self):
        first = plan_multi_start(6, 4, BLOCKED, STARTS, (5, 3),
                                 max_expanded=3, snapshot=True, trace=True)
        record = resume(first["checkpoint"], max_expanded=3)
        self.assertEqual(record["status"], "budget_exhausted")
        self.assertIn("checkpoint", record)
        self.assertTrue(
            audit(STARTS, round_trip(record), max_expanded=3,
                  snapshot=True)["valid"]
        )


if __name__ == "__main__":
    unittest.main()
