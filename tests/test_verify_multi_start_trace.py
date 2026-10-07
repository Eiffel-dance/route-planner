import json
import unittest

from app import plan_multi_start, verify_multi_start_trace


BLOCKED = {(2, 0), (2, 1), (2, 2)}
FRAMES = [
    [(0, 1), (0, 2)],
    [(0, 1), (1, 2), (2, 0), (2, 2)],
    [(2, 1), (2, 2)],
    [(0, 0), (1, 2), (2, 1), (2, 2)],
    [],
    [(0, 1)],
]
STARTS = [(0, 0), (5, 3)]


def round_trip(result):
    return json.loads(json.dumps(result))


class VerifyMultiStartStaticTest(unittest.TestCase):
    args = (6, 4, BLOCKED, STARTS, (5, 0))

    def test_valid_record_direct(self):
        record = plan_multi_start(*self.args, trace=True)
        self.assertEqual(
            verify_multi_start_trace(*self.args, record),
            {"valid": True, "mismatch": None, "index": None},
        )

    def test_valid_record_json_round_trip(self):
        record = round_trip(plan_multi_start(*self.args, trace=True))
        self.assertTrue(
            verify_multi_start_trace(*self.args, record)["valid"]
        )

    def test_mixed_sequences_and_extra_keys(self):
        record = plan_multi_start(*self.args, trace=True)
        record["path"] = [list(point) for point in record["path"]]
        record["expanded_nodes"] = [
            list(entry) for entry in record["expanded_nodes"]
        ]
        record["something_else"] = {"ignored": True}
        self.assertTrue(
            verify_multi_start_trace(*self.args, record)["valid"]
        )

    def test_start_permutation_matches_same_record(self):
        # Multi-source merging makes the input order of starts irrelevant,
        # so a record audits against any permutation of the same starts.
        record = round_trip(plan_multi_start(*self.args, trace=True))
        self.assertTrue(
            verify_multi_start_trace(
                6, 4, BLOCKED, [(5, 3), (0, 0)], (5, 0), record
            )["valid"]
        )

    def test_costs_matrix_record(self):
        costs = [[1, 2, 1], [3, 1, 4], [1, 1, 1]]
        record = plan_multi_start(3, 3, set(), [(0, 0), (2, 2)], (2, 0),
                                  costs=costs, trace=True)
        self.assertTrue(
            verify_multi_start_trace(
                3, 3, set(), [(0, 0), (2, 2)], (2, 0), record,
                costs=costs)["valid"]
        )
        other = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        report = verify_multi_start_trace(
            3, 3, set(), [(0, 0), (2, 2)], (2, 0), record, costs=other)
        self.assertFalse(report["valid"])

    def test_unreachable_record(self):
        wall = {(1, y) for y in range(3)}
        record = plan_multi_start(3, 3, wall, [(0, 0), (0, 2)], (2, 2),
                                  trace=True)
        self.assertIsNone(record["path"])
        self.assertTrue(
            verify_multi_start_trace(
                3, 3, wall, [(0, 0), (0, 2)], (2, 2), record)["valid"]
        )

    def test_start_equals_goal_record(self):
        record = plan_multi_start(3, 3, set(), [(1, 1), (0, 0)], (1, 1),
                                  trace=True)
        self.assertTrue(
            verify_multi_start_trace(
                3, 3, set(), [(1, 1), (0, 0)], (1, 1), record)["valid"]
        )

    def test_result_is_json_serializable(self):
        record = plan_multi_start(*self.args, trace=True)
        report = verify_multi_start_trace(*self.args, record)
        self.assertEqual(json.loads(json.dumps(report)), report)
        record["cost"] += 1
        report = verify_multi_start_trace(*self.args, record)
        self.assertEqual(json.loads(json.dumps(report)), report)

    def test_record_is_not_modified(self):
        record = round_trip(plan_multi_start(*self.args, trace=True))
        before = json.dumps(record, sort_keys=True)
        verify_multi_start_trace(*self.args, record)
        self.assertEqual(json.dumps(record, sort_keys=True), before)


class VerifyMultiStartMismatchTest(unittest.TestCase):
    def setUp(self):
        self.args = (6, 4, BLOCKED, STARTS, (5, 0))
        self.record = round_trip(plan_multi_start(*self.args, trace=True))

    def test_path_mismatch_index(self):
        self.record["path"][2] = list(self.record["path"][1])
        report = verify_multi_start_trace(*self.args, self.record)
        self.assertEqual(
            report, {"valid": False, "mismatch": "path", "index": 2}
        )

    def test_path_length_mismatch_index(self):
        del self.record["path"][2:]
        report = verify_multi_start_trace(*self.args, self.record)
        self.assertEqual(
            report, {"valid": False, "mismatch": "path", "index": 2}
        )

    def test_path_null_against_route(self):
        self.record["path"] = None
        report = verify_multi_start_trace(*self.args, self.record)
        self.assertEqual(
            report, {"valid": False, "mismatch": "path", "index": 0}
        )

    def test_cost_mismatch(self):
        self.record["cost"] += 1
        report = verify_multi_start_trace(*self.args, self.record)
        self.assertEqual(
            report, {"valid": False, "mismatch": "cost", "index": None}
        )

    def test_expanded_mismatch(self):
        self.record["expanded"] += 1
        report = verify_multi_start_trace(*self.args, self.record)
        self.assertEqual(
            report, {"valid": False, "mismatch": "expanded", "index": None}
        )

    def test_expanded_nodes_mismatch_index(self):
        nodes = self.record["expanded_nodes"]
        nodes[1], nodes[2] = nodes[2], nodes[1]
        report = verify_multi_start_trace(*self.args, self.record)
        self.assertEqual(
            report,
            {"valid": False, "mismatch": "expanded_nodes", "index": 1},
        )

    def test_expanded_nodes_prefix_mismatch(self):
        del self.record["expanded_nodes"][-1]
        report = verify_multi_start_trace(*self.args, self.record)
        self.assertEqual(report["mismatch"], "expanded_nodes")
        self.assertEqual(report["index"],
                         len(self.record["expanded_nodes"]))

    def test_mismatch_order_path_before_cost(self):
        self.record["path"][1] = list(self.record["path"][0])
        self.record["cost"] += 5
        report = verify_multi_start_trace(*self.args, self.record)
        self.assertEqual(report["mismatch"], "path")

    def test_mismatch_order_expanded_before_nodes(self):
        self.record["expanded"] += 1
        self.record["expanded_nodes"][0] = [1, 0]
        report = verify_multi_start_trace(*self.args, self.record)
        self.assertEqual(report["mismatch"], "expanded")

    def test_status_mismatch(self):
        record = round_trip(plan_multi_start(*self.args, max_expanded=3,
                                            trace=True))
        self.assertEqual(record["status"], "budget_exhausted")
        record["status"] = "found"
        report = verify_multi_start_trace(*self.args, record,
                                          max_expanded=3)
        self.assertEqual(
            report, {"valid": False, "mismatch": "status", "index": None}
        )

    def test_status_missing_from_record(self):
        record = round_trip(plan_multi_start(*self.args, max_expanded=3,
                                            trace=True))
        del record["status"]
        report = verify_multi_start_trace(*self.args, record,
                                          max_expanded=3)
        self.assertEqual(report["mismatch"], "status")

    def test_extra_status_ignored_without_budget(self):
        self.record["status"] = "found"
        report = verify_multi_start_trace(*self.args, self.record)
        self.assertTrue(report["valid"])

    def test_checkpoint_mismatch(self):
        record = round_trip(plan_multi_start(*self.args, max_expanded=3,
                                            snapshot=True, trace=True))
        self.assertIn("checkpoint", record)
        record["checkpoint"]["costs"] = [[1] * 6 for _ in range(4)]
        report = verify_multi_start_trace(*self.args, record,
                                          max_expanded=3, snapshot=True)
        self.assertEqual(
            report,
            {"valid": False, "mismatch": "checkpoint", "index": None},
        )

    def test_checkpoint_missing_from_record(self):
        record = round_trip(plan_multi_start(*self.args, max_expanded=3,
                                            snapshot=True, trace=True))
        del record["checkpoint"]
        report = verify_multi_start_trace(*self.args, record,
                                          max_expanded=3, snapshot=True)
        self.assertEqual(report["mismatch"], "checkpoint")

    def test_checkpoint_valid(self):
        record = round_trip(plan_multi_start(*self.args, max_expanded=3,
                                            snapshot=True, trace=True))
        report = verify_multi_start_trace(*self.args, record,
                                         max_expanded=3, snapshot=True)
        self.assertTrue(report["valid"])

    def test_extra_checkpoint_ignored_without_snapshot(self):
        record = round_trip(plan_multi_start(*self.args, max_expanded=3,
                                            snapshot=True, trace=True))
        report = verify_multi_start_trace(*self.args, record,
                                         max_expanded=3)
        self.assertTrue(report["valid"])


class VerifyMultiStartDynamicTest(unittest.TestCase):
    args = (3, 3, set(), [(1, 0), (0, 0)], (2, 2))

    def test_dynamic_blocked(self):
        record = plan_multi_start(*self.args, dynamic_blocked=FRAMES,
                                  trace=True)
        self.assertTrue(all(len(node) == 3
                            for node in record["expanded_nodes"]))
        self.assertTrue(
            verify_multi_start_trace(
                *self.args, record, dynamic_blocked=FRAMES)["valid"]
        )
        self.assertTrue(
            verify_multi_start_trace(
                *self.args, round_trip(record),
                dynamic_blocked=FRAMES)["valid"]
        )

    def test_dynamic_blocked_mismatch_index(self):
        record = round_trip(plan_multi_start(
            *self.args, dynamic_blocked=FRAMES, trace=True))
        record["expanded_nodes"][0] = [1, 0, 1]
        report = verify_multi_start_trace(
            *self.args, record, dynamic_blocked=FRAMES)
        self.assertEqual(report["mismatch"], "expanded_nodes")
        self.assertEqual(report["index"], 0)

    def test_allow_wait(self):
        record = plan_multi_start(*self.args, dynamic_blocked=FRAMES,
                                  allow_wait=True, trace=True)
        self.assertTrue(
            verify_multi_start_trace(
                *self.args, record, dynamic_blocked=FRAMES,
                allow_wait=True)["valid"]
        )
        without_wait = verify_multi_start_trace(
            *self.args, record, dynamic_blocked=FRAMES)
        self.assertFalse(without_wait["valid"])

    def test_dynamic_costs_frames(self):
        frames = [
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [[2, 2, 2], [2, 1, 2], [2, 2, 2]],
        ]
        record = plan_multi_start(3, 3, set(), [(0, 0), (2, 2)], (2, 0),
                                  dynamic_costs=frames, trace=True)
        self.assertTrue(
            verify_multi_start_trace(
                3, 3, set(), [(0, 0), (2, 2)], (2, 0), record,
                dynamic_costs=frames)["valid"]
        )

    def test_reservations(self):
        reservations = [[(1, 0), (1, 1), (1, 2)]]
        record = plan_multi_start(3, 3, set(), [(0, 0), (2, 0)], (2, 2),
                                  reservations=reservations, trace=True)
        self.assertTrue(
            verify_multi_start_trace(
                3, 3, set(), [(0, 0), (2, 0)], (2, 2), record,
                reservations=reservations)["valid"]
        )

    def test_dynamic_unreachable_record(self):
        # A wall plus a frame that seals the only gap leaves no route for
        # any start.
        blocked = {(1, 0), (1, 2)}
        frames = [[(1, 1)]]
        record = plan_multi_start(3, 3, blocked, [(0, 0), (0, 2)], (2, 1),
                                  dynamic_blocked=frames, trace=True)
        self.assertIsNone(record["path"])
        self.assertTrue(
            verify_multi_start_trace(
                3, 3, blocked, [(0, 0), (0, 2)], (2, 1), record,
                dynamic_blocked=frames)["valid"]
        )


class VerifyMultiStartLimitsTest(unittest.TestCase):
    args = (6, 4, BLOCKED, STARTS, (5, 0))

    def test_max_cost_truncation(self):
        # The winning start needs cost 3; cost 2 cannot reach the goal.
        record = plan_multi_start(*self.args, max_cost=2, trace=True)
        self.assertEqual(record["status"], "cost_exhausted")
        self.assertIsNone(record["path"])
        self.assertTrue(
            verify_multi_start_trace(*self.args, record,
                                     max_cost=2)["valid"]
        )

    def test_max_expanded_zero_budget(self):
        args = (3, 3, set(), [(1, 1)], (1, 1))
        record = plan_multi_start(*args, max_expanded=0, trace=True)
        self.assertEqual(record["status"], "budget_exhausted")
        self.assertEqual(record["expanded"], 0)
        self.assertEqual(record["expanded_nodes"], [])
        self.assertTrue(
            verify_multi_start_trace(*args, record,
                                     max_expanded=0)["valid"]
        )

    def test_budgeted_traces_verify(self):
        full = plan_multi_start(*self.args, trace=True)
        for budget in range(0, full["expanded"] + 1):
            record = plan_multi_start(*self.args, trace=True,
                                      max_expanded=budget)
            self.assertTrue(
                verify_multi_start_trace(
                    *self.args, round_trip(record),
                    max_expanded=budget)["valid"],
                f"budget={budget}",
            )

    def test_max_steps_static_truncation(self):
        # The goal is four columns away; a two-step window cannot reach it.
        record = plan_multi_start(*self.args, max_steps=2, trace=True)
        self.assertEqual(record["status"], "step_exhausted")
        self.assertIsNone(record["path"])
        self.assertTrue(
            verify_multi_start_trace(*self.args, record,
                                     max_steps=2)["valid"]
        )

    def test_max_steps_dynamic_truncation(self):
        args = (3, 3, set(), [(0, 0)], (2, 2))
        record = plan_multi_start(*args, dynamic_blocked=FRAMES,
                                  max_steps=3, trace=True)
        self.assertEqual(record["status"], "step_exhausted")
        self.assertTrue(
            verify_multi_start_trace(*args, record,
                                     dynamic_blocked=FRAMES,
                                     max_steps=3)["valid"]
        )

    def test_snapshot_dynamic(self):
        args = (3, 3, set(), [(1, 0), (0, 0)], (2, 2))
        record = round_trip(plan_multi_start(
            *args, dynamic_blocked=FRAMES, max_expanded=3,
            snapshot=True, trace=True))
        self.assertEqual(record["checkpoint"]["planner"],
                         "plan_multi_start")
        self.assertTrue(
            verify_multi_start_trace(
                *args, record, dynamic_blocked=FRAMES, max_expanded=3,
                snapshot=True)["valid"]
        )


class VerifyMultiStartValidationTest(unittest.TestCase):
    def setUp(self):
        self.args = (6, 4, BLOCKED, STARTS, (5, 0))
        self.record = round_trip(plan_multi_start(*self.args, trace=True))

    def test_grid_errors_precede_record_errors(self):
        # A bad width raises before the missing-field record error.
        with self.assertRaises(TypeError):
            verify_multi_start_trace("6", 4, BLOCKED, STARTS, (5, 0), {})
        # Bad starts structure precedes the goal structure.
        with self.assertRaises(TypeError):
            verify_multi_start_trace(
                6, 4, BLOCKED, None, "goal", self.record)
        # An out-of-bounds start precedes an out-of-bounds goal (and the
        # record inspection).
        with self.assertRaises(ValueError):
            verify_multi_start_trace(
                6, 4, BLOCKED, [(9, 9)], (9, 9), self.record)
        # A start on an obstacle precedes record inspection.
        with self.assertRaises(ValueError):
            verify_multi_start_trace(
                6, 4, BLOCKED, [(2, 0)], (5, 0), self.record)
        # A blocked goal precedes record inspection.
        with self.assertRaises(ValueError):
            verify_multi_start_trace(
                6, 4, BLOCKED, STARTS, (2, 0), "record")
        # A bad budget precedes record inspection.
        with self.assertRaises(ValueError):
            verify_multi_start_trace(
                *self.args, [], max_expanded=-1)
        # A bad reservation precedes record inspection.
        with self.assertRaises(ValueError):
            verify_multi_start_trace(
                *self.args, [], reservations=[[(2, 0)]])
        # Empty starts raise before the record is inspected.
        with self.assertRaises(ValueError):
            verify_multi_start_trace(
                6, 4, BLOCKED, [], (5, 0), None)

    def test_frame_zero_check_precedes_record(self):
        with self.assertRaises(ValueError):
            verify_multi_start_trace(
                3, 3, set(), [(0, 0)], (2, 2), None,
                dynamic_blocked=[[(0, 0)]])

    def test_record_not_an_object(self):
        for bad in (None, [], "record", 42, True):
            with self.assertRaises(TypeError, msg=f"record={bad!r}"):
                verify_multi_start_trace(*self.args, bad)

    def test_record_missing_fields(self):
        for key in ("path", "cost", "expanded", "expanded_nodes"):
            record = dict(self.record)
            del record[key]
            with self.assertRaises(TypeError, msg=f"missing {key}"):
                verify_multi_start_trace(*self.args, record)

    def test_record_field_type_errors(self):
        for key, bad in (("path", "path"), ("cost", "cost"),
                         ("expanded", True), ("expanded_nodes", 7)):
            record = dict(self.record)
            record[key] = bad
            with self.assertRaises(TypeError, msg=f"bad {key}"):
                verify_multi_start_trace(*self.args, record)

    def test_record_path_shape_errors(self):
        for bad_entry in ("xy", [1], [1, 2, 3], [1, "y"], [True, 0]):
            record = dict(self.record)
            record["path"] = [bad_entry]
            with self.assertRaises(TypeError):
                verify_multi_start_trace(*self.args, record)

    def test_record_trajectory_shape_errors(self):
        for bad_entry in ("xyt", [1], [1, 2, 3, 4], [1, 2, "t"]):
            record = dict(self.record)
            record["expanded_nodes"] = [bad_entry]
            with self.assertRaises(TypeError):
                verify_multi_start_trace(*self.args, record)

    def test_record_coordinate_out_of_bounds(self):
        record = dict(self.record)
        record["path"] = [[9, 0]]
        with self.assertRaises(ValueError):
            verify_multi_start_trace(*self.args, record)
        record = dict(self.record)
        record["expanded_nodes"] = [[0, 9]]
        with self.assertRaises(ValueError):
            verify_multi_start_trace(*self.args, record)

    def test_record_trajectory_mode_mismatch(self):
        # A triple in static mode is a ValueError, not a mismatch.
        record = dict(self.record)
        record["expanded_nodes"] = [[0, 0, 0]]
        with self.assertRaises(ValueError):
            verify_multi_start_trace(*self.args, record)
        # A pair in dynamic mode is a ValueError too.
        dynamic = round_trip(plan_multi_start(
            3, 3, set(), [(1, 0), (0, 0)], (2, 2),
            dynamic_blocked=FRAMES, trace=True))
        dynamic["expanded_nodes"] = [[0, 0]]
        with self.assertRaises(ValueError):
            verify_multi_start_trace(
                3, 3, set(), [(1, 0), (0, 0)], (2, 2), dynamic,
                dynamic_blocked=FRAMES)

    def test_record_dynamic_negative_time(self):
        record = round_trip(plan_multi_start(
            3, 3, set(), [(1, 0), (0, 0)], (2, 2),
            dynamic_blocked=FRAMES, trace=True))
        record["expanded_nodes"][0] = [1, 0, -1]
        with self.assertRaises(ValueError):
            verify_multi_start_trace(
                3, 3, set(), [(1, 0), (0, 0)], (2, 2), record,
                dynamic_blocked=FRAMES)

    def test_record_static_duplicate_state(self):
        record = dict(self.record)
        record["expanded_nodes"] = (record["expanded_nodes"]
                                    + [record["expanded_nodes"][0]])
        with self.assertRaises(ValueError):
            verify_multi_start_trace(*self.args, record)

    def test_record_bad_checkpoint(self):
        record = dict(self.record)
        record["checkpoint"] = {"version": 1}
        with self.assertRaises(ValueError):
            verify_multi_start_trace(*self.args, record)
        record = dict(self.record)
        record["checkpoint"] = 42
        with self.assertRaises(ValueError):
            verify_multi_start_trace(*self.args, record)

    def test_record_bad_status_type(self):
        record = dict(self.record)
        record["status"] = 7
        with self.assertRaises(TypeError):
            verify_multi_start_trace(*self.args, record)

    def test_dynamic_mode_selected_by_dynamic_costs_only(self):
        # Time-varying costs alone select the dynamic (triple) mode.
        frames = [[[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                  [[2, 2, 2], [2, 1, 2], [2, 2, 2]]]
        record = round_trip(plan_multi_start(
            3, 3, set(), [(0, 0)], (2, 2),
            dynamic_costs=frames, trace=True))
        self.assertTrue(all(len(node) == 3
                            for node in record["expanded_nodes"]))
        record["expanded_nodes"][0] = [0, 0]
        with self.assertRaises(ValueError):
            verify_multi_start_trace(
                3, 3, set(), [(0, 0)], (2, 2), record,
                dynamic_costs=frames)


if __name__ == "__main__":
    unittest.main()
