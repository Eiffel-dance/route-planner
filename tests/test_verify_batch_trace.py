import json
import unittest

from app import plan_batch, verify_batch_trace


BLOCKED = {(2, 0), (2, 1), (2, 2)}
REQUESTS = [[(0, 0), (5, 3)], [(5, 3), (0, 0)], [(1, 1), (1, 1)]]
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


class VerifyBatchTraceStaticTest(unittest.TestCase):
    def test_valid_record_direct(self):
        record = plan_batch(6, 4, BLOCKED, REQUESTS, trace=True)
        self.assertEqual(
            verify_batch_trace(6, 4, BLOCKED, REQUESTS, record),
            {"valid": True, "request_index": None, "mismatch": None,
             "index": None},
        )

    def test_valid_record_json_round_trip(self):
        record = round_trip(plan_batch(6, 4, BLOCKED, REQUESTS, trace=True))
        self.assertTrue(
            verify_batch_trace(6, 4, BLOCKED, REQUESTS, record)["valid"]
        )

    def test_mixed_sequences_and_extra_keys(self):
        record = plan_batch(6, 4, BLOCKED, REQUESTS, trace=True)
        for item in record["results"]:
            item["path"] = [list(point) for point in item["path"]]
            item["expanded_nodes"] = [
                list(entry) for entry in item["expanded_nodes"]
            ]
            item["something_else"] = {"ignored": True}
        record["something_else"] = {"ignored": True}
        self.assertTrue(
            verify_batch_trace(6, 4, BLOCKED, REQUESTS, record)["valid"]
        )

    def test_unreachable_and_empty_results(self):
        blocked = {(1, y) for y in range(3)}
        requests = [[(0, 0), (2, 2)], [(0, 0), (0, 0)]]
        record = plan_batch(3, 3, blocked, requests, trace=True)
        self.assertIsNone(record["results"][0]["path"])
        self.assertTrue(
            verify_batch_trace(3, 3, blocked, requests, record)["valid"]
        )

    def test_duplicate_requests(self):
        requests = [[(0, 0), (5, 3)], [(0, 0), (5, 3)]]
        record = plan_batch(6, 4, BLOCKED, requests, trace=True)
        self.assertTrue(
            verify_batch_trace(6, 4, BLOCKED, requests, record)["valid"]
        )

    def test_costs_matrix_record(self):
        costs = [[1, 2, 1], [3, 1, 4], [1, 1, 1]]
        requests = [[(0, 0), (2, 2)], [(2, 2), (0, 0)]]
        record = plan_batch(3, 3, set(), requests, costs=costs, trace=True)
        self.assertTrue(
            verify_batch_trace(3, 3, set(), requests, record,
                               costs=costs)["valid"]
        )

    def test_result_is_json_serializable(self):
        record = plan_batch(6, 4, BLOCKED, REQUESTS, trace=True)
        report = verify_batch_trace(6, 4, BLOCKED, REQUESTS, record)
        self.assertEqual(json.loads(json.dumps(report)), report)
        record["results"][1]["cost"] += 1
        report = verify_batch_trace(6, 4, BLOCKED, REQUESTS, record)
        self.assertEqual(json.loads(json.dumps(report)), report)

    def test_record_is_not_modified(self):
        record = round_trip(plan_batch(6, 4, BLOCKED, REQUESTS, trace=True))
        before = json.dumps(record, sort_keys=True)
        verify_batch_trace(6, 4, BLOCKED, REQUESTS, record)
        self.assertEqual(json.dumps(record, sort_keys=True), before)


class VerifyBatchTraceMismatchTest(unittest.TestCase):
    def setUp(self):
        self.args = (6, 4, BLOCKED, REQUESTS)
        self.record = round_trip(plan_batch(*self.args, trace=True))

    def test_path_mismatch_reports_request_and_element_index(self):
        self.record["results"][1]["path"][2] = (
            list(self.record["results"][1]["path"][1])
        )
        report = verify_batch_trace(*self.args, self.record)
        self.assertEqual(
            report, {"valid": False, "request_index": 1,
                     "mismatch": "path", "index": 2}
        )

    def test_smaller_request_index_wins(self):
        self.record["results"][0]["cost"] += 1
        self.record["results"][1]["path"][0] = [9, 9]
        # [9, 9] is out of bounds, so use an in-bounds wrong cell instead.
        self.record["results"][1]["path"][0] = [1, 0]
        report = verify_batch_trace(*self.args, self.record)
        self.assertEqual(
            report, {"valid": False, "request_index": 0,
                     "mismatch": "cost", "index": None}
        )

    def test_cost_mismatch(self):
        self.record["results"][2]["cost"] += 1
        report = verify_batch_trace(*self.args, self.record)
        self.assertEqual(
            report, {"valid": False, "request_index": 2,
                     "mismatch": "cost", "index": None}
        )

    def test_expanded_mismatch(self):
        self.record["results"][0]["expanded"] += 1
        report = verify_batch_trace(*self.args, self.record)
        self.assertEqual(
            report, {"valid": False, "request_index": 0,
                     "mismatch": "expanded", "index": None}
        )

    def test_expanded_nodes_mismatch_index(self):
        nodes = self.record["results"][1]["expanded_nodes"]
        nodes[1], nodes[2] = nodes[2], nodes[1]
        report = verify_batch_trace(*self.args, self.record)
        self.assertEqual(
            report, {"valid": False, "request_index": 1,
                     "mismatch": "expanded_nodes", "index": 1}
        )

    def test_field_order_path_before_cost(self):
        item = self.record["results"][0]
        item["path"][1] = list(item["path"][0])
        item["cost"] += 5
        report = verify_batch_trace(*self.args, self.record)
        self.assertEqual(report["mismatch"], "path")

    def test_status_mismatch(self):
        record = round_trip(plan_batch(6, 4, BLOCKED, REQUESTS,
                                       max_expanded=3, trace=True))
        record["results"][0]["status"] = "found"
        report = verify_batch_trace(6, 4, BLOCKED, REQUESTS, record,
                                    max_expanded=3)
        self.assertEqual(
            report, {"valid": False, "request_index": 0,
                     "mismatch": "status", "index": None}
        )

    def test_status_missing_from_record(self):
        record = round_trip(plan_batch(6, 4, BLOCKED, REQUESTS,
                                       max_expanded=3, trace=True))
        del record["results"][1]["status"]
        report = verify_batch_trace(6, 4, BLOCKED, REQUESTS, record,
                                    max_expanded=3)
        self.assertEqual(report["request_index"], 1)
        self.assertEqual(report["mismatch"], "status")

    def test_checkpoint_mismatch(self):
        record = round_trip(plan_batch(6, 4, BLOCKED, REQUESTS,
                                       max_expanded=3, snapshot=True,
                                       trace=True))
        item = record["results"][0]
        self.assertIn("checkpoint", item)
        item["checkpoint"]["costs"] = [[1] * 6 for _ in range(4)]
        report = verify_batch_trace(6, 4, BLOCKED, REQUESTS, record,
                                    max_expanded=3, snapshot=True)
        self.assertEqual(
            report, {"valid": False, "request_index": 0,
                     "mismatch": "checkpoint", "index": None}
        )

    def test_checkpoint_missing_from_record(self):
        record = round_trip(plan_batch(6, 4, BLOCKED, REQUESTS,
                                       max_expanded=3, snapshot=True,
                                       trace=True))
        del record["results"][0]["checkpoint"]
        report = verify_batch_trace(6, 4, BLOCKED, REQUESTS, record,
                                    max_expanded=3, snapshot=True)
        self.assertEqual(report["mismatch"], "checkpoint")

    def test_checkpoint_valid(self):
        record = round_trip(plan_batch(6, 4, BLOCKED, REQUESTS,
                                       max_expanded=3, snapshot=True,
                                       trace=True))
        report = verify_batch_trace(6, 4, BLOCKED, REQUESTS, record,
                                    max_expanded=3, snapshot=True)
        self.assertTrue(report["valid"])


class VerifyBatchTraceValidationTest(unittest.TestCase):
    def setUp(self):
        self.record = round_trip(plan_batch(6, 4, BLOCKED, REQUESTS,
                                            trace=True))

    def test_grid_errors_precede_record_errors(self):
        with self.assertRaises(TypeError):
            verify_batch_trace("6", 4, BLOCKED, REQUESTS, {})
        with self.assertRaises(ValueError):
            verify_batch_trace(6, 4, BLOCKED, [[(0, 0), (9, 9)]], None)
        with self.assertRaises(ValueError):
            verify_batch_trace(6, 4, BLOCKED, [[(0, 0), (2, 0)]], "record")
        with self.assertRaises(ValueError):
            verify_batch_trace(6, 4, BLOCKED, REQUESTS, [], max_expanded=-1)
        with self.assertRaises(ValueError):
            verify_batch_trace(6, 4, BLOCKED, REQUESTS, [],
                               reservations=[[(2, 0)]])

    def test_record_not_an_object(self):
        for bad in (None, [], "record", 42, True):
            with self.assertRaises(TypeError):
                verify_batch_trace(6, 4, BLOCKED, REQUESTS, bad)

    def test_record_missing_results(self):
        with self.assertRaises(TypeError):
            verify_batch_trace(6, 4, BLOCKED, REQUESTS, {})

    def test_record_results_not_a_sequence(self):
        for bad in ("results", 7, None):
            with self.assertRaises(TypeError):
                verify_batch_trace(6, 4, BLOCKED, REQUESTS,
                                   {"results": bad})

    def test_record_results_length_mismatch(self):
        record = {"results": self.record["results"][:-1]}
        with self.assertRaises(ValueError):
            verify_batch_trace(6, 4, BLOCKED, REQUESTS, record)
        record = {"results": self.record["results"] * 2}
        with self.assertRaises(ValueError):
            verify_batch_trace(6, 4, BLOCKED, REQUESTS, record)

    def test_entry_missing_fields(self):
        for key in ("path", "cost", "expanded", "expanded_nodes"):
            record = {"results": [dict(item) for item in
                                  self.record["results"]]}
            del record["results"][1][key]
            with self.assertRaises(TypeError):
                verify_batch_trace(6, 4, BLOCKED, REQUESTS, record)

    def test_entry_field_type_errors(self):
        for key, bad in (("path", "path"), ("cost", "cost"),
                         ("expanded", True), ("expanded_nodes", 7)):
            record = {"results": [dict(item) for item in
                                  self.record["results"]]}
            record["results"][0][key] = bad
            with self.assertRaises(TypeError):
                verify_batch_trace(6, 4, BLOCKED, REQUESTS, record)

    def test_entry_path_shape_errors(self):
        for bad_entry in ("xy", [1], [1, 2, 3], [1, "y"], [True, 0]):
            record = {"results": [dict(item) for item in
                                  self.record["results"]]}
            record["results"][0]["path"] = [bad_entry]
            with self.assertRaises(TypeError):
                verify_batch_trace(6, 4, BLOCKED, REQUESTS, record)

    def test_entry_trajectory_mode_mismatch(self):
        record = {"results": [dict(item) for item in
                              self.record["results"]]}
        record["results"][0]["expanded_nodes"] = [[0, 0, 0]]
        with self.assertRaises(ValueError):
            verify_batch_trace(6, 4, BLOCKED, REQUESTS, record)

    def test_entry_static_duplicate_state(self):
        record = {"results": [dict(item) for item in
                              self.record["results"]]}
        nodes = record["results"][0]["expanded_nodes"]
        record["results"][0]["expanded_nodes"] = nodes + [nodes[0]]
        with self.assertRaises(ValueError):
            verify_batch_trace(6, 4, BLOCKED, REQUESTS, record)

    def test_entry_bad_checkpoint(self):
        record = {"results": [dict(item) for item in
                              self.record["results"]]}
        record["results"][0]["checkpoint"] = {"version": 1}
        with self.assertRaises(ValueError):
            verify_batch_trace(6, 4, BLOCKED, REQUESTS, record)


class VerifyBatchTraceModesTest(unittest.TestCase):
    def test_dynamic_blocked(self):
        record = plan_batch(6, 4, BLOCKED, REQUESTS,
                            dynamic_blocked=FRAMES, trace=True)
        self.assertTrue(
            verify_batch_trace(6, 4, BLOCKED, REQUESTS, record,
                               dynamic_blocked=FRAMES)["valid"]
        )
        self.assertTrue(
            verify_batch_trace(6, 4, BLOCKED, REQUESTS,
                               round_trip(record),
                               dynamic_blocked=FRAMES)["valid"]
        )

    def test_dynamic_mode_mismatch(self):
        record = round_trip(plan_batch(6, 4, BLOCKED, REQUESTS,
                                       dynamic_blocked=FRAMES, trace=True))
        record["results"][1]["expanded_nodes"][0] = [0, 0, 1]
        report = verify_batch_trace(6, 4, BLOCKED, REQUESTS, record,
                                    dynamic_blocked=FRAMES)
        self.assertEqual(report["request_index"], 1)
        self.assertEqual(report["mismatch"], "expanded_nodes")
        self.assertEqual(report["index"], 0)

    def test_dynamic_pair_in_record_raises(self):
        record = round_trip(plan_batch(6, 4, BLOCKED, REQUESTS,
                                       dynamic_blocked=FRAMES, trace=True))
        record["results"][0]["expanded_nodes"] = [[0, 0]]
        with self.assertRaises(ValueError):
            verify_batch_trace(6, 4, BLOCKED, REQUESTS, record,
                               dynamic_blocked=FRAMES)

    def test_allow_wait(self):
        record = plan_batch(6, 4, BLOCKED, REQUESTS,
                            dynamic_blocked=FRAMES, allow_wait=True,
                            trace=True)
        self.assertTrue(
            verify_batch_trace(6, 4, BLOCKED, REQUESTS, record,
                               dynamic_blocked=FRAMES,
                               allow_wait=True)["valid"]
        )

    def test_dynamic_costs(self):
        frames = [
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [[2, 2, 2], [2, 1, 2], [2, 2, 2]],
        ]
        requests = [[(0, 0), (2, 2)], [(2, 0), (0, 2)]]
        record = plan_batch(3, 3, set(), requests,
                            dynamic_costs=frames, trace=True)
        self.assertTrue(
            verify_batch_trace(3, 3, set(), requests, record,
                               dynamic_costs=frames)["valid"]
        )

    def test_reservations(self):
        reservations = [[(1, 0), (1, 1), (1, 2)]]
        requests = [[(0, 0), (2, 2)], [(2, 2), (0, 0)]]
        record = plan_batch(3, 3, set(), requests,
                            reservations=reservations, trace=True)
        self.assertTrue(
            verify_batch_trace(3, 3, set(), requests, record,
                               reservations=reservations)["valid"]
        )

    def test_max_cost(self):
        record = plan_batch(6, 4, BLOCKED, REQUESTS, max_cost=4, trace=True)
        self.assertTrue(
            verify_batch_trace(6, 4, BLOCKED, REQUESTS, record,
                               max_cost=4)["valid"]
        )

    def test_max_expanded_zero_budget(self):
        requests = [[(1, 1), (1, 1)], [(0, 0), (2, 2)]]
        record = plan_batch(3, 3, set(), requests, max_expanded=0,
                            trace=True)
        self.assertEqual(record["results"][0]["status"],
                         "budget_exhausted")
        self.assertTrue(
            verify_batch_trace(3, 3, set(), requests, record,
                               max_expanded=0)["valid"]
        )


if __name__ == "__main__":
    unittest.main()
