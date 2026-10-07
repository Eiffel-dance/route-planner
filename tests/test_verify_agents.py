import json
import unittest

from app import plan_agents, verify_agents


FRAMES = [[(1, 0)], [], []]
DYNAMIC_COSTS = [[[1, 1, 1], [1, 1, 1]], [[5, 5, 5], [1, 1, 1]]]
RESERVATIONS = [[(0, 1), (1, 1), (2, 1)]]
REQUESTS = [[(0, 0), (2, 0)], [(2, 0), (0, 0)]]


def round_trip(result):
    return json.loads(json.dumps(result))


class VerifyAgentsValidTest(unittest.TestCase):
    def test_valid_record_direct(self):
        record = plan_agents(3, 2, set(), REQUESTS, trace=True)
        self.assertEqual(
            verify_agents(3, 2, set(), REQUESTS, record),
            {"valid": True, "request_index": None,
             "mismatch": None, "index": None},
        )

    def test_valid_record_json_round_trip(self):
        record = round_trip(plan_agents(3, 2, set(), REQUESTS, trace=True))
        self.assertTrue(verify_agents(3, 2, set(), REQUESTS, record)["valid"])

    def test_mixed_sequences_and_extra_keys(self):
        record = plan_agents(3, 2, set(), REQUESTS, trace=True)
        for item in record["results"]:
            item["path"] = [list(point) for point in item["path"]]
            item["expanded_nodes"] = [
                list(entry) for entry in item["expanded_nodes"]
            ]
            item["something_else"] = {"ignored": True}
        record["batch_note"] = "ignored"
        self.assertTrue(verify_agents(3, 2, set(), REQUESTS, record)["valid"])

    def test_single_agent(self):
        record = plan_agents(3, 3, {(1, 1)}, [[(0, 0), (2, 2)]], trace=True)
        self.assertTrue(
            verify_agents(3, 3, {(1, 1)}, [[(0, 0), (2, 2)]], record)["valid"]
        )

    def test_start_equals_goal(self):
        record = plan_agents(3, 2, set(), [[(0, 0), (2, 0)], [(1, 1), (1, 1)]],
                             trace=True)
        self.assertTrue(
            verify_agents(3, 2, set(), [[(0, 0), (2, 0)], [(1, 1), (1, 1)]],
                          record)["valid"]
        )

    def test_unreachable_record(self):
        blocked = {(1, 2), (2, 1)}
        requests = [[(0, 0), (1, 0)], [(0, 2), (2, 2)], [(0, 0), (1, 1)]]
        record = plan_agents(3, 3, blocked, requests, trace=True)
        self.assertEqual(record["status"], "unreachable")
        self.assertEqual(record["failed_index"], 1)
        self.assertTrue(
            verify_agents(3, 3, blocked, requests, record)["valid"]
        )
        self.assertTrue(
            verify_agents(3, 3, blocked, requests,
                          round_trip(record))["valid"]
        )

    def test_first_agent_unreachable(self):
        blocked = {(1, 0), (0, 1)}
        requests = [[(0, 0), (2, 2)], [(1, 1), (2, 2)]]
        record = plan_agents(3, 3, blocked, requests, trace=True)
        self.assertEqual(record["failed_index"], 0)
        self.assertTrue(
            verify_agents(3, 3, blocked, requests, record)["valid"]
        )

    def test_report_is_json_serializable(self):
        record = plan_agents(3, 2, set(), REQUESTS, trace=True)
        report = verify_agents(3, 2, set(), REQUESTS, record)
        self.assertEqual(json.loads(json.dumps(report)), report)
        record["results"][1]["cost"] += 1
        report = verify_agents(3, 2, set(), REQUESTS, record)
        self.assertEqual(json.loads(json.dumps(report)), report)

    def test_record_is_not_modified(self):
        record = round_trip(plan_agents(3, 2, set(), REQUESTS, trace=True))
        before = json.dumps(record, sort_keys=True)
        verify_agents(3, 2, set(), REQUESTS, record)
        self.assertEqual(json.dumps(record, sort_keys=True), before)


class VerifyAgentsMismatchTest(unittest.TestCase):
    def setUp(self):
        self.args = (3, 2, set(), REQUESTS)
        self.record = round_trip(plan_agents(*self.args, trace=True))

    def test_path_mismatch_reports_request_and_index(self):
        self.record["results"][1]["path"][2] = list(
            self.record["results"][1]["path"][1]
        )
        report = verify_agents(*self.args, self.record)
        self.assertEqual(
            report,
            {"valid": False, "request_index": 1,
             "mismatch": "path", "index": 2},
        )

    def test_earlier_agent_wins(self):
        self.record["results"][0]["cost"] += 1
        self.record["results"][1]["path"] = None
        report = verify_agents(*self.args, self.record)
        self.assertEqual(
            report,
            {"valid": False, "request_index": 0,
             "mismatch": "cost", "index": None},
        )

    def test_field_order_path_before_cost(self):
        item = self.record["results"][1]
        item["path"] = [list(point) for point in item["path"]]
        item["path"][1], item["path"][2] = item["path"][2], item["path"][1]
        item["cost"] = 99
        report = verify_agents(*self.args, self.record)
        self.assertEqual(report["request_index"], 1)
        self.assertEqual(report["mismatch"], "path")
        self.assertEqual(report["index"], 1)

    def test_expanded_mismatch(self):
        self.record["results"][0]["expanded"] += 1
        report = verify_agents(*self.args, self.record)
        self.assertEqual(
            report,
            {"valid": False, "request_index": 0,
             "mismatch": "expanded", "index": None},
        )

    def test_expanded_nodes_mismatch_index(self):
        nodes = self.record["results"][1]["expanded_nodes"]
        nodes[1], nodes[2] = nodes[2], nodes[1]
        report = verify_agents(*self.args, self.record)
        self.assertEqual(
            report,
            {"valid": False, "request_index": 1,
             "mismatch": "expanded_nodes", "index": 1},
        )

    def test_batch_status_mismatch(self):
        # The record claims success while the recomputed batch fails on
        # the very first agent; the recorded prefix still matches, so
        # the batch status is the first difference.
        blocked = {(1, 0), (0, 1)}
        requests = [[(0, 0), (2, 2)], [(1, 1), (2, 2)]]
        actual = plan_agents(3, 3, blocked, requests, trace=True)
        self.assertEqual(actual["status"], "unreachable")
        record = round_trip(actual)
        record["status"] = "found"
        # The second entry must be a legal dynamic-mode record (triples);
        # it is never compared because the recomputed batch stops at the
        # first agent.
        record["results"] = [
            record["results"][0],
            {"path": None, "cost": None, "expanded": 0,
             "expanded_nodes": []},
        ]
        del record["failed_index"]
        report = verify_agents(3, 3, blocked, requests, record)
        self.assertEqual(
            report,
            {"valid": False, "request_index": None,
             "mismatch": "status", "index": None},
        )

    def test_batch_failed_index_mismatch(self):
        blocked = {(1, 2), (2, 1)}
        requests = [[(0, 0), (1, 0)], [(0, 2), (2, 2)], [(0, 0), (1, 1)]]
        record = round_trip(plan_agents(3, 3, blocked, requests, trace=True))
        self.assertEqual(record["failed_index"], 1)
        record["failed_index"] = 0
        record["results"] = record["results"][:1]
        record["expanded"] = record["results"][0]["expanded"]
        report = verify_agents(3, 3, blocked, requests, record)
        self.assertEqual(
            report,
            {"valid": False, "request_index": None,
             "mismatch": "failed_index", "index": None},
        )

    def test_batch_expanded_mismatch(self):
        self.record["expanded"] += 1
        report = verify_agents(*self.args, self.record)
        self.assertEqual(
            report,
            {"valid": False, "request_index": None,
             "mismatch": "expanded", "index": None},
        )


class VerifyAgentsValidationTest(unittest.TestCase):
    def setUp(self):
        self.record = round_trip(plan_agents(3, 2, set(), REQUESTS,
                                             trace=True))

    def test_grid_errors_precede_record_errors(self):
        # A bad width raises before the missing-results record error.
        with self.assertRaises(TypeError):
            verify_agents("3", 2, set(), REQUESTS, {})
        # A bad request coordinate raises before the record is inspected.
        with self.assertRaises(ValueError):
            verify_agents(3, 2, set(), [[(0, 0), (9, 9)]], None)
        # A request endpoint on an obstacle raises first.
        with self.assertRaises(ValueError):
            verify_agents(3, 2, {(1, 0)}, [[(0, 0), (1, 0)]], "record")
        # A start blocked at frame 0 raises before the record is inspected.
        with self.assertRaises(ValueError):
            verify_agents(3, 2, set(), REQUESTS, [],
                          dynamic_blocked=[[(0, 0)]])
        # A bad waiting flag raises before the record is inspected.
        with self.assertRaises(TypeError):
            verify_agents(3, 2, set(), REQUESTS, [], allow_wait=1)
        # Providing both costs and dynamic_costs raises first.
        with self.assertRaises(ValueError):
            verify_agents(3, 2, set(), REQUESTS, [],
                          costs=[[1, 1, 1], [1, 1, 1]],
                          dynamic_costs=[[[1, 1, 1], [1, 1, 1]]])
        # A bad reservation raises before the record is inspected.
        with self.assertRaises(ValueError):
            verify_agents(3, 2, set(), REQUESTS, [], reservations=[[]])

    def test_record_not_an_object(self):
        for bad in (None, [], "record", 42, True):
            with self.assertRaises(TypeError):
                verify_agents(3, 2, set(), REQUESTS, bad)

    def test_record_missing_fields(self):
        for key in ("status", "results", "expanded"):
            record = round_trip(self.record)
            del record[key]
            with self.assertRaises(TypeError, msg=key):
                verify_agents(3, 2, set(), REQUESTS, record)

    def test_record_status_type_error(self):
        for bad in (None, 7, True, ["found"]):
            record = round_trip(self.record)
            record["status"] = bad
            with self.assertRaises(TypeError, msg=repr(bad)):
                verify_agents(3, 2, set(), REQUESTS, record)

    def test_record_unknown_status_value_error(self):
        record = round_trip(self.record)
        record["status"] = "maybe"
        with self.assertRaises(ValueError):
            verify_agents(3, 2, set(), REQUESTS, record)

    def test_record_expanded_type_error(self):
        for bad in (None, "3", 2.5, True):
            record = round_trip(self.record)
            record["expanded"] = bad
            with self.assertRaises(TypeError, msg=repr(bad)):
                verify_agents(3, 2, set(), REQUESTS, record)

    def test_record_results_not_a_sequence(self):
        for bad in (None, "results", 42, True):
            record = round_trip(self.record)
            record["results"] = bad
            with self.assertRaises(TypeError, msg=repr(bad)):
                verify_agents(3, 2, set(), REQUESTS, record)

    def test_found_record_results_length_mismatch(self):
        for bad in ([], self.record["results"][:1],
                    self.record["results"] + [self.record["results"][0]]):
            record = round_trip(self.record)
            record["results"] = bad
            with self.assertRaises(ValueError):
                verify_agents(3, 2, set(), REQUESTS, record)

    def test_unreachable_record_requires_failed_index(self):
        blocked = {(1, 2), (2, 1)}
        requests = [[(0, 0), (1, 0)], [(0, 2), (2, 2)]]
        record = round_trip(plan_agents(3, 3, blocked, requests, trace=True))
        self.assertEqual(record["status"], "unreachable")
        del record["failed_index"]
        with self.assertRaises(TypeError):
            verify_agents(3, 3, blocked, requests, record)

    def test_failed_index_type_error(self):
        blocked = {(1, 2), (2, 1)}
        requests = [[(0, 0), (1, 0)], [(0, 2), (2, 2)]]
        record = round_trip(plan_agents(3, 3, blocked, requests, trace=True))
        for bad in (None, "1", 1.5, True):
            record["failed_index"] = bad
            with self.assertRaises(TypeError, msg=repr(bad)):
                verify_agents(3, 3, blocked, requests, record)

    def test_failed_index_out_of_range(self):
        blocked = {(1, 2), (2, 1)}
        requests = [[(0, 0), (1, 0)], [(0, 2), (2, 2)]]
        record = round_trip(plan_agents(3, 3, blocked, requests, trace=True))
        for bad in (-1, 2, 10):
            record["failed_index"] = bad
            with self.assertRaises(ValueError, msg=repr(bad)):
                verify_agents(3, 3, blocked, requests, record)

    def test_unreachable_record_results_length_mismatch(self):
        blocked = {(1, 2), (2, 1)}
        requests = [[(0, 0), (1, 0)], [(0, 2), (2, 2)], [(0, 0), (1, 1)]]
        record = round_trip(plan_agents(3, 3, blocked, requests, trace=True))
        self.assertEqual(record["failed_index"], 1)
        # The failure covers exactly failed_index + 1 entries.
        record["results"] = record["results"][:1]
        with self.assertRaises(ValueError):
            verify_agents(3, 3, blocked, requests, record)
        record = round_trip(plan_agents(3, 3, blocked, requests, trace=True))
        record["results"] = record["results"] + [record["results"][0]]
        with self.assertRaises(ValueError):
            verify_agents(3, 3, blocked, requests, record)

    def test_entry_not_an_object(self):
        record = round_trip(self.record)
        record["results"][1] = None
        with self.assertRaises(TypeError):
            verify_agents(3, 2, set(), REQUESTS, record)

    def test_entry_missing_fields(self):
        for key in ("path", "cost", "expanded", "expanded_nodes"):
            record = round_trip(self.record)
            del record["results"][1][key]
            with self.assertRaises(TypeError, msg=key):
                verify_agents(3, 2, set(), REQUESTS, record)

    def test_entry_field_type_errors(self):
        for key, bad in (("path", "path"), ("cost", "cost"),
                         ("expanded", True), ("expanded_nodes", 7)):
            record = round_trip(self.record)
            record["results"][0][key] = bad
            with self.assertRaises(TypeError, msg=key):
                verify_agents(3, 2, set(), REQUESTS, record)

    def test_entry_path_shape_errors(self):
        for bad_entry in ("xy", [1], [1, 2, 3], [1, "y"], [True, 0]):
            record = round_trip(self.record)
            record["results"][0]["path"] = [bad_entry]
            with self.assertRaises(TypeError, msg=repr(bad_entry)):
                verify_agents(3, 2, set(), REQUESTS, record)

    def test_entry_trajectory_shape_errors(self):
        for bad_entry in ("xyt", [1], [1, 2, 3, 4], [1, 2, "t"]):
            record = round_trip(self.record)
            record["results"][0]["expanded_nodes"] = [bad_entry]
            with self.assertRaises(TypeError, msg=repr(bad_entry)):
                verify_agents(3, 2, set(), REQUESTS, record)

    def test_entry_coordinate_out_of_bounds(self):
        record = round_trip(self.record)
        record["results"][0]["path"] = [[9, 0]]
        with self.assertRaises(ValueError):
            verify_agents(3, 2, set(), REQUESTS, record)
        record = round_trip(self.record)
        record["results"][0]["expanded_nodes"] = [[0, 9]]
        with self.assertRaises(ValueError):
            verify_agents(3, 2, set(), REQUESTS, record)

    def test_entry_trajectory_mode_mismatch(self):
        # The first agent searches the static grid (pairs); a triple in
        # its trace is a ValueError, not a mismatch.
        record = round_trip(self.record)
        record["results"][0]["expanded_nodes"] = [[0, 0, 0]]
        with self.assertRaises(ValueError):
            verify_agents(3, 2, set(), REQUESTS, record)
        # Later agents search the time-expanded space (triples); a pair
        # in the second agent's trace is a ValueError too.
        record = round_trip(self.record)
        record["results"][1]["expanded_nodes"] = [[2, 0]]
        with self.assertRaises(ValueError):
            verify_agents(3, 2, set(), REQUESTS, record)

    def test_entry_static_duplicate_state(self):
        record = round_trip(self.record)
        nodes = record["results"][0]["expanded_nodes"]
        record["results"][0]["expanded_nodes"] = nodes + [nodes[0]]
        with self.assertRaises(ValueError):
            verify_agents(3, 2, set(), REQUESTS, record)

    def test_entry_negative_time(self):
        record = round_trip(self.record)
        record["results"][1]["expanded_nodes"] = [[2, 0, -1]]
        with self.assertRaises(ValueError):
            verify_agents(3, 2, set(), REQUESTS, record)


class VerifyAgentsModesTest(unittest.TestCase):
    def test_costs(self):
        costs = [[1, 10, 1], [1, 1, 1]]
        record = plan_agents(3, 2, set(), REQUESTS, costs=costs, trace=True)
        self.assertTrue(
            verify_agents(3, 2, set(), REQUESTS, record, costs=costs)["valid"]
        )
        self.assertTrue(
            verify_agents(3, 2, set(), REQUESTS, round_trip(record),
                          costs=costs)["valid"]
        )

    def test_dynamic_blocked(self):
        requests = [[(0, 0), (2, 0)], [(0, 1), (2, 1)]]
        record = plan_agents(3, 2, set(), requests,
                             dynamic_blocked=FRAMES, trace=True)
        self.assertTrue(
            verify_agents(3, 2, set(), requests, record,
                          dynamic_blocked=FRAMES)["valid"]
        )
        self.assertTrue(
            verify_agents(3, 2, set(), requests, round_trip(record),
                          dynamic_blocked=FRAMES)["valid"]
        )

    def test_dynamic_blocked_mismatch(self):
        requests = [[(0, 0), (2, 0)], [(0, 1), (2, 1)]]
        record = round_trip(plan_agents(3, 2, set(), requests,
                                        dynamic_blocked=FRAMES, trace=True))
        record["results"][0]["expanded_nodes"][0] = [0, 0, 1]
        report = verify_agents(3, 2, set(), requests, record,
                               dynamic_blocked=FRAMES)
        self.assertEqual(report["request_index"], 0)
        self.assertEqual(report["mismatch"], "expanded_nodes")
        self.assertEqual(report["index"], 0)

    def test_allow_wait(self):
        requests = [[(0, 0), (2, 0)], [(2, 0), (0, 1)]]
        record = plan_agents(3, 2, set(), requests, allow_wait=True,
                             trace=True)
        self.assertTrue(
            verify_agents(3, 2, set(), requests, record,
                          allow_wait=True)["valid"]
        )

    def test_dynamic_costs(self):
        requests = [[(0, 0), (2, 0)], [(0, 1), (2, 1)]]
        record = plan_agents(3, 2, set(), requests,
                             dynamic_costs=DYNAMIC_COSTS, allow_wait=True,
                             trace=True)
        self.assertEqual(record["status"], "unreachable")
        self.assertTrue(
            verify_agents(3, 2, set(), requests, record,
                          dynamic_costs=DYNAMIC_COSTS,
                          allow_wait=True)["valid"]
        )

    def test_reservations(self):
        record = plan_agents(3, 2, set(), REQUESTS,
                             reservations=RESERVATIONS, trace=True)
        self.assertTrue(
            verify_agents(3, 2, set(), REQUESTS, record,
                          reservations=RESERVATIONS)["valid"]
        )
        self.assertTrue(
            verify_agents(3, 2, set(), REQUESTS, round_trip(record),
                          reservations=RESERVATIONS)["valid"]
        )

    def test_reservation_tail_blocks_goal(self):
        record = plan_agents(3, 3, set(), [[(0, 0), (2, 1)]],
                             reservations=RESERVATIONS, trace=True)
        self.assertEqual(record["status"], "unreachable")
        self.assertEqual(record["failed_index"], 0)
        self.assertTrue(
            verify_agents(3, 3, set(), [[(0, 0), (2, 1)]], record,
                          reservations=RESERVATIONS)["valid"]
        )

    def test_three_agents(self):
        requests = [[(0, 0), (3, 3)], [(3, 0), (0, 3)], [(0, 3), (3, 0)]]
        record = plan_agents(4, 4, set(), requests, trace=True)
        self.assertEqual(record["status"], "found")
        self.assertTrue(verify_agents(4, 4, set(), requests, record)["valid"])
        self.assertTrue(
            verify_agents(4, 4, set(), requests, round_trip(record))["valid"]
        )


if __name__ == "__main__":
    unittest.main()
