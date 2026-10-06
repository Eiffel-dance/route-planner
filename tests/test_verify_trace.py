import copy
import json
import unittest

from app import plan, resume, verify_trace


BLOCKED = {(2, 0), (2, 1), (2, 2)}
FRAMES = [
    [(0, 1), (0, 2)],
    [(0, 1), (1, 2), (2, 0), (2, 2)],
    [(2, 1), (2, 2)],
    [(0, 0), (1, 2), (2, 1), (2, 2)],
    [],
    [(0, 1)],
]


class VerifyTraceValidTest(unittest.TestCase):
    def assert_valid(self, *args, **kwargs):
        outcome = verify_trace(*args, **kwargs)
        self.assertEqual(
            outcome, {"valid": True, "mismatch": None, "index": None}
        )

    def test_static_record_direct_and_json_round_trip(self):
        record = plan(6, 4, BLOCKED, (0, 0), (5, 3), trace=True)
        self.assert_valid(6, 4, BLOCKED, (0, 0), (5, 3), record=record)
        round_tripped = json.loads(json.dumps(record))
        self.assert_valid(6, 4, BLOCKED, (0, 0), (5, 3),
                          record=round_tripped)

    def test_unreachable(self):
        wall = {(1, y) for y in range(3)}
        record = plan(3, 3, wall, (0, 0), (2, 2), trace=True)
        self.assertIsNone(record["path"])
        self.assert_valid(3, 3, wall, (0, 0), (2, 2), record=record)

    def test_start_equals_goal(self):
        record = plan(3, 3, set(), (1, 1), (1, 1), trace=True)
        self.assert_valid(3, 3, set(), (1, 1), (1, 1), record=record)

    def test_costs_matrix(self):
        costs = [[1, 2, 1], [1, 9, 1], [1, 1, 1]]
        record = plan(3, 3, set(), (0, 0), (2, 2), costs=costs, trace=True)
        self.assert_valid(3, 3, set(), (0, 0), (2, 2), costs=costs,
                          record=record)

    def test_dynamic_blocked(self):
        record = plan(6, 4, BLOCKED, (0, 0), (5, 3), trace=True,
                      dynamic_blocked=FRAMES)
        self.assert_valid(6, 4, BLOCKED, (0, 0), (5, 3),
                          dynamic_blocked=FRAMES, record=record)

    def test_dynamic_costs(self):
        dynamic_costs = [
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [[9, 9, 9], [1, 1, 1], [1, 1, 1]],
        ]
        record = plan(3, 3, set(), (0, 0), (2, 2), trace=True,
                      dynamic_costs=dynamic_costs)
        self.assert_valid(3, 3, set(), (0, 0), (2, 2),
                          dynamic_costs=dynamic_costs, record=record)

    def test_allow_wait(self):
        frames = [[(1, 0)], [(1, 0)], []]
        record = plan(3, 3, set(), (0, 0), (2, 0), trace=True,
                      dynamic_blocked=frames, allow_wait=True)
        self.assertIn((1, 0), record["path"])
        self.assert_valid(3, 3, set(), (0, 0), (2, 0),
                          dynamic_blocked=frames, allow_wait=True,
                          record=record)

    def test_reservations(self):
        reservations = [[(1, 0), (1, 1), (1, 2)]]
        record = plan(3, 3, set(), (0, 0), (2, 2), trace=True,
                      reservations=reservations)
        self.assert_valid(3, 3, set(), (0, 0), (2, 2),
                          reservations=reservations, record=record)

    def test_budget_status(self):
        record = plan(6, 4, BLOCKED, (0, 0), (5, 3), trace=True,
                      max_expanded=3)
        self.assertEqual(record["status"], "budget_exhausted")
        self.assert_valid(6, 4, BLOCKED, (0, 0), (5, 3), max_expanded=3,
                          record=record)

    def test_max_cost_status(self):
        record = plan(6, 4, BLOCKED, (0, 0), (5, 3), trace=True, max_cost=2)
        self.assertEqual(record["status"], "cost_exhausted")
        self.assert_valid(6, 4, BLOCKED, (0, 0), (5, 3), max_cost=2,
                          record=record)

    def test_checkpoint(self):
        record = plan(6, 4, BLOCKED, (0, 0), (5, 3), trace=True,
                      max_expanded=3, snapshot=True)
        self.assertIn("checkpoint", record)
        self.assert_valid(6, 4, BLOCKED, (0, 0), (5, 3), max_expanded=3,
                          snapshot=True, record=record)
        self.assert_valid(6, 4, BLOCKED, (0, 0), (5, 3), max_expanded=3,
                          snapshot=True,
                          record=json.loads(json.dumps(record)))

    def test_dynamic_checkpoint(self):
        record = plan(6, 4, BLOCKED, (0, 0), (5, 3), trace=True,
                      dynamic_blocked=FRAMES, max_expanded=4, snapshot=True)
        self.assertEqual(record["status"], "budget_exhausted")
        self.assert_valid(6, 4, BLOCKED, (0, 0), (5, 3),
                          dynamic_blocked=FRAMES, max_expanded=4,
                          snapshot=True, record=record)

    def test_resume_record(self):
        first = plan(6, 4, BLOCKED, (0, 0), (5, 3), max_expanded=3,
                     snapshot=True)
        final = resume(first["checkpoint"], max_expanded=1000)
        self.assertEqual(final["status"], "found")
        self.assert_valid(6, 4, BLOCKED, (0, 0), (5, 3), max_expanded=1000,
                          snapshot=True, record=final)

    def test_extra_fields_ignored(self):
        record = plan(6, 4, BLOCKED, (0, 0), (5, 3), trace=True)
        decorated = dict(record, note="kept", status="ignored",
                         checkpoint="ignored")
        self.assert_valid(6, 4, BLOCKED, (0, 0), (5, 3), record=decorated)

    def test_key_order_and_container_types_irrelevant(self):
        record = plan(3, 3, set(), (0, 0), (2, 2), trace=True)
        shuffled = {
            "expanded_nodes": [list(entry) for entry in
                               record["expanded_nodes"]],
            "expanded": record["expanded"],
            "cost": record["cost"],
            "path": [list(point) for point in record["path"]],
        }
        self.assert_valid(3, 3, set(), (0, 0), (2, 2), record=shuffled)

    def test_result_is_json_serializable(self):
        record = plan(6, 4, BLOCKED, (0, 0), (5, 3), trace=True)
        for outcome in (
            verify_trace(6, 4, BLOCKED, (0, 0), (5, 3), record=record),
            verify_trace(6, 4, BLOCKED, (0, 0), (5, 3),
                         record=dict(record, cost=record["cost"] + 1)),
        ):
            self.assertEqual(json.loads(json.dumps(outcome)), outcome)

    def test_inputs_are_not_modified(self):
        record = json.loads(json.dumps(
            plan(6, 4, BLOCKED, (0, 0), (5, 3), trace=True,
                 max_expanded=3, snapshot=True)
        ))
        before = copy.deepcopy(record)
        verify_trace(6, 4, BLOCKED, (0, 0), (5, 3), max_expanded=3,
                     snapshot=True, record=record)
        self.assertEqual(record, before)


class VerifyTraceMismatchTest(unittest.TestCase):
    def setUp(self):
        self.args = (6, 4, BLOCKED, (0, 0), (5, 3))
        self.record = plan(*self.args, trace=True)

    def tampered(self, **changes):
        record = dict(self.record)
        record.update(changes)
        return record

    def test_path_mismatch_reports_first_differing_index(self):
        path = list(self.record["path"])
        path[1] = (0, 1) if path[1] != (0, 1) else (1, 0)
        outcome = verify_trace(*self.args, record=self.tampered(path=path))
        self.assertEqual(outcome,
                         {"valid": False, "mismatch": "path", "index": 1})

    def test_path_length_mismatch_reports_shorter_end(self):
        path = list(self.record["path"])[:-1]
        outcome = verify_trace(*self.args, record=self.tampered(path=path))
        self.assertEqual(outcome, {"valid": False, "mismatch": "path",
                                   "index": len(path)})

    def test_path_none_mismatch(self):
        outcome = verify_trace(*self.args, record=self.tampered(path=None))
        self.assertEqual(outcome,
                         {"valid": False, "mismatch": "path", "index": 0})

    def test_cost_mismatch(self):
        outcome = verify_trace(
            *self.args, record=self.tampered(cost=self.record["cost"] + 1)
        )
        self.assertEqual(outcome,
                         {"valid": False, "mismatch": "cost", "index": None})

    def test_expanded_mismatch(self):
        outcome = verify_trace(
            *self.args,
            record=self.tampered(expanded=self.record["expanded"] + 1)
        )
        self.assertEqual(outcome, {"valid": False, "mismatch": "expanded",
                                   "index": None})

    def test_expanded_nodes_mismatch_reports_index(self):
        nodes = list(self.record["expanded_nodes"])
        nodes[0] = (1, 1)
        outcome = verify_trace(*self.args,
                               record=self.tampered(expanded_nodes=nodes))
        self.assertEqual(outcome, {"valid": False,
                                   "mismatch": "expanded_nodes", "index": 0})

    def test_expanded_nodes_length_mismatch(self):
        nodes = list(self.record["expanded_nodes"])[:-1]
        outcome = verify_trace(*self.args,
                               record=self.tampered(expanded_nodes=nodes))
        self.assertEqual(outcome, {"valid": False,
                                   "mismatch": "expanded_nodes",
                                   "index": len(nodes)})

    def test_first_mismatch_in_fixed_order_wins(self):
        record = self.tampered(path=None, cost=999, expanded=999)
        outcome = verify_trace(*self.args, record=record)
        self.assertEqual(outcome["mismatch"], "path")
        record = self.tampered(cost=999, expanded=999)
        outcome = verify_trace(*self.args, record=record)
        self.assertEqual(outcome["mismatch"], "cost")

    def test_status_mismatch(self):
        record = plan(6, 4, BLOCKED, (0, 0), (5, 3), trace=True,
                      max_expanded=1000)
        tampered = dict(record, status="budget_exhausted")
        outcome = verify_trace(6, 4, BLOCKED, (0, 0), (5, 3),
                               max_expanded=1000, record=tampered)
        self.assertEqual(outcome, {"valid": False, "mismatch": "status",
                                   "index": None})

    def test_checkpoint_mismatch(self):
        record = plan(6, 4, BLOCKED, (0, 0), (5, 3), trace=True,
                      max_expanded=3, snapshot=True)
        other = plan(6, 4, BLOCKED, (0, 0), (5, 3), trace=True,
                     max_expanded=4, snapshot=True)
        tampered = dict(record, checkpoint=other["checkpoint"])
        outcome = verify_trace(6, 4, BLOCKED, (0, 0), (5, 3), max_expanded=3,
                               snapshot=True, record=tampered)
        self.assertEqual(outcome, {"valid": False, "mismatch": "checkpoint",
                                   "index": None})

    def test_missing_checkpoint_is_a_mismatch_not_an_error(self):
        record = plan(6, 4, BLOCKED, (0, 0), (5, 3), trace=True,
                      max_expanded=3, snapshot=True)
        del record["checkpoint"]
        outcome = verify_trace(6, 4, BLOCKED, (0, 0), (5, 3), max_expanded=3,
                               snapshot=True, record=record)
        self.assertEqual(outcome, {"valid": False, "mismatch": "checkpoint",
                                   "index": None})

    def test_dynamic_record_mismatch(self):
        record = plan(6, 4, BLOCKED, (0, 0), (5, 3), trace=True,
                      dynamic_blocked=FRAMES)
        nodes = list(record["expanded_nodes"])
        nodes[2] = (nodes[2][0], nodes[2][1], nodes[2][2] + 1)
        outcome = verify_trace(6, 4, BLOCKED, (0, 0), (5, 3),
                               dynamic_blocked=FRAMES,
                               record=dict(record, expanded_nodes=nodes))
        self.assertEqual(outcome, {"valid": False,
                                   "mismatch": "expanded_nodes", "index": 2})


class VerifyTraceValidationTest(unittest.TestCase):
    def test_grid_validation_precedes_record_checks(self):
        # A malformed record never skips grid or constraint validation.
        with self.assertRaises(ValueError):
            verify_trace(0, 4, BLOCKED, (0, 0), (5, 3), record=None)
        with self.assertRaises(TypeError):
            verify_trace(6, 4, BLOCKED, (0, 0), (5, 3), costs="x",
                         record=None)
        with self.assertRaises(ValueError):
            verify_trace(6, 4, BLOCKED, (0, 0), (5, 3), max_expanded=-1,
                         record=None)
        with self.assertRaises(TypeError):
            verify_trace(6, 4, BLOCKED, (0, 0), (5, 3), allow_wait=1,
                         record=None)
        with self.assertRaises(TypeError):
            verify_trace(6, 4, BLOCKED, (0, 0), (5, 3), snapshot=1,
                         record=None)
        with self.assertRaises(ValueError):
            verify_trace(6, 4, BLOCKED, (0, 0), (5, 3), max_cost=-2,
                         record=None)
        with self.assertRaises(ValueError):
            verify_trace(3, 3, set(), (0, 0), (2, 2), costs=[[1] * 3] * 3,
                         dynamic_costs=[[[1] * 3] * 3], record=None)

    def test_record_must_be_an_object(self):
        for bad in (None, 42, "record", b"record", [1, 2], (1, 2), 1.5):
            with self.assertRaises(TypeError, msg=f"record={bad!r}"):
                verify_trace(3, 3, set(), (0, 0), (2, 2), record=bad)

    def test_missing_required_fields(self):
        record = plan(3, 3, set(), (0, 0), (2, 2), trace=True)
        for key in ("path", "cost", "expanded", "expanded_nodes"):
            broken = {k: v for k, v in record.items() if k != key}
            with self.assertRaises(TypeError, msg=key):
                verify_trace(3, 3, set(), (0, 0), (2, 2), record=broken)

    def test_missing_status_when_limits_are_given(self):
        record = plan(3, 3, set(), (0, 0), (2, 2), trace=True,
                      max_expanded=10)
        del record["status"]
        with self.assertRaises(TypeError):
            verify_trace(3, 3, set(), (0, 0), (2, 2), max_expanded=10,
                         record=record)
        record = plan(3, 3, set(), (0, 0), (2, 2), trace=True, max_cost=9)
        del record["status"]
        with self.assertRaises(TypeError):
            verify_trace(3, 3, set(), (0, 0), (2, 2), max_cost=9,
                         record=record)

    def test_field_type_errors(self):
        record = plan(3, 3, set(), (0, 0), (2, 2), trace=True)
        cases = (
            ("path", 12), ("path", "abc"), ("path", [1, 2]),
            ("cost", "x"), ("cost", 1.5), ("cost", True),
            ("expanded", "3"), ("expanded", True), ("expanded", 2.0),
            ("expanded_nodes", 7), ("expanded_nodes", "xy"),
        )
        for key, bad in cases:
            with self.assertRaises(TypeError, msg=f"{key}={bad!r}"):
                verify_trace(3, 3, set(), (0, 0), (2, 2),
                             record=dict(record, **{key: bad}))

    def test_status_type_error(self):
        record = plan(3, 3, set(), (0, 0), (2, 2), trace=True,
                      max_expanded=10)
        for bad in (None, 1, True, ["found"]):
            with self.assertRaises(TypeError, msg=f"status={bad!r}"):
                verify_trace(3, 3, set(), (0, 0), (2, 2), max_expanded=10,
                             record=dict(record, status=bad))

    def test_path_shape_errors(self):
        record = plan(3, 3, set(), (0, 0), (2, 2), trace=True)
        for bad_entry in (5, "ab", [1], [1, 2, 3], [1, "x"], [True, 0]):
            with self.assertRaises(TypeError, msg=repr(bad_entry)):
                verify_trace(3, 3, set(), (0, 0), (2, 2),
                             record=dict(record, path=[bad_entry]))

    def test_trace_shape_errors(self):
        record = plan(3, 3, set(), (0, 0), (2, 2), trace=True)
        for bad_entry in (5, "ab", [1], [1, 2, 3, 4], [1, "x"], [0, True]):
            with self.assertRaises(TypeError, msg=repr(bad_entry)):
                verify_trace(3, 3, set(), (0, 0), (2, 2),
                             record=dict(record, expanded_nodes=[bad_entry]))

    def test_out_of_bounds_coordinates(self):
        record = plan(3, 3, set(), (0, 0), (2, 2), trace=True)
        with self.assertRaises(ValueError):
            verify_trace(3, 3, set(), (0, 0), (2, 2),
                         record=dict(record, path=[(0, 0), (3, 0)]))
        with self.assertRaises(ValueError):
            verify_trace(3, 3, set(), (0, 0), (2, 2),
                         record=dict(record,
                                     expanded_nodes=[(0, 0), (-1, 0)]))

    def test_trace_mode_mismatch(self):
        # Static mode never records triples.
        record = plan(3, 3, set(), (0, 0), (2, 2), trace=True)
        with self.assertRaises(ValueError):
            verify_trace(3, 3, set(), (0, 0), (2, 2),
                         record=dict(record, expanded_nodes=[(0, 0, 0)]))
        # Dynamic mode never records pairs.
        record = plan(6, 4, BLOCKED, (0, 0), (5, 3), trace=True,
                      dynamic_blocked=FRAMES)
        with self.assertRaises(ValueError):
            verify_trace(6, 4, BLOCKED, (0, 0), (5, 3),
                         dynamic_blocked=FRAMES,
                         record=dict(record, expanded_nodes=[(0, 0)]))
        # Negative times are not valid states either.
        with self.assertRaises(ValueError):
            verify_trace(6, 4, BLOCKED, (0, 0), (5, 3),
                         dynamic_blocked=FRAMES,
                         record=dict(record, expanded_nodes=[(0, 0, -1)]))

    def test_checkpoint_structure_errors(self):
        record = plan(6, 4, BLOCKED, (0, 0), (5, 3), trace=True,
                      max_expanded=3, snapshot=True)
        for bad in (None, 42, "cp", [], {"planner": "plan"}):
            with self.assertRaises(ValueError, msg=repr(bad)):
                verify_trace(6, 4, BLOCKED, (0, 0), (5, 3), max_expanded=3,
                             snapshot=True,
                             record=dict(record, checkpoint=bad))

    def test_checkpoint_inconsistent_state(self):
        record = plan(6, 4, BLOCKED, (0, 0), (5, 3), trace=True,
                      max_expanded=3, snapshot=True)
        checkpoint = json.loads(json.dumps(record["checkpoint"]))
        checkpoint["closed"] += 1
        with self.assertRaises(ValueError):
            verify_trace(6, 4, BLOCKED, (0, 0), (5, 3), max_expanded=3,
                         snapshot=True,
                         record=dict(record, checkpoint=checkpoint))

    def test_checkpoint_field_ignored_without_snapshot(self):
        record = plan(6, 4, BLOCKED, (0, 0), (5, 3), trace=True,
                      max_expanded=3)
        decorated = dict(record, checkpoint="not a checkpoint")
        outcome = verify_trace(6, 4, BLOCKED, (0, 0), (5, 3), max_expanded=3,
                               record=decorated)
        self.assertTrue(outcome["valid"])


if __name__ == "__main__":
    unittest.main()
