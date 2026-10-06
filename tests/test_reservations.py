import json
import unittest

import app
from app import (plan, plan_any, plan_batch, plan_k, plan_multi_start,
                 replay, resume)


class BackwardCompatTest(unittest.TestCase):
    def test_omitted_none_and_empty_are_identical(self):
        args = (4, 3, {(1, 1)}, (0, 0), (3, 2))
        base = plan(*args, trace=True)
        self.assertEqual(plan(*args, reservations=None, trace=True), base)
        self.assertEqual(plan(*args, reservations=[], trace=True), base)
        self.assertEqual(plan(*args, reservations=(), trace=True), base)

    def test_empty_reservations_keep_static_result_shape(self):
        result = plan(3, 3, set(), (0, 0), (2, 2), reservations=[])
        # Static mode: plain coordinate pairs in the trace, no new keys.
        self.assertEqual(
            result,
            {"path": [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)], "cost": 4,
             "expanded": result["expanded"]},
        )

    def test_other_entry_points_untouched_by_empty(self):
        self.assertEqual(
            plan_any(3, 3, set(), (0, 0), [(2, 0), (0, 2)]),
            plan_any(3, 3, set(), (0, 0), [(2, 0), (0, 2)], reservations=[]),
        )
        self.assertEqual(
            plan_k(3, 3, set(), (0, 0), (2, 2), 3),
            plan_k(3, 3, set(), (0, 0), (2, 2), 3, reservations=[]),
        )
        self.assertEqual(
            plan_multi_start(3, 3, set(), [(0, 0), (0, 2)], (2, 1)),
            plan_multi_start(3, 3, set(), [(0, 0), (0, 2)], (2, 1),
                             reservations=[]),
        )
        self.assertEqual(
            plan_batch(3, 3, set(), [[(0, 0), (2, 2)]]),
            plan_batch(3, 3, set(), [[(0, 0), (2, 2)]], reservations=[]),
        )
        self.assertEqual(
            replay(3, 3, set(), (0, 0), (2, 2),
                   [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]),
            replay(3, 3, set(), (0, 0), (2, 2),
                   [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)],
                   reservations=[]),
        )


class ValidationTest(unittest.TestCase):
    def test_type_errors(self):
        bad_values = ["path", b"path", 42, {(0, 0)}]
        for value in bad_values:
            with self.assertRaises(TypeError, msg=repr(value)):
                plan(3, 3, set(), (0, 0), (2, 2), reservations=value)
        # A path that is not a sequence.
        with self.assertRaises(TypeError):
            plan(3, 3, set(), (0, 0), (2, 2), reservations=["abc"])
        with self.assertRaises(TypeError):
            plan(3, 3, set(), (0, 0), (2, 2), reservations=[None])
        # Coordinate structure: not exactly two integers.
        with self.assertRaises(TypeError):
            plan(3, 3, set(), (0, 0), (2, 2), reservations=[[(1,)]])
        with self.assertRaises(TypeError):
            plan(3, 3, set(), (0, 0), (2, 2), reservations=[[(1, 0, 0)]])
        with self.assertRaises(TypeError):
            plan(3, 3, set(), (0, 0), (2, 2), reservations=[[(1.5, 0)]])
        with self.assertRaises(TypeError):
            plan(3, 3, set(), (0, 0), (2, 2), reservations=[[(True, 0)]])

    def test_value_errors(self):
        # Empty path.
        with self.assertRaises(ValueError):
            plan(3, 3, set(), (0, 0), (2, 2), reservations=[[]])
        # Out-of-bounds coordinate.
        with self.assertRaises(ValueError):
            plan(3, 3, set(), (0, 0), (2, 2), reservations=[[(3, 0)]])
        with self.assertRaises(ValueError):
            plan(3, 3, set(), (0, 0), (2, 2), reservations=[[(0, -1)]])
        # Static obstacle coordinate.
        with self.assertRaises(ValueError):
            plan(3, 3, {(1, 1)}, (0, 0), (2, 2), reservations=[[(1, 1)]])
        # Non-consecutive repeat inside one path.
        with self.assertRaises(ValueError):
            plan(3, 3, set(), (0, 0), (2, 2),
                 reservations=[[(0, 0), (1, 0), (0, 0)]])

    def test_consecutive_stays_are_kept(self):
        # A consecutive stay inside a reservation path is legal input.
        result = plan(3, 3, set(), (0, 1), (2, 1),
                      reservations=[[(1, 1), (1, 1)]])
        self.assertEqual(result["cost"], 4)
        self.assertNotIn((1, 1), result["path"])

    def test_validation_runs_in_every_entry_point(self):
        for call in (
            lambda: plan_any(3, 3, set(), (0, 0), [(2, 2)],
                             reservations=[[]]),
            lambda: plan_batch(3, 3, set(), [[(0, 0), (2, 2)]],
                               reservations=[[]]),
            lambda: plan_multi_start(3, 3, set(), [(0, 0)], (2, 2),
                                     reservations=[[]]),
            lambda: plan_k(3, 3, set(), (0, 0), (2, 2), 1,
                           reservations=[[]]),
            lambda: replay(3, 3, set(), (0, 0), (2, 2), [(0, 0), (2, 2)],
                           reservations=[[]]),
        ):
            with self.assertRaises(ValueError):
                call()

    def test_validation_order_after_existing_checks(self):
        # Bad grid argument wins over bad reservations.
        with self.assertRaises(TypeError):
            plan("3", 3, set(), (0, 0), (2, 2), reservations=[[]])
        # Bad allow_wait (TypeError) wins over bad reservations.
        with self.assertRaises(TypeError):
            plan(3, 3, set(), (0, 0), (2, 2), allow_wait=1,
                 reservations=[[]])
        # costs + dynamic_costs conflict wins over bad reservations.
        with self.assertRaises(ValueError):
            plan(3, 3, set(), (0, 0), (2, 2),
                 costs=[[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                 dynamic_costs=[[[1, 1, 1], [1, 1, 1], [1, 1, 1]]],
                 reservations=[[]])

    def test_duplicates_and_entry_order_do_not_matter(self):
        res_a = [[(1, 1), (1, 1)], [(2, 0), (2, 1)]]
        res_b = [[(2, 0), (2, 1)], [(1, 1), (1, 1)], [(1, 1), (1, 1)]]
        self.assertEqual(
            plan(4, 4, set(), (0, 0), (3, 3), reservations=res_a, trace=True),
            plan(4, 4, set(), (0, 0), (3, 3), reservations=res_b, trace=True),
        )


class ReservationSemanticsTest(unittest.TestCase):
    def test_reserved_vertex_forces_detour(self):
        # (1, 1) is occupied at frames 0 and 1 and then persists.
        result = plan(3, 3, set(), (0, 1), (2, 1),
                      reservations=[[(1, 1), (1, 1)]])
        self.assertEqual(result["cost"], 4)
        self.assertNotIn((1, 1), result["path"])
        self.assertEqual(result["path"][0], (0, 1))
        self.assertEqual(result["path"][-1], (2, 1))

    def test_persistent_tail_blocks_forever(self):
        # The reservation's last coordinate persists: the goal is
        # unreachable on a 1-wide corridor.
        result = plan(3, 1, set(), (0, 0), (2, 0), reservations=[[(2, 0)]])
        self.assertIsNone(result["path"])
        self.assertIsNone(result["cost"])

    def test_timed_vertex_only_blocks_its_frame(self):
        # (1, 0) is reserved only at frame 0; the direct route arrives
        # there at frame 1 and is unaffected.
        result = plan(3, 2, set(), (0, 0), (2, 0),
                      reservations=[[(1, 0), (2, 1)]])
        self.assertEqual(result["path"], [(0, 0), (1, 0), (2, 0)])
        self.assertEqual(result["cost"], 2)

    def test_reverse_edge_is_forbidden(self):
        # The other agent moves (1, 0) -> (0, 0) between frames 0 and 1;
        # the candidate may not move (0, 0) -> (1, 0) on that transition
        # even though both cells are free at the candidate's times.
        res = [[(1, 0), (0, 0)]]
        result = plan(2, 2, set(), (0, 0), (1, 0), reservations=res)
        self.assertEqual(result["path"], [(0, 0), (0, 1), (1, 1), (1, 0)])
        self.assertEqual(result["cost"], 3)

    def test_head_on_corridor_is_unreachable(self):
        # Two agents cannot swap places in a 1-wide corridor.
        result = plan(2, 1, set(), (0, 0), (1, 0),
                      reservations=[[(1, 0), (0, 0)]], allow_wait=True)
        self.assertIsNone(result["path"])

    def test_start_reserved_at_frame_zero(self):
        result = plan(3, 3, set(), (0, 0), (2, 2), reservations=[[(0, 0)]])
        self.assertEqual(
            result, {"path": None, "cost": None, "expanded": 0}
        )

    def test_waiting_with_reservations_only(self):
        # (1, 0) is reserved at frame 1 only; waiting one frame at the
        # start yields the cheapest route at no entering-cell cost.
        res = [[(0, 1), (1, 0), (2, 1)]]
        result = plan(3, 2, set(), (0, 0), (2, 0), allow_wait=True,
                      reservations=res)
        self.assertEqual(result["path"], [(0, 0), (0, 0), (1, 0), (2, 0)])
        self.assertEqual(result["cost"], 2)
        # Without waiting the route must detour around the reservation.
        detour = plan(3, 2, set(), (0, 0), (2, 0), reservations=res)
        self.assertGreater(detour["cost"], 2)

    def test_wait_only_within_provided_frames(self):
        # The reservation provides frames 0..1; a stay at frame 2 is
        # beyond the provided frames and is rejected by replay.
        res = [[(0, 1), (1, 1)]]
        path = [(0, 0), (0, 0), (0, 0), (1, 0), (2, 0)]
        result = replay(3, 2, set(), (0, 0), (2, 0), path,
                        allow_wait=True, reservations=res, diagnose=True)
        self.assertEqual(result["error"], "wait_after_final_frame")
        self.assertEqual(result["error_index"], 2)

    def test_trace_records_space_time_triples(self):
        result = plan(3, 2, set(), (0, 0), (2, 0),
                      reservations=[[(0, 1), (1, 0), (2, 1)]], trace=True)
        self.assertTrue(all(len(node) == 3 for node in result["expanded_nodes"]))
        self.assertEqual(result["expanded"], len(result["expanded_nodes"]))
        self.assertEqual(result["expanded_nodes"][0], (0, 0, 0))

    def test_combines_with_dynamic_blocked(self):
        # (1, 0) is dynamically blocked at frame 1 and the row-1 detour
        # cell (0, 1) is reserved forever, so the route waits one frame
        # at the start and then takes the corridor.
        res = [[(0, 1), (0, 1)]]
        frames = [set(), {(1, 0)}, set()]
        result = plan(3, 2, set(), (0, 0), (2, 0), allow_wait=True,
                      dynamic_blocked=frames, reservations=res)
        self.assertEqual(result["path"], [(0, 0), (0, 0), (1, 0), (2, 0)])
        self.assertEqual(result["cost"], 2)

    def test_combines_with_costs(self):
        costs = [[1, 9, 1], [1, 1, 1]]
        result = plan(3, 2, set(), (0, 0), (2, 0), costs=costs,
                      reservations=[[(1, 0), (1, 0)]])
        # (1, 0) is reserved forever; the route goes through row 1.
        self.assertEqual(result["path"],
                         [(0, 0), (0, 1), (1, 1), (2, 1), (2, 0)])
        self.assertEqual(result["cost"], 4)

    def test_max_cost_status(self):
        result = plan(3, 3, set(), (0, 1), (2, 1),
                      reservations=[[(1, 1), (1, 1)]], max_cost=2)
        self.assertEqual(result["status"], "cost_exhausted")
        self.assertIsNone(result["path"])
        found = plan(3, 3, set(), (0, 1), (2, 1),
                     reservations=[[(1, 1), (1, 1)]], max_cost=4)
        self.assertEqual(found["status"], "found")
        self.assertEqual(found["cost"], 4)

    def test_max_expanded_status_and_prefix_trace(self):
        limited = plan(4, 4, set(), (0, 0), (3, 3),
                       reservations=[[(1, 1), (1, 1)]],
                       max_expanded=2, trace=True)
        self.assertEqual(limited["status"], "budget_exhausted")
        self.assertEqual(limited["expanded"], 2)
        full = plan(4, 4, set(), (0, 0), (3, 3),
                    reservations=[[(1, 1), (1, 1)]], trace=True)
        self.assertEqual(full["expanded_nodes"][:2],
                         limited["expanded_nodes"])


class OtherEntryPointsTest(unittest.TestCase):
    def test_plan_any(self):
        # The direct route to (2, 0) is reserved forever; (0, 2) wins.
        result = plan_any(3, 3, set(), (0, 0), [(2, 0), (0, 2)],
                          reservations=[[(1, 0), (1, 0)]])
        self.assertEqual(result["path"], [(0, 0), (0, 1), (0, 2)])
        self.assertEqual(result["cost"], 2)

    def test_plan_any_trace_triples(self):
        result = plan_any(3, 3, set(), (0, 0), [(2, 2)],
                          reservations=[[(1, 1), (1, 1)]], trace=True)
        self.assertTrue(all(len(node) == 3
                            for node in result["expanded_nodes"]))

    def test_plan_batch_matches_independent_calls(self):
        res = [[(1, 1), (1, 1)]]
        requests = [[(0, 1), (2, 1)], [(0, 0), (2, 2)]]
        batch = plan_batch(3, 3, set(), requests, reservations=res)
        for (start, goal), result in zip(requests, batch["results"]):
            self.assertEqual(
                result, plan(3, 3, set(), start, goal, reservations=res)
            )

    def test_plan_batch_validates_before_any_search(self):
        with self.assertRaises(ValueError):
            plan_batch(3, 3, set(), [[(0, 0), (2, 2)]], reservations=[[]])

    def test_plan_multi_start_skips_reserved_start(self):
        # (0, 0) is reserved at frame 0, so only (0, 2) can root a route.
        result = plan_multi_start(3, 3, set(), [(0, 0), (0, 2)], (2, 1),
                                  reservations=[[(0, 0), (0, 0)]])
        self.assertEqual(result["path"], [(0, 2), (0, 1), (1, 1), (2, 1)])
        self.assertEqual(result["cost"], 3)

    def test_plan_multi_start_all_starts_reserved(self):
        result = plan_multi_start(3, 3, set(), [(0, 0)], (2, 2),
                                  reservations=[[(0, 0)]])
        self.assertIsNone(result["path"])
        self.assertEqual(result["expanded"], 0)

    def test_plan_k(self):
        result = plan_k(3, 3, set(), (0, 1), (2, 1), 3,
                        reservations=[[(1, 1), (1, 1)]])
        self.assertEqual(result["costs"], [4, 4])
        self.assertEqual(len(result["paths"]), 2)
        for path, cost in zip(result["paths"], result["costs"]):
            self.assertNotIn((1, 1), path)
            self.assertEqual(cost, len(path) - 1)
        # The first entry is the route plan returns.
        self.assertEqual(
            result["paths"][0],
            plan(3, 3, set(), (0, 1), (2, 1),
                 reservations=[[(1, 1), (1, 1)]])["path"],
        )

    def test_plan_k_with_wait(self):
        res = [[(0, 1), (1, 0), (2, 1)]]
        result = plan_k(3, 2, set(), (0, 0), (2, 0), 2, allow_wait=True,
                        reservations=res)
        # plan_k ranks by total cost then the complete sequence, so the
        # lexicographically smallest zero-cost-wait route comes first.
        self.assertEqual(result["paths"][0],
                         [(0, 0), (0, 0), (0, 0), (1, 0), (2, 0)])
        self.assertEqual(result["costs"][0], 2)
        for path in result["paths"]:
            self.assertNotEqual(path[1] if len(path) > 1 else None, (1, 0))

    def test_plan_k_limits(self):
        result = plan_k(3, 3, set(), (0, 1), (2, 1), 3,
                        reservations=[[(1, 1), (1, 1)]], max_expanded=1)
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(result["paths"], [])
        self.assertEqual(result["expanded"], 1)


class SnapshotResumeTest(unittest.TestCase):
    def test_checkpoint_records_normalized_reservations(self):
        res = [[(2, 0), (2, 1)], [(1, 1), (1, 1)], [(2, 0), (2, 1)]]
        result = plan(4, 4, set(), (0, 0), (3, 3), reservations=res,
                      max_expanded=3, snapshot=True)
        self.assertEqual(result["status"], "budget_exhausted")
        checkpoint = result["checkpoint"]
        # Duplicates merged and the list sorted.
        self.assertEqual(checkpoint["reservations"],
                         [[[1, 1], [1, 1]], [[2, 0], [2, 1]]])
        json.dumps(checkpoint)  # fully JSON-serializable

    def test_checkpoint_omits_field_without_reservations(self):
        result = plan(4, 4, set(), (0, 0), (3, 3),
                      max_expanded=3, snapshot=True)
        self.assertNotIn("reservations", result["checkpoint"])

    def test_resume_matches_uninterrupted(self):
        res = [[(1, 1), (1, 1)], [(2, 0), (2, 1)]]
        stopped = plan(4, 4, set(), (0, 0), (3, 3), reservations=res,
                       max_expanded=3, snapshot=True)
        resumed = resume(stopped["checkpoint"], max_expanded=500)
        full = plan(4, 4, set(), (0, 0), (3, 3), reservations=res,
                    max_expanded=500, trace=True)
        for key in ("path", "cost", "expanded", "status", "expanded_nodes"):
            self.assertEqual(resumed[key], full[key], msg=key)

    def test_resume_plan_any_and_multi_start(self):
        res = [[(1, 0), (1, 0)]]
        stopped = plan_any(3, 3, set(), (0, 0), [(2, 0), (0, 2)],
                           reservations=res, max_expanded=1, snapshot=True)
        self.assertEqual(stopped["checkpoint"]["reservations"],
                         [[[1, 0], [1, 0]]])
        resumed = resume(stopped["checkpoint"], max_expanded=100)
        full = plan_any(3, 3, set(), (0, 0), [(2, 0), (0, 2)],
                        reservations=res, max_expanded=100, trace=True)
        for key in ("path", "cost", "expanded", "status", "expanded_nodes"):
            self.assertEqual(resumed[key], full[key], msg=key)

        stopped = plan_multi_start(4, 4, set(), [(0, 0), (0, 3)], (3, 3),
                                   reservations=res, max_expanded=2,
                                   snapshot=True)
        resumed = resume(stopped["checkpoint"], max_expanded=200)
        full = plan_multi_start(4, 4, set(), [(0, 0), (0, 3)], (3, 3),
                                reservations=res, max_expanded=200,
                                trace=True)
        for key in ("path", "cost", "expanded", "status", "expanded_nodes"):
            self.assertEqual(resumed[key], full[key], msg=key)

    def test_resume_with_wait_and_reservations(self):
        res = [[(0, 1), (1, 0), (2, 1)]]
        stopped = plan(3, 2, set(), (0, 0), (2, 0), allow_wait=True,
                       reservations=res, max_expanded=2, snapshot=True)
        self.assertEqual(stopped["status"], "budget_exhausted")
        resumed = resume(stopped["checkpoint"], max_expanded=100)
        full = plan(3, 2, set(), (0, 0), (2, 0), allow_wait=True,
                    reservations=res, max_expanded=100, trace=True)
        for key in ("path", "cost", "expanded", "status", "expanded_nodes"):
            self.assertEqual(resumed[key], full[key], msg=key)
        self.assertEqual(resumed["path"], [(0, 0), (0, 0), (1, 0), (2, 0)])

    def test_batch_checkpoint_records_reservations(self):
        res = [[(1, 1), (1, 1)]]
        result = plan_batch(4, 4, set(), [[(0, 0), (3, 3)]],
                            reservations=res, max_expanded=3, snapshot=True)
        checkpoint = result["results"][0]["checkpoint"]
        self.assertEqual(checkpoint["reservations"], [[[1, 1], [1, 1]]])
        resumed = resume(checkpoint, max_expanded=500)
        full = plan(4, 4, set(), (0, 0), (3, 3), reservations=res,
                    max_expanded=500, trace=True)
        for key in ("path", "cost", "expanded", "status", "expanded_nodes"):
            self.assertEqual(resumed[key], full[key], msg=key)

    def test_corrupt_checkpoint_reservations(self):
        stopped = plan(4, 4, set(), (0, 0), (3, 3),
                       reservations=[[(1, 1), (1, 1)]],
                       max_expanded=3, snapshot=True)
        checkpoint = stopped["checkpoint"]
        for bad, error in (
            ("nope", TypeError),
            ([[]], ValueError),
            ([[[9, 9]]], ValueError),
            ([[[0, 0], [1, 0], [0, 0]]], ValueError),
            ([[[1.5, 0]]], TypeError),
        ):
            corrupt = dict(checkpoint)
            corrupt["reservations"] = bad
            with self.assertRaises(error, msg=repr(bad)):
                resume(corrupt)

    def test_resumed_checkpoint_roundtrip(self):
        # A budget stop on a resumed search snapshots again with the
        # same reservations and can be resumed in turn.
        res = [[(1, 1), (1, 1)], [(2, 0), (2, 1)]]
        first = plan(4, 4, set(), (0, 0), (3, 3), reservations=res,
                     max_expanded=2, snapshot=True)
        second = resume(first["checkpoint"], max_expanded=4)
        self.assertEqual(second["status"], "budget_exhausted")
        self.assertEqual(second["checkpoint"]["reservations"],
                         [[[1, 1], [1, 1]], [[2, 0], [2, 1]]])
        final = resume(second["checkpoint"], max_expanded=500)
        full = plan(4, 4, set(), (0, 0), (3, 3), reservations=res,
                    max_expanded=500, trace=True)
        for key in ("path", "cost", "expanded", "status", "expanded_nodes"):
            self.assertEqual(final[key], full[key], msg=key)


class ReplayTest(unittest.TestCase):
    def test_successful_path_verifies(self):
        res = [[(1, 1), (1, 1)]]
        result = plan(3, 3, set(), (0, 1), (2, 1), reservations=res)
        check = replay(3, 3, set(), (0, 1), (2, 1), result["path"],
                       reservations=res)
        self.assertEqual(check, {"valid": True, "cost": result["cost"],
                                 "steps": len(result["path"]) - 1})

    def test_successful_wait_path_verifies(self):
        res = [[(0, 1), (1, 0), (2, 1)]]
        result = plan(3, 2, set(), (0, 0), (2, 0), allow_wait=True,
                      reservations=res)
        check = replay(3, 2, set(), (0, 0), (2, 0), result["path"],
                       allow_wait=True, reservations=res)
        self.assertEqual(check, {"valid": True, "cost": 2, "steps": 3})

    def test_reservation_vertex_diagnostic(self):
        result = replay(3, 1, set(), (0, 0), (2, 0), [(0, 0), (1, 0), (2, 0)],
                        reservations=[[(2, 0), (1, 0)]], diagnose=True)
        self.assertEqual(result, {"valid": False, "cost": None,
                                  "steps": None,
                                  "error": "reservation_vertex",
                                  "error_index": 1})

    def test_reservation_vertex_persistent_tail(self):
        # The final coordinate persists: frame 5 still sees (1, 0).
        result = replay(3, 1, set(), (0, 0), (2, 0), [(0, 0), (1, 0), (2, 0)],
                        reservations=[[(1, 0)]], diagnose=True)
        self.assertEqual(result["error"], "reservation_vertex")
        self.assertEqual(result["error_index"], 1)

    def test_reservation_edge_diagnostic(self):
        # The other agent moves (1, 0) -> (0, 0) between frames 0 and 1;
        # the candidate's first move traverses it in reverse.
        result = replay(3, 1, set(), (0, 0), (2, 0), [(0, 0), (1, 0), (2, 0)],
                        reservations=[[(1, 0), (0, 0)]], diagnose=True)
        self.assertEqual(result, {"valid": False, "cost": None,
                                  "steps": None,
                                  "error": "reservation_edge",
                                  "error_index": 1})

    def test_first_violation_index(self):
        # (1, 0) is reserved at frame 3 and (2, 0) from frame 4 on; the
        # first conflict the path meets is the vertex at index 3.
        res = [[(2, 1), (2, 1), (2, 1), (1, 0), (2, 0)]]
        path = [(0, 0), (0, 1), (1, 1), (1, 0), (2, 0)]
        result = replay(3, 2, set(), (0, 0), (2, 0), path,
                        reservations=res, diagnose=True)
        self.assertEqual(result["error"], "reservation_vertex")
        self.assertEqual(result["error_index"], 3)

    def test_invalid_without_diagnose_keeps_fixed_structure(self):
        result = replay(3, 1, set(), (0, 0), (2, 0), [(0, 0), (1, 0), (2, 0)],
                        reservations=[[(1, 0)]])
        self.assertEqual(result, {"valid": False, "cost": None,
                                  "steps": None})

    def test_other_errors_keep_existing_codes(self):
        # A statically blocked cell is still reported as such.
        result = replay(3, 2, {(1, 0)}, (0, 0), (2, 0),
                        [(0, 0), (1, 0), (2, 0)],
                        reservations=[[(0, 1), (0, 1)]], diagnose=True)
        self.assertEqual(result["error"], "static_blocked")
        # A repeated coordinate is still reported as such.
        result = replay(3, 2, set(), (0, 0), (2, 0),
                        [(0, 0), (1, 0), (1, 1), (1, 0), (2, 0)],
                        reservations=[[(0, 1), (0, 1)]], diagnose=True)
        self.assertEqual(result["error"], "repeated_coordinate")

    def test_start_equals_goal_reserved(self):
        result = replay(3, 3, set(), (1, 1), (1, 1), [(1, 1)],
                        reservations=[[(1, 1)]], diagnose=True)
        self.assertEqual(result["error"], "reservation_vertex")
        self.assertEqual(result["error_index"], 0)

    def test_replay_validation_order(self):
        # Bad reservations raise before the path structure is inspected.
        with self.assertRaises(ValueError):
            replay(3, 3, set(), (0, 0), (2, 2), "not-a-path",
                   reservations=[[]])
        with self.assertRaises(TypeError):
            replay(3, 3, set(), (0, 0), (2, 2), [(0, 0), (2, 2)],
                   reservations="nope")


if __name__ == "__main__":
    unittest.main()
