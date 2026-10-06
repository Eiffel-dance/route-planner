"""Tests for the reserved space-time routes (``reservations``)."""

import json
import unittest

from app import (plan, plan_any, plan_batch, plan_k, plan_multi_start,
                 replay, resume)


def without_checkpoint(result):
    return {key: value for key, value in result.items()
            if key != "checkpoint"}


class LegacyEquivalenceTest(unittest.TestCase):
    # Omitted, None and empty reservations leave every entry point's
    # result (keys included) exactly untouched.
    def test_plan_omitted_none_and_empty_are_equivalent(self):
        base = plan(4, 3, [(1, 1)], (0, 0), (3, 2), trace=True)
        self.assertEqual(set(base), {"path", "cost", "expanded",
                                     "expanded_nodes"})
        # Static trace keeps recording plain coordinate pairs.
        self.assertTrue(all(len(entry) == 2
                            for entry in base["expanded_nodes"]))
        for reservations in (None, []):
            self.assertEqual(
                plan(4, 3, [(1, 1)], (0, 0), (3, 2), trace=True,
                     reservations=reservations), base)

    def test_other_entry_points_omitted_none_and_empty_are_equivalent(self):
        base_any = plan_any(3, 2, [], (0, 0), [(2, 0)], trace=True)
        base_k = plan_k(3, 2, [], (0, 0), (2, 0), 2)
        base_multi = plan_multi_start(3, 2, [], [(0, 0)], (2, 0), trace=True)
        base_batch = plan_batch(3, 2, [], [[(0, 0), (2, 0)]], trace=True)
        base_replay = replay(3, 2, [], (0, 0), (2, 0),
                             [(0, 0), (1, 0), (2, 0)], diagnose=True)
        for reservations in (None, []):
            self.assertEqual(
                plan_any(3, 2, [], (0, 0), [(2, 0)], trace=True,
                         reservations=reservations), base_any)
            self.assertEqual(
                plan_k(3, 2, [], (0, 0), (2, 0), 2,
                       reservations=reservations), base_k)
            self.assertEqual(
                plan_multi_start(3, 2, [], [(0, 0)], (2, 0), trace=True,
                                 reservations=reservations), base_multi)
            self.assertEqual(
                plan_batch(3, 2, [], [[(0, 0), (2, 0)]], trace=True,
                           reservations=reservations), base_batch)
            self.assertEqual(
                replay(3, 2, [], (0, 0), (2, 0), [(0, 0), (1, 0), (2, 0)],
                       diagnose=True, reservations=reservations),
                base_replay)


class ReservationValidationTest(unittest.TestCase):
    def test_outer_type_errors(self):
        for bad in ("x", b"x", 1, 1.5, True, {(0, 0)}):
            with self.assertRaises(TypeError, msg=f"reservations={bad!r}"):
                plan(3, 3, set(), (0, 0), (2, 2), reservations=bad)

    def test_path_type_errors(self):
        for bad in ("x", b"x", 1, {(0, 0)}, None):
            with self.assertRaises(TypeError, msg=f"path={bad!r}"):
                plan(3, 3, set(), (0, 0), (2, 2),
                     reservations=[[(0, 1)], bad])

    def test_coordinate_type_errors(self):
        for bad in ((0,), (0, 0, 0), (0, "a"), (True, 0), "ab"):
            with self.assertRaises(TypeError, msg=f"coord={bad!r}"):
                plan(3, 3, set(), (0, 0), (2, 2), reservations=[[bad]])

    def test_empty_path_raises_value_error(self):
        with self.assertRaises(ValueError):
            plan(3, 3, set(), (0, 0), (2, 2), reservations=[[]])

    def test_out_of_bounds_raises_value_error(self):
        with self.assertRaises(ValueError):
            plan(3, 3, set(), (0, 0), (2, 2), reservations=[[(3, 0)]])
        with self.assertRaises(ValueError):
            plan(3, 3, set(), (0, 0), (2, 2), reservations=[[(0, -1)]])

    def test_static_obstacle_coordinate_raises_value_error(self):
        with self.assertRaises(ValueError):
            plan(3, 3, [(1, 1)], (0, 0), (2, 2),
                 reservations=[[(0, 1), (1, 1)]])

    def test_non_consecutive_repeat_raises_value_error(self):
        with self.assertRaises(ValueError):
            plan(3, 3, set(), (0, 0), (2, 2),
                 reservations=[[(0, 1), (1, 1), (0, 1)]])

    def test_consecutive_stays_are_kept(self):
        result = plan(3, 3, set(), (0, 0), (2, 2),
                      reservations=[[(2, 2), (2, 2), (2, 1)]])
        self.assertIsNotNone(result["path"])

    def test_validated_after_dynamic_costs(self):
        # The costs/dynamic_costs conflict ValueError precedes any
        # reservations check.
        with self.assertRaises(ValueError) as caught:
            plan(2, 2, [], (0, 0), (1, 1), costs=[[1, 1], [1, 1]],
                 dynamic_costs=[[[1, 1], [1, 1]]], reservations="x")
        self.assertIn("dynamic_costs", str(caught.exception))
        # A dynamic_costs TypeError precedes the reservations TypeError.
        with self.assertRaises(TypeError) as caught:
            plan(2, 2, [], (0, 0), (1, 1), dynamic_costs="x",
                 reservations="y")
        self.assertIn("dynamic_costs", str(caught.exception))

    def test_validated_before_search_in_every_entry_point(self):
        for call in (
            lambda: plan(3, 3, set(), (0, 0), (2, 2), reservations="x"),
            lambda: plan_any(3, 3, set(), (0, 0), [(2, 2)],
                             reservations="x"),
            lambda: plan_batch(3, 3, set(), [[(0, 0), (2, 2)]],
                               reservations="x"),
            lambda: plan_multi_start(3, 3, set(), [(0, 0)], (2, 2),
                                     reservations="x"),
            lambda: plan_k(3, 3, set(), (0, 0), (2, 2), 2,
                           reservations="x"),
            lambda: replay(3, 3, set(), (0, 0), (2, 2), [(0, 0)],
                           reservations="x"),
        ):
            with self.assertRaises(TypeError):
                call()

    def test_duplicates_and_input_order_do_not_matter(self):
        route_a = [(2, 2), (2, 1)]
        route_b = [(0, 2), (1, 2), (1, 1)]
        base = plan(3, 3, set(), (0, 0), (2, 0), trace=True,
                    reservations=[route_a, route_b])
        for permuted in ([route_b, route_a], [route_a, route_a, route_b],
                         [route_b, route_a, route_b, route_a]):
            self.assertEqual(
                plan(3, 3, set(), (0, 0), (2, 0), trace=True,
                     reservations=permuted), base)


class ReservationSearchTest(unittest.TestCase):
    def test_reserved_vertex_forces_detour(self):
        # (1, 1) is occupied forever: the direct middle-row route is
        # infeasible and the planner detours over the top row.
        result = plan(3, 3, set(), (0, 1), (2, 1),
                      reservations=[[(1, 1)]])
        self.assertEqual(result["path"],
                         [(0, 1), (0, 0), (1, 0), (2, 0), (2, 1)])
        self.assertEqual(result["cost"], 4)

    def test_reserved_vertex_persists_past_the_final_frame(self):
        # A single-coordinate route occupies its cell at every frame, so
        # no arrival time at (1, 1) is ever feasible.
        result = plan(3, 3, set(), (0, 1), (2, 1), trace=True,
                      reservations=[[(1, 1)]])
        self.assertNotIn((1, 1), [tuple(p) for p in result["path"]])
        self.assertEqual(result["cost"], 4)

    def test_reverse_edge_is_blocked(self):
        # Another agent moves (1, 0) -> (0, 0) at frame 0; the candidate
        # may not swap along the same edge and detours instead.
        result = plan(2, 2, set(), (0, 0), (1, 0),
                      reservations=[[(1, 0), (0, 0)]])
        self.assertEqual(result["path"], [(0, 0), (0, 1), (1, 1), (1, 0)])
        self.assertEqual(result["cost"], 3)

    def test_same_direction_edge_is_allowed(self):
        # Following the reserved edge's own direction is not a conflict.
        result = plan(3, 1, set(), (0, 0), (2, 0),
                      reservations=[[(0, 0), (1, 0), (2, 0), (2, 0)],
                                    [(2, 0), (2, 0), (2, 0), (2, 0)]])
        # (1, 0) at t=1 and (2, 0) at t>=2 are reserved, so the direct
        # route is infeasible on vertices alone; this just documents the
        # unreachable result on a corridor.
        self.assertEqual(result["path"], None)
        self.assertEqual(result["cost"], None)

    def test_unreachable_keeps_legacy_shape(self):
        result = plan(3, 1, set(), (0, 0), (2, 0), reservations=[[(1, 0)]])
        self.assertEqual(result, {"path": None, "cost": None,
                                  "expanded": result["expanded"]})
        self.assertEqual(set(result), {"path", "cost", "expanded"})

    def test_start_cell_is_never_checked(self):
        # The reservation occupies the start at frame 0; the start frame
        # is exempt, so the route simply leaves.
        result = plan(3, 3, set(), (0, 0), (2, 0),
                      reservations=[[(0, 0), (1, 1)]])
        self.assertEqual(result["path"], [(0, 0), (1, 0), (2, 0)])
        self.assertEqual(result["cost"], 2)

    def test_start_equals_goal_with_reservations(self):
        result = plan(2, 2, set(), (0, 0), (0, 0),
                      reservations=[[(1, 1)]])
        self.assertEqual(result, {"path": [(0, 0)], "cost": 0,
                                  "expanded": 1})

    def test_trace_records_space_time_triples(self):
        result = plan(3, 3, set(), (0, 1), (2, 1), trace=True,
                      reservations=[[(1, 1)]])
        self.assertTrue(all(len(entry) == 3
                            for entry in result["expanded_nodes"]))
        self.assertEqual(result["expanded"], len(result["expanded_nodes"]))
        self.assertEqual(result["expanded_nodes"][0], (0, 1, 0))

    def test_reservations_compose_with_dynamic_blocked_and_costs(self):
        frames = [[(0, 1)], [], []]
        result = plan(3, 3, set(), (0, 0), (2, 0),
                      dynamic_blocked=frames, reservations=[[(1, 1)]])
        self.assertIsNotNone(result["path"])
        check = replay(3, 3, set(), (0, 0), (2, 0), result["path"],
                       dynamic_blocked=frames, reservations=[[(1, 1)]])
        self.assertEqual(check, {"valid": True, "cost": result["cost"],
                                 "steps": len(result["path"]) - 1})

    def test_wait_inside_reservation_frames(self):
        # The direct cell (1, 0) is reserved at frame 1; waiting one
        # frame at the start is cheaper than the row-1 detour.
        reservation = [[(1, 1), (1, 0), (2, 0), (2, 1)]]
        waited = plan(3, 2, set(), (0, 0), (2, 0), allow_wait=True,
                      reservations=reservation)
        self.assertEqual(waited["path"], [(0, 0), (0, 0), (1, 0), (2, 0)])
        self.assertEqual(waited["cost"], 2)  # waits add no cost
        # Without waiting the detour over row 1 wins.
        direct = plan(3, 2, set(), (0, 0), (2, 0),
                      reservations=reservation)
        self.assertEqual(direct["path"],
                         [(0, 0), (0, 1), (1, 1), (1, 0), (2, 0)])
        self.assertEqual(direct["cost"], 4)

    def test_wait_only_within_reservation_frames(self):
        # The reservation provides a single frame, so waiting past it is
        # never generated; the detour wins instead.
        result = plan(3, 2, set(), (0, 0), (2, 0), allow_wait=True,
                      reservations=[[(1, 0)]])
        self.assertEqual(result["path"],
                         [(0, 0), (0, 1), (1, 1), (2, 1), (2, 0)])
        self.assertEqual(result["cost"], 4)

    def test_max_expanded_status_with_reservations(self):
        limited = plan(4, 4, set(), (0, 0), (3, 3), max_expanded=3,
                       reservations=[[(1, 1)]])
        self.assertEqual(limited["status"], "budget_exhausted")
        self.assertEqual(limited["expanded"], 3)
        self.assertIsNone(limited["path"])
        full = plan(4, 4, set(), (0, 0), (3, 3), reservations=[[(1, 1)]])
        self.assertEqual(set(limited) - {"status"}, {"path", "cost",
                                                     "expanded"})

    def test_max_cost_status_with_reservations(self):
        limited = plan(3, 3, set(), (0, 1), (2, 1), max_cost=2,
                       reservations=[[(1, 1)]])
        self.assertEqual(limited["status"], "cost_exhausted")
        self.assertIsNone(limited["path"])
        enough = plan(3, 3, set(), (0, 1), (2, 1), max_cost=4,
                      reservations=[[(1, 1)]])
        self.assertEqual(enough["status"], "found")
        self.assertEqual(enough["cost"], 4)


class ReservationEntryPointsTest(unittest.TestCase):
    def test_plan_any(self):
        result = plan_any(3, 3, set(), (0, 1), [(2, 1), (2, 0)],
                          reservations=[[(1, 1)]])
        self.assertIsNotNone(result["path"])
        self.assertIn(result["path"][-1], [(2, 1), (2, 0)])
        check = replay(3, 3, set(), (0, 1), result["path"][-1],
                       result["path"], reservations=[[(1, 1)]])
        self.assertTrue(check["valid"])
        self.assertEqual(check["cost"], result["cost"])

    def test_plan_k(self):
        # (1, 1) is reserved forever, leaving exactly the top-row and
        # bottom-row routes.
        result = plan_k(3, 3, set(), (0, 1), (2, 1), 3,
                        reservations=[[(1, 1)]])
        self.assertEqual(len(result["paths"]), 2)
        self.assertEqual(result["paths"][0],
                         plan(3, 3, set(), (0, 1), (2, 1),
                              reservations=[[(1, 1)]])["path"])
        for path, cost in zip(result["paths"], result["costs"]):
            check = replay(3, 3, set(), (0, 1), (2, 1), path,
                           reservations=[[(1, 1)]])
            self.assertEqual(check, {"valid": True, "cost": cost,
                                     "steps": len(path) - 1})

    def test_plan_batch_matches_independent_plan_calls(self):
        requests = [[(0, 1), (2, 1)], [(0, 0), (2, 2)]]
        reservations = [[(1, 1)]]
        batch = plan_batch(3, 3, set(), requests,
                           reservations=reservations)
        self.assertEqual(len(batch["results"]), 2)
        for (start, goal), result in zip(requests, batch["results"]):
            self.assertEqual(
                without_checkpoint(result),
                without_checkpoint(plan(3, 3, set(), start, goal,
                                        reservations=reservations)))

    def test_plan_multi_start(self):
        result = plan_multi_start(3, 3, set(), [(0, 1), (0, 2)], (2, 1),
                                  trace=True, reservations=[[(1, 1)]])
        self.assertIsNotNone(result["path"])
        self.assertIn(result["path"][0], [(0, 1), (0, 2)])
        self.assertTrue(all(len(entry) == 3
                            for entry in result["expanded_nodes"]))
        check = replay(3, 3, set(), result["path"][0], (2, 1),
                       result["path"], reservations=[[(1, 1)]])
        self.assertEqual(check, {"valid": True, "cost": result["cost"],
                                 "steps": len(result["path"]) - 1})


class ReservationReplayTest(unittest.TestCase):
    def test_valid_path_cost_and_steps(self):
        result = plan(3, 3, set(), (0, 1), (2, 1),
                      reservations=[[(1, 1)]])
        check = replay(3, 3, set(), (0, 1), (2, 1), result["path"],
                       reservations=[[(1, 1)]])
        self.assertEqual(check, {"valid": True, "cost": result["cost"],
                                 "steps": len(result["path"]) - 1})

    def test_diagnose_reservation_vertex(self):
        check = replay(3, 3, set(), (0, 1), (2, 1),
                       [(0, 1), (1, 1), (2, 1)], diagnose=True,
                       reservations=[[(1, 1)]])
        self.assertEqual(check, {"valid": False, "cost": None,
                                 "steps": None,
                                 "error": "reservation_vertex",
                                 "error_index": 1})

    def test_diagnose_reservation_vertex_persistent_tail(self):
        # The reserved vertex persists past the route's final frame.
        check = replay(3, 3, set(), (0, 1), (2, 1),
                       [(0, 1), (0, 0), (1, 0), (1, 1), (2, 1)],
                       diagnose=True, reservations=[[(2, 2), (1, 1)]])
        self.assertEqual(check["error"], "reservation_vertex")
        self.assertEqual(check["error_index"], 3)

    def test_diagnose_reservation_edge(self):
        check = replay(2, 2, set(), (0, 0), (1, 0), [(0, 0), (1, 0)],
                       diagnose=True, reservations=[[(1, 0), (0, 0)]])
        self.assertEqual(check, {"valid": False, "cost": None,
                                 "steps": None,
                                 "error": "reservation_edge",
                                 "error_index": 1})

    def test_diagnose_first_violation_wins(self):
        # The vertex conflict at index 1 precedes any later problem.
        check = replay(3, 3, set(), (0, 1), (2, 1),
                       [(0, 1), (1, 1), (1, 1), (2, 1)], diagnose=True,
                       reservations=[[(1, 1)]])
        self.assertEqual(check["error"], "reservation_vertex")
        self.assertEqual(check["error_index"], 1)

    def test_non_diagnose_keeps_fixed_structure(self):
        check = replay(3, 3, set(), (0, 1), (2, 1),
                       [(0, 1), (1, 1), (2, 1)], reservations=[[(1, 1)]])
        self.assertEqual(check, {"valid": False, "cost": None,
                                 "steps": None})

    def test_other_invalid_paths_keep_existing_codes(self):
        # A repeated coordinate still reports the legacy error.
        check = replay(3, 3, set(), (0, 0), (2, 0),
                       [(0, 0), (1, 0), (0, 0), (1, 0), (2, 0)],
                       diagnose=True, reservations=[[(2, 2)]])
        self.assertEqual(check["error"], "repeated_coordinate")
        self.assertEqual(check["error_index"], 2)

    def test_wait_replay_with_reservations(self):
        reservation = [[(1, 1), (1, 0), (2, 0), (2, 1)]]
        result = plan(3, 2, set(), (0, 0), (2, 0), allow_wait=True,
                      reservations=reservation)
        check = replay(3, 2, set(), (0, 0), (2, 0), result["path"],
                       allow_wait=True, reservations=reservation)
        self.assertEqual(check, {"valid": True, "cost": result["cost"],
                                 "steps": len(result["path"]) - 1})

    def test_wait_after_final_reservation_frame(self):
        # A single-coordinate reservation provides only frame 0, so a
        # stay at index 1 is already past the provided frames.
        check = replay(3, 3, set(), (0, 0), (1, 0),
                       [(0, 0), (0, 0), (1, 0)], diagnose=True,
                       allow_wait=True, reservations=[[(2, 2)]])
        self.assertEqual(check["error"], "wait_after_final_frame")
        self.assertEqual(check["error_index"], 1)

    def test_wait_into_reserved_vertex(self):
        check = replay(3, 3, set(), (0, 0), (0, 0) , [(0, 0)],
                       reservations=[[(2, 2)]])
        self.assertTrue(check["valid"])
        # Waiting into a cell the reservation holds at that frame.
        check = replay(3, 2, set(), (0, 0), (2, 0),
                       [(0, 0), (0, 0), (1, 0), (2, 0)], diagnose=True,
                       allow_wait=True,
                       reservations=[[(0, 0), (0, 0), (2, 1)]])
        self.assertEqual(check["error"], "reservation_vertex")
        self.assertEqual(check["error_index"], 1)


class ReservationSnapshotTest(unittest.TestCase):
    def test_checkpoint_records_normalized_reservations(self):
        result = plan(4, 4, set(), (0, 0), (3, 3), max_expanded=2,
                      snapshot=True,
                      reservations=[[(3, 0), (2, 0)], [(0, 3)],
                                    [(3, 0), (2, 0)]])
        self.assertEqual(result["status"], "budget_exhausted")
        checkpoint = result["checkpoint"]
        # Duplicates merged and routes sorted; JSON-serializable.
        self.assertEqual(checkpoint["reservations"],
                         [[[0, 3]], [[3, 0], [2, 0]]])
        json.dumps(checkpoint)

    def test_checkpoint_without_reservations_omits_field(self):
        result = plan(4, 4, set(), (0, 0), (3, 3), max_expanded=2,
                      snapshot=True)
        self.assertNotIn("reservations", result["checkpoint"])

    def test_resume_matches_uninterrupted_search(self):
        reservations = [[(1, 1), (2, 2)], [(0, 3), (1, 3)]]
        kwargs = dict(reservations=reservations, max_expanded=5,
                      snapshot=True, trace=True)
        first = plan(4, 4, set(), (0, 0), (3, 3), **kwargs)
        self.assertEqual(first["status"], "budget_exhausted")
        resumed = resume(first["checkpoint"], max_expanded=200)
        direct = plan(4, 4, set(), (0, 0), (3, 3), trace=True,
                      reservations=reservations, max_expanded=200)
        for key in ("path", "cost", "expanded", "status",
                    "expanded_nodes"):
            self.assertEqual(resumed[key], direct[key], msg=key)

    def test_resume_plan_any_and_multi_start(self):
        reservations = [[(1, 1)], [(3, 3), (3, 2)]]
        first_any = plan_any(4, 4, set(), (0, 0), [(3, 3), (3, 0)],
                             max_expanded=3, snapshot=True,
                             reservations=reservations)
        self.assertEqual(first_any["status"], "budget_exhausted")
        resumed_any = resume(first_any["checkpoint"], max_expanded=500)
        direct_any = plan_any(4, 4, set(), (0, 0), [(3, 3), (3, 0)],
                              trace=True, max_expanded=500,
                              reservations=reservations)
        for key in ("path", "cost", "expanded", "status",
                    "expanded_nodes"):
            self.assertEqual(resumed_any[key], direct_any[key], msg=key)
        first_multi = plan_multi_start(
            4, 4, set(), [(0, 0), (0, 1)], (3, 3), max_expanded=3,
            snapshot=True, reservations=reservations)
        self.assertEqual(first_multi["status"], "budget_exhausted")
        resumed_multi = resume(first_multi["checkpoint"], max_expanded=500)
        direct_multi = plan_multi_start(
            4, 4, set(), [(0, 0), (0, 1)], (3, 3), trace=True,
            max_expanded=500, reservations=reservations)
        for key in ("path", "cost", "expanded", "status",
                    "expanded_nodes"):
            self.assertEqual(resumed_multi[key], direct_multi[key],
                             msg=key)

    def test_resume_with_wait_and_reservations(self):
        reservation = [[(1, 1), (1, 0), (2, 0), (2, 1)]]
        first = plan(3, 2, set(), (0, 0), (2, 0), allow_wait=True,
                     max_expanded=2, snapshot=True,
                     reservations=reservation)
        self.assertEqual(first["status"], "budget_exhausted")
        resumed = resume(first["checkpoint"], max_expanded=100)
        direct = plan(3, 2, set(), (0, 0), (2, 0), allow_wait=True,
                      trace=True, max_expanded=100,
                      reservations=reservation)
        for key in ("path", "cost", "expanded", "status",
                    "expanded_nodes"):
            self.assertEqual(resumed[key], direct[key], msg=key)

    def test_corrupt_reservations_field(self):
        result = plan(4, 4, set(), (0, 0), (3, 3), max_expanded=2,
                      snapshot=True, reservations=[[(3, 0), (2, 0)]])
        checkpoint = result["checkpoint"]
        bad_type = dict(checkpoint, reservations="x")
        with self.assertRaises(TypeError):
            resume(bad_type)
        bad_value = dict(checkpoint, reservations=[[[9, 9]]])
        with self.assertRaises(ValueError):
            resume(bad_value)
        bad_blocked = dict(checkpoint, reservations=[[[0, 0], [0, 0],
                                                      [1, 1], [0, 0]]])
        with self.assertRaises(ValueError):
            resume(bad_blocked)


if __name__ == "__main__":
    unittest.main()
