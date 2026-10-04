import copy
import json
import unittest

from app import plan, plan_any, resume, replay

HISTORY_FRAMES = [
    [(0, 1), (0, 2)],
    [(0, 1), (1, 2), (2, 0), (2, 2)],
    [(2, 1), (2, 2)],
    [(0, 0), (1, 2), (2, 1), (2, 2)],
    [],
    [(0, 1)],
]
WALL = {(1, y) for y in range(3)}
BLOCKED = {(2, 0), (2, 1), (2, 2), (0, 2), (4, 1)}


def _kwargs(kind, **over):
    kw = dict(width=6, height=4, blocked={(2, 0), (2, 1), (2, 2)},
              start=(0, 0), costs=None, dynamic_blocked=None, trace=False)
    if kind == "plan":
        kw["goal"] = (5, 3)
    else:
        kw["goals"] = [(5, 3), (5, 2)]
    kw.update(over)
    return kw


def _call(kind, **kw):
    return plan(**kw) if kind == "plan" else plan_any(**kw)


def _canon(cp):
    return json.dumps(cp, sort_keys=True)


class SnapshotShapeTest(unittest.TestCase):
    def test_default_omits_checkpoint(self):
        for kwargs in ({}, {"snapshot": False}):
            r = plan(6, 4, {(2, 0), (2, 1)}, (0, 0), (5, 3), **kwargs)
            self.assertEqual(set(r), {"path", "cost", "expanded"})
        r = plan(6, 4, {(2, 0), (2, 1)}, (0, 0), (5, 3),
                 max_expanded=2, snapshot=False)
        self.assertEqual(set(r), {"path", "cost", "expanded", "status"})

    def test_checkpoint_present_only_on_budget_exhausted(self):
        r = plan(6, 4, {(2, 0), (2, 1)}, (0, 0), (5, 3),
                 max_expanded=2, snapshot=True)
        self.assertEqual(set(r),
                         {"path", "cost", "expanded", "status", "checkpoint"})
        self.assertIsNone(r["path"])
        # found carries no checkpoint
        found = plan(6, 4, {(2, 0), (2, 1)}, (0, 0), (5, 3),
                     max_expanded=1000, snapshot=True)
        self.assertEqual(found["status"], "found")
        self.assertNotIn("checkpoint", found)
        # unreachable carries no checkpoint
        gone = plan(3, 3, WALL, (0, 0), (2, 2),
                    max_expanded=10, snapshot=True)
        self.assertEqual(gone["status"], "unreachable")
        self.assertNotIn("checkpoint", gone)
        # snapshot without a budget never emits one either
        nob = plan(6, 4, {(2, 0), (2, 1)}, (0, 0), (5, 3), snapshot=True)
        self.assertNotIn("checkpoint", nob)

    def test_checkpoint_is_json_only_and_round_trips(self):
        r = plan(6, 4, BLOCKED, (0, 0), (5, 3), trace=True,
                 max_expanded=4, snapshot=True)
        cp = r["checkpoint"]

        def walk(value):
            self.assertNotIsInstance(value, (tuple, set, frozenset))
            if isinstance(value, dict):
                for key, item in value.items():
                    self.assertIsInstance(key, str)
                    walk(item)
            elif isinstance(value, list):
                for item in value:
                    walk(item)
            else:
                self.assertTrue(
                    value is None or isinstance(value, (str, int, bool))
                )

        walk(cp)
        restored = json.loads(json.dumps(cp))
        self.assertEqual(resume(restored, max_expanded=1000)["path"],
                         plan(6, 4, BLOCKED, (0, 0), (5, 3))["path"])

    def test_plan_any_checkpoint_shape(self):
        r = plan_any(6, 4, {(2, 0)}, (0, 0), [(5, 3), (5, 2)],
                     max_expanded=3, snapshot=True)
        cp = r["checkpoint"]
        self.assertIn("goals", cp)
        self.assertNotIn("goal", cp)
        # Static any-search is a route-tree traversal: no static-only keys.
        self.assertNotIn("closed", cp)
        self.assertEqual(cp["kind"], "plan_any")
        self.assertFalse(cp["dynamic"])
        self.assertEqual(cp["goals"], [[5, 2], [5, 3]])  # sorted
        self.assertEqual(cp["blocked"], [[2, 0]])

    def test_static_plan_checkpoint_shape(self):
        r = plan(6, 4, BLOCKED, (0, 0), (5, 3), max_expanded=4, snapshot=True)
        cp = r["checkpoint"]
        self.assertEqual(cp["kind"], "plan")
        self.assertFalse(cp["dynamic"])
        for key in ("version", "kind", "dynamic", "trace", "width", "height",
                    "blocked", "start", "goal", "costs", "frames",
                    "expanded", "expanded_nodes", "open", "closed",
                    "g_score", "came_from"):
            self.assertIn(key, cp, key)
        self.assertIsNone(cp["costs"])
        self.assertIsNone(cp["frames"])
        self.assertIsNone(cp["expanded_nodes"])  # trace was false

    def test_dynamic_checkpoint_shape(self):
        r = plan(3, 3, set(), (1, 0), (2, 2), trace=True,
                 dynamic_blocked=HISTORY_FRAMES, max_expanded=3,
                 snapshot=True)
        cp = r["checkpoint"]
        self.assertTrue(cp["dynamic"])
        self.assertEqual(len(cp["frames"]), len(HISTORY_FRAMES))
        # Eight-field route-tree entries.
        self.assertEqual(len(cp["open"][0]), 8)
        self.assertNotIn("closed", cp)
        self.assertEqual(cp["expanded_nodes"][0], [1, 0, 0])


class SnapshotValidationTest(unittest.TestCase):
    def test_snapshot_must_be_bool(self):
        for bad in (1, 0, "true", None, [], 1.0):
            with self.assertRaises(TypeError, msg=f"snapshot={bad!r}"):
                plan(3, 3, set(), (0, 0), (2, 2), snapshot=bad)
            with self.assertRaises(TypeError, msg=f"snapshot={bad!r}"):
                plan_any(3, 3, set(), (0, 0), [(2, 2)], snapshot=bad)

    def test_snapshot_validation_position(self):
        # Pre-existing grid ValueErrors come first.
        with self.assertRaises(ValueError):
            plan(0, 3, set(), (0, 0), (2, 2), snapshot=1)
        with self.assertRaises(ValueError):
            plan(3, 3, {(0, 0)}, (0, 0), (2, 2), snapshot=1)
        # trace TypeError precedes the snapshot TypeError.
        with self.assertRaises(TypeError):
            plan(3, 3, set(), (0, 0), (2, 2), trace=1, snapshot=True)
        # frame-0 ValueError precedes the snapshot TypeError.
        with self.assertRaises(ValueError):
            plan(3, 3, set(), (0, 0), (2, 2),
                 dynamic_blocked=[[(0, 0)]], snapshot=True)
        # snapshot TypeError precedes the budget TypeError.
        with self.assertRaises(TypeError):
            plan(3, 3, set(), (0, 0), (2, 2),
                 snapshot="x", max_expanded="y")

    def test_replay_rejects_snapshot(self):
        with self.assertRaises(TypeError):
            replay(3, 3, set(), (0, 0), (2, 2),
                   [(0, 0), (1, 0)], snapshot=True)


def _checkpoint(kind, **over):
    kw = _kwargs(kind)
    kw.update(trace=True, max_expanded=2, snapshot=True)
    kw.update(over)
    return _call(kind, **kw)["checkpoint"]


class ResumeValidationTest(unittest.TestCase):
    def test_checkpoint_must_be_object(self):
        for bad in (None, 1, "x", [], (), True, 1.5):
            with self.assertRaises(TypeError, msg=repr(bad)):
                resume(bad)

    def test_missing_fields_are_type_errors(self):
        cp = _checkpoint("plan")
        for key in cp:
            mut = copy.deepcopy(cp)
            del mut[key]
            with self.assertRaises(TypeError, msg=key):
                resume(mut)

    def test_field_type_errors(self):
        cp = _checkpoint("plan")
        swaps = [
            ("version", "1"), ("kind", 5), ("dynamic", 1), ("trace", 1),
            ("width", "3"), ("blocked", {}), ("start", (0, 0)),
            ("goal", [0]), ("costs", {}), ("frames", {}),
            ("expanded", "2"), ("expanded_nodes", {}), ("open", {}),
            ("closed", {}), ("g_score", {}), ("came_from", {}),
        ]
        for key, value in swaps:
            if key not in cp:
                continue
            mut = copy.deepcopy(cp)
            mut[key] = value
            with self.assertRaises(TypeError, msg=key):
                resume(mut)

    def test_inconsistency_value_errors(self):
        cp = _checkpoint("plan")
        swaps = [
            ("version", 99), ("kind", "plan_other"), ("width", 0),
            ("height", 0), ("dynamic", True),
            ("start", [5, 5]), ("goal", [5, 5]),
        ]
        for key, value in swaps:
            mut = copy.deepcopy(cp)
            mut[key] = value
            with self.assertRaises(ValueError, msg=key):
                resume(mut)
        # Static snapshot promising frames.
        mut = copy.deepcopy(cp)
        mut["dynamic"] = True
        with self.assertRaises(ValueError):
            resume(mut)
        # Added obstacle.
        mut = copy.deepcopy(cp)
        mut["blocked"] = sorted(cp["blocked"] + [[1, 0]])
        with self.assertRaises(ValueError):
            resume(mut)
        # Tampered g / f in a pending entry.
        mut = copy.deepcopy(cp)
        mut["g_score"][0][1] += 5
        with self.assertRaises(ValueError):
            resume(mut)
        mut = copy.deepcopy(cp)
        mut["open"][0][0] += 1
        with self.assertRaises(ValueError):
            resume(mut)
        # Trace/count disagreement.
        mut = copy.deepcopy(cp)
        mut["trace"] = True
        mut["expanded_nodes"] = []
        with self.assertRaises(ValueError):
            resume(mut)
        # Empty pending frontier is inconsistent with budget_exhausted.
        mut = copy.deepcopy(cp)
        mut["open"] = []
        with self.assertRaises(ValueError):
            resume(mut)
        # Unknown field.
        mut = copy.deepcopy(cp)
        mut["extra"] = 1
        with self.assertRaises(ValueError):
            resume(mut)

    def test_dynamic_inconsistency_value_errors(self):
        cp = _checkpoint("plan", width=3, height=3, blocked=set(),
                         start=(1, 0), goal=(2, 2),
                         dynamic_blocked=HISTORY_FRAMES, max_expanded=3)
        mut = copy.deepcopy(cp)
        mut["frames"][0].append([1, 0])  # start now blocked at frame 0
        with self.assertRaises(ValueError):
            resume(mut)
        mut = copy.deepcopy(cp)
        mut["dynamic"] = True
        mut["frames"] = None
        with self.assertRaises(ValueError):
            resume(mut)
        mut = copy.deepcopy(cp)
        mut["open"][0][6].append([2, 0])  # path no longer matches t
        with self.assertRaises(ValueError):
            resume(mut)

    def test_kind_field_mismatch(self):
        cp = _checkpoint("any")
        mut = copy.deepcopy(cp)
        mut["kind"] = "plan"
        # 'goal' is now required and absent -> TypeError.
        with self.assertRaises(TypeError):
            resume(mut)

    def test_resume_budget_rules(self):
        cp = _checkpoint("plan")
        for bad in ("5", 1.5, True, [5]):
            with self.assertRaises(TypeError, msg=repr(bad)):
                resume(cp, max_expanded=bad)
        with self.assertRaises(ValueError):
            resume(cp, max_expanded=-1)


class ResumeEquivalenceTest(unittest.TestCase):
    SCENARIOS = [
        # kind, kwargs, target-overrides
        ("plan", dict(width=3, height=3, blocked=WALL, start=(0, 0),
                      goal=(2, 2)), None),
        ("plan", dict(width=6, height=4, blocked=BLOCKED, start=(0, 0),
                      goal=(5, 3)), None),
        ("plan", dict(width=3, height=3, blocked=set(), start=(0, 0),
                      goal=(2, 0),
                      costs=[[1, 10, 1], [1, 10, 1], [1, 1, 1]]), None),
        ("plan", dict(width=3, height=3, blocked=set(), start=(1, 0),
                      goal=(2, 2),
                      dynamic_blocked=HISTORY_FRAMES), None),
        ("plan", dict(width=2, height=2, blocked=set(), start=(0, 0),
                      goal=(1, 1),
                      dynamic_blocked=[[], [(1, 0), (0, 1)]]), None),
        ("any", dict(width=6, height=4,
                     blocked={(2, 0), (2, 1), (2, 2)}, start=(0, 0),
                     goals=[(5, 3), (5, 2)]), None),
        ("any", dict(width=3, height=3, blocked=WALL, start=(0, 0),
                     goals=[(2, 0), (2, 2)]), None),
        ("any", dict(width=3, height=3, blocked=set(), start=(1, 0),
                     goals=[(2, 2), (0, 2)],
                     dynamic_blocked=HISTORY_FRAMES), None),
    ]

    def _run(self, kind, base_kw, trace, max_expanded=None, cp=None,
             snapshot=False):
        kw = dict(base_kw)
        kw["trace"] = trace
        if cp is None:
            kw["max_expanded"] = max_expanded
            if snapshot:
                kw["snapshot"] = True
            return _call(kind, **kw)
        return resume(cp, max_expanded=max_expanded)

    def test_resume_equals_single_call_at_every_budget(self):
        for kind, base_kw, _ in self.SCENARIOS:
            for trace in (False, True):
                truth = self._run(kind, base_kw, trace)
                total = truth["expanded"]
                for budget in range(0, total + 2):
                    single = self._run(kind, base_kw, trace,
                                       max_expanded=budget)
                    # A zero budget always pauses (nothing closes); resume
                    # straight to ``budget``. It must not matter that some
                    # smaller budget in between could already close a goal
                    # (a terminal ``found`` carries no checkpoint) -- the
                    # resumed run reproduces exactly plan(max=budget),
                    # including expansions after an early goal close.
                    paused = self._run(kind, base_kw, trace,
                                       max_expanded=0, snapshot=True)
                    self.assertEqual(paused["status"], "budget_exhausted")
                    paused = resume(
                        json.loads(json.dumps(paused["checkpoint"])),
                        max_expanded=budget)
                    for key in ("path", "cost", "expanded", "status"):
                        self.assertEqual(paused.get(key), single.get(key),
                                         (kind, budget, key))
                    if trace:
                        self.assertEqual(paused["expanded_nodes"],
                                         single["expanded_nodes"])
                    if single["status"] == "budget_exhausted":
                        self.assertIn("checkpoint", paused)
                    else:
                        self.assertNotIn("checkpoint", paused)

    def test_resume_to_completion_matches_unbudgeted_truth(self):
        for kind, base_kw, _ in self.SCENARIOS:
            for trace in (False, True):
                truth = self._run(kind, base_kw, trace)
                total = truth["expanded"]
                for pause in range(0, max(1, total)):
                    paused = self._run(kind, base_kw, trace,
                                       max_expanded=pause, snapshot=True)
                    if paused["status"] != "budget_exhausted":
                        continue
                    cp = json.loads(json.dumps(paused["checkpoint"]))
                    done = resume(cp)  # no budget: run to completion
                    self.assertEqual(done["path"], truth["path"])
                    self.assertEqual(done["cost"], truth["cost"])
                    self.assertEqual(done["expanded"], truth["expanded"])
                    self.assertNotIn("status", done)
                    self.assertNotIn("checkpoint", done)
                    if trace:
                        self.assertEqual(done["expanded_nodes"],
                                         truth["expanded_nodes"])
                    if done["path"] is not None:
                        goal = (base_kw.get("goal")
                                if kind == "plan" else done["path"][-1])
                        checked = replay(
                            base_kw["width"], base_kw["height"],
                            base_kw["blocked"], base_kw["start"], goal,
                            done["path"], costs=base_kw.get("costs"),
                            dynamic_blocked=base_kw.get("dynamic_blocked"))
                        self.assertTrue(checked["valid"])
                        self.assertEqual(checked["cost"], done["cost"])
                        self.assertEqual(checked["steps"],
                                         len(done["path"]) - 1)

    def test_repeated_resume_does_not_double_count(self):
        cp = _checkpoint("plan", trace=True)
        n = cp["expanded"]
        self.assertGreaterEqual(n, 1)
        again = resume(cp, max_expanded=n)
        self.assertEqual(again["status"], "budget_exhausted")
        self.assertEqual(again["checkpoint"]["expanded"], n)
        third = resume(json.loads(json.dumps(again["checkpoint"])),
                       max_expanded=n)
        self.assertEqual(third["expanded"], n)

    def test_zero_budget_start_equals_goal_resume(self):
        r = plan(3, 3, set(), (1, 1), (1, 1), trace=True,
                 max_expanded=0, snapshot=True)
        self.assertEqual(r["status"], "budget_exhausted")
        cp = json.loads(json.dumps(r["checkpoint"]))
        found = resume(cp, max_expanded=1)
        self.assertEqual(found, {"path": [(1, 1)], "cost": 0,
                                 "expanded": 1, "status": "found",
                                 "expanded_nodes": [(1, 1)]})
        # No checkpoint on the found result.
        self.assertNotIn("checkpoint", found)
        # Dynamic plan_any analogue records a triple.
        r = plan_any(3, 3, set(), (1, 1), [(1, 1), (2, 2)], trace=True,
                     dynamic_blocked=[[], [(0, 0)]],
                     max_expanded=0, snapshot=True)
        found = resume(json.loads(json.dumps(r["checkpoint"])),
                       max_expanded=1)
        self.assertEqual(found["path"], [(1, 1)])
        self.assertEqual(found["expanded_nodes"], [(1, 1, 0)])

    def test_chunked_resume_checkpoint_is_byte_stable(self):
        kw = dict(width=3, height=3, blocked=set(), start=(1, 0),
                  goal=(2, 2), dynamic_blocked=HISTORY_FRAMES, trace=True)
        total = plan(**kw)["expanded"]
        for pause in range(1, total):
            single = plan(**kw, max_expanded=pause, snapshot=True)
            if single["status"] != "budget_exhausted":
                continue
            r = plan(**kw, max_expanded=1, snapshot=True)
            cp = r["checkpoint"]
            mid = pause // 2
            if mid >= 1:
                r = resume(cp, max_expanded=mid)
                if r["status"] == "budget_exhausted":
                    cp = r["checkpoint"]
            r = resume(cp, max_expanded=pause)
            if r["status"] == "budget_exhausted":
                self.assertEqual(_canon(r["checkpoint"]),
                                 _canon(single["checkpoint"]))

    def test_checkpoint_order_independence(self):
        cells = [(x, 1) for x in range(5) if x != 2]
        orders = (cells, list(reversed(cells)),
                  [cells[2], cells[0], cells[3], cells[1]])
        cps = {
            _canon(plan(5, 3, order, (0, 0), (4, 2), max_expanded=4,
                        snapshot=True)["checkpoint"])
            for order in orders
        }
        self.assertEqual(len(cps), 1)
        cps = {
            _canon(plan_any(5, 3, cells, (0, 0), gl, max_expanded=4,
                            snapshot=True)["checkpoint"])
            for gl in ([(4, 2), (4, 0)], [(4, 0), (4, 2)])
        }
        self.assertEqual(len(cps), 1)
        cps = {
            _canon(plan(3, 3, set(), (1, 0), (2, 2),
                        dynamic_blocked=frames, max_expanded=4,
                        snapshot=True)["checkpoint"])
            for frames in (HISTORY_FRAMES,
                           [list(reversed(f)) for f in HISTORY_FRAMES],
                           [set(f) for f in HISTORY_FRAMES])
        }
        self.assertEqual(len(cps), 1)

    def test_resume_trace_preference_comes_from_checkpoint(self):
        cp = plan(3, 3, set(), (0, 0), (2, 2), max_expanded=1,
                  snapshot=True)["checkpoint"]
        self.assertNotIn("expanded_nodes", resume(cp))
        cp = plan(3, 3, set(), (0, 0), (2, 2), trace=True, max_expanded=1,
                  snapshot=True)["checkpoint"]
        self.assertIn("expanded_nodes", resume(cp))

    def test_checkpoint_object_not_mutated_by_resume(self):
        cp = plan(4, 4, {(1, 1)}, (0, 0), (3, 3), trace=True,
                  max_expanded=3, snapshot=True)["checkpoint"]
        before = copy.deepcopy(cp)
        resume(cp, max_expanded=100)
        self.assertEqual(cp, before)


if __name__ == "__main__":
    unittest.main()
