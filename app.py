"""Deterministic 2D grid A* planner.

Public entry points: ``plan(width, height, blocked, start, goal, costs=None,
trace=False, dynamic_blocked=None, max_expanded=None, snapshot=False,
max_cost=None)``, ``plan_any(width, height, blocked, start, goals,
costs=None, trace=False, dynamic_blocked=None, max_expanded=None,
snapshot=False, max_cost=None)``, ``replay(width, height, blocked, start,
goal, path, costs=None, dynamic_blocked=None, diagnose=False)`` and
``resume(checkpoint, max_expanded=None)``.

Semantics:
- Four-neighborhood moves, Manhattan heuristic. Without ``costs`` each step
  has unit cost. With ``costs`` (a height-by-width matrix of positive
  integers, row-major: ``costs[y][x]``) entering a cell costs that cell's
  value; the start cell's cost is never counted, so ``cost`` is the sum of
  the costs of the cells entered along the path.
- Tie-breaking is fully specified and order-independent: compare f, then h,
  then the (x, y) coordinate in lexicographic order.
- ``expanded`` counts exactly the nodes popped from the priority queue and
  closed for the first time (the goal counts when reached; stale heap
  entries and duplicate closings do not).
- Returns ``{"path": [...], "cost": int, "expanded": int}`` on success and
  ``{"path": None, "cost": None, "expanded": int}`` when unreachable.
- When ``trace`` is true, both kinds of results additionally include
  ``expanded_nodes``: the list of coordinate tuples closed for the first
  time, in expansion order (includes the start, and the goal on success);
  ``expanded == len(expanded_nodes)``. The field is omitted entirely when
  ``trace`` is false or omitted.

Search budget (``max_expanded``):
- ``max_expanded`` may be omitted or ``None``, which keeps every key,
  value, exception type and validation order of the default behavior
  untouched. Otherwise it must be a non-negative integer that is not a
  bool: other types raise ``TypeError`` and negative values raise
  ``ValueError``. These checks run after all existing grid, costs,
  ``trace`` and ``dynamic_blocked`` validation (including the frame-0
  check) and before the search starts.
- The budget counts closed nodes (exactly what ``expanded`` counts); once
  the limit is reached no further node is closed. When the budget is
  provided the result always carries an additional ``status`` field:
  ``"found"`` when the goal was closed within the limit (the usual
  ``path``/``cost`` are returned), ``"unreachable"`` when the candidates
  were exhausted without reaching the goal, and ``"budget_exhausted"``
  when live candidates remained but the budget stopped the search (then
  ``path``/``cost`` are ``None`` and ``expanded`` is the actual closed
  count, at most ``max_expanded``).
- The budget applies identically in static and dynamic mode and never
  changes expansion order, path choice, trace entries or determinism: a
  budgeted trace is a prefix of the unbudgeted one. With ``trace=True``
  only actually closed nodes are recorded, so a zero budget records
  nothing -- even when ``start == goal`` a zero budget returns
  ``budget_exhausted`` with ``expanded == 0``; the single-point success
  result requires at least one closed node (``max_expanded >= 1``).
- ``replay`` is unaffected: it keeps validating candidate paths only and
  accepts no budget parameter.

Cost limit (``max_cost``):
- ``max_cost`` may be omitted or ``None``, which keeps every key, value,
  exception type and validation order of the default behavior untouched.
  Otherwise it must be a non-negative integer that is not a bool: other
  types raise ``TypeError`` and negative values raise ``ValueError``.
  These checks run after all existing grid, costs, ``trace``,
  ``dynamic_blocked``, ``max_expanded`` and ``snapshot`` validation and
  before the search starts.
- The limit caps the cumulative entering-cell cost measured from the
  start: the start cell still costs 0 and entering a cell costs its
  ``costs`` value (1 without ``costs``). A candidate whose cumulative
  cost exceeds the limit is discarded and never closed. The limit
  applies identically in static mode, dynamic mode and ``plan_any`` and
  never changes the dynamic frames, the no-wait/no-revisit rules, the
  fixed tie-break, the minimum-total-cost choice or the multi-goal
  endpoint and complete-path lexicographic tie-breaks.
- When ``max_cost`` is provided the result always carries ``status``:
  ``"found"`` on success; otherwise ``path``/``cost`` are ``None``,
  ``expanded`` counts only actually closed nodes, and the status is
  ``"cost_exhausted"`` when at least one candidate was discarded for
  exceeding the limit and ``"unreachable"`` when none was. When
  ``max_expanded`` is also given, a budget stop takes priority and
  reports ``"budget_exhausted"``. ``trace`` still records only closed
  nodes. ``start == goal`` still returns the zero-cost single-point
  path, and the zero-expansion-budget rules keep priority.
- With ``snapshot=True`` the checkpoint records ``max_cost`` (and
  whether any candidate has already been discarded for exceeding it)
  whenever the limit is provided. ``resume`` treats a checkpoint
  without the field as ``None``; a present field is validated with the
  same rules (type errors raise ``TypeError``; negative values or a
  state inconsistent with the limit raise ``ValueError``), all before
  any searching, and a resumed search matches an uninterrupted call
  with the same cumulative budget in ``path``, ``cost``, ``status``,
  ``expanded`` and trace.
- ``replay`` accepts no ``max_cost`` parameter and is unaffected.

Dynamic obstacles (``dynamic_blocked``):
- ``dynamic_blocked`` may be omitted, ``None`` or an empty sequence, which
  leaves the static behavior above untouched. Otherwise it must be a
  sequence of frames; each frame is an iterable of grid coordinates that
  are blocked at that time frame. Frame 0 constrains ``start``; the
  coordinate at path index ``t`` must avoid both the static obstacles and
  frame ``t``, and frames beyond the last one reuse the last frame.
- Moves are still four-neighborhood only: no waiting in place and no
  repeated coordinates along the path. Because feasibility of a route
  depends on the coordinates it already visited, candidates that reach the
  same ``(x, y, t)`` through different coordinate histories are distinct
  states and are never merged: every route that satisfies the constraints
  can be explored. ``cost`` is still the sum of the costs of the cells
  entered; the start cell is never counted.
- The minimum-cost feasible route is returned. When total costs tie, the
  fixed priority is f, then h, then x, y, t, and candidates still tied are
  ordered lexicographically by their complete coordinate sequence, so the
  result never depends on the iteration order of the obstacle sets, the
  coordinates inside a frame, or the traversal order.
- In dynamic mode ``expanded_nodes`` records ``(x, y, t)`` triples in
  actual closing order and ``expanded`` counts closed space-time route
  states one per entry; a discarded candidate is never recorded. When the
  same ``(x, y, t)`` is closed through different histories, its triple
  appears once per closing in closing order, and ``expanded`` counts each
  such entry. On failure ``path``/``cost`` are ``None``.
- Validation: the outer value must be a sequence and every frame an
  iterable of coordinates; strings/bytes and non-two-integer coordinates
  raise ``TypeError``, out-of-bounds coordinates raise ``ValueError``,
  duplicates within a frame are merged, and a ``start`` blocked at frame 0
  raises ``ValueError``. All checks run before the search starts.

Multi-goal planning (``plan_any``):
- ``plan_any(width, height, blocked, start, goals, ...)`` accepts the same
  grid, blocked, start, costs, trace, dynamic_blocked and max_expanded
  arguments as ``plan``, plus a non-empty ``goals`` sequence of candidate
  endpoint coordinates. It returns one deterministic minimum-cost route to
  any of the candidates, in the same result structure as ``plan``; the
  path's last point is the winning endpoint.
- The grid, costs, trace, dynamic-blocked and budget checks run in
  ``plan``'s order with the same exception types; ``goals`` takes
  ``goal``'s position in that sequence. The outer value must be a
  sequence: strings, bytes, ``None`` and other non-sequences raise
  ``TypeError`` and an empty sequence raises ``ValueError``. Each endpoint
  follows the coordinate structure and integer rules (bad shape or
  coordinate types raise ``TypeError``); out-of-bounds coordinates and
  endpoints on static obstacles raise ``ValueError``. Repeated candidates
  merge, so the input ordering never changes any result.
- The search keeps four-neighborhood moves, no waiting, no revisits, the
  persistent last frame and the entering-cell cost accumulation. A
  candidate blocked by the dynamic frame at the time a particular route
  arrives is not rejected up front (its coordinate is statically valid);
  that arrival is simply infeasible for that route's time frame while
  other arrival times still compete.
- Total ``cost`` decides; ties compare the endpoint coordinate ``(x, y)``
  lexicographically, then the complete coordinate path lexicographically.
  As in dynamic ``plan``, routes reaching the same cell through different
  coordinate histories are distinct states and are never merged, so the
  traversal is a best-first walk of the feasible route tree; expansion
  order and the trace never depend on goal order, the blocked sets or
  in-frame iteration order. A budget never replaces an already determined
  better route with a worse one.
- When no endpoint is reachable, ``path`` and ``cost`` are ``None``. With a
  budget the ``status`` field follows the same ``"found"``,
  ``"unreachable"`` and ``"budget_exhausted"`` rules, and ``trace`` only
  records actually closed nodes: coordinate pairs in static mode and
  ``(x, y, t)`` triples in dynamic mode. A ``goals`` containing ``start``
  yields the single-point zero-cost route under the same start-equals-goal
  and zero-budget rules as ``plan``. A successful path verifies offline in
  ``replay`` when its last point is passed as ``goal``, with the same cost
  and step count.

Snapshots and resumable planning (``snapshot`` and ``resume``):
- ``plan`` and ``plan_any`` accept a final ``snapshot=False`` option.
  Omitting it or passing ``False`` keeps every key, value, exception type,
  validation order, tie-break, unreachable result and trace of the
  previous behavior untouched. A non-bool ``snapshot`` raises
  ``TypeError`` after all existing checks (the ``max_expanded`` validation
  included) and before the search starts.
- With ``snapshot=True``, a search that stops because ``max_expanded``
  was reached (``status == "budget_exhausted"``) additionally returns
  ``checkpoint``: a fully JSON-serializable snapshot of the planner state.
  Results that already found the goal or proved it unreachable never
  carry the field.
- The checkpoint records the snapshot version, the planner kind, the
  normalized grid constraints (dimensions, obstacles, start, goal or
  candidate goals, costs and dynamic frames), the cost limit when
  ``max_cost`` is provided, the closed-node count, the trace recorded
  so far and the complete pending-candidate state. Every
  set-derived list is stored sorted, so neither obstacle-set iteration
  order, goal order nor in-frame coordinate order influences the
  checkpoint's keys or values. A caller may store it as-is and hand it
  back to ``resume`` later.
- ``resume(checkpoint, max_expanded=None)`` continues the very same
  deterministic search described by the snapshot. ``max_expanded`` keeps
  its cumulative meaning: the total number of nodes that may be closed
  counted from the original start, so a limit not greater than the
  already closed count closes nothing more. The result is equivalent to
  one uninterrupted call with the same cumulative budget: ``path``,
  ``cost``, ``expanded``, ``status`` (present exactly when
  ``max_expanded`` is given) and ``expanded_nodes`` (always present in
  resume results; coordinate pairs in static mode, ``(x, y, t)`` triples
  in dynamic mode) all match, and repeated resumes never count an already
  closed node twice.
- When a resumed search reaches the budget again its result carries a
  fresh ``checkpoint`` that can be resumed in turn; ``found`` and
  ``unreachable`` results omit it. A successful resumed ``path`` verifies
  in ``replay`` with the same cost and step count.
- Validation: a checkpoint that is not an object, has missing required
  fields or wrongly typed fields raises ``TypeError``; an unsupported
  snapshot version or planner kind, grid constraints that violate the
  usual ``plan`` rules (obstacles, endpoints, costs, dynamic frames) or
  an internally inconsistent state raise ``ValueError``. Every check is
  decided before any searching, and ``resume``'s ``max_expanded`` follows
  the same non-negative-integer rules as ``plan``'s.

Offline replay (``replay``):
- ``replay`` re-checks a saved candidate path without running any search;
  it never returns ``expanded`` or ``expanded_nodes``. It accepts the same
  grid arguments as ``plan`` (no ``trace``) plus ``path``.
- The grid arguments are validated with exactly the public ``plan``
  checks, and all of them run before the path is judged, so the same
  malformed grid inputs raise the same ``TypeError``/``ValueError``.
- ``path`` must be a non-empty sequence of grid coordinates; strings,
  bytes, ``None`` and other non-sequences raise ``TypeError``, as do
  elements that are not exactly two integers. Coordinates outside the
  grid raise ``ValueError``.
- A structurally valid but semantically invalid path does not raise: it
  returns ``{"valid": False, "cost": None, "steps": None}`` with no
  partial accumulation. Invalid means: not beginning at ``start``, not
  ending at ``goal``, visiting a static obstacle, visiting a cell blocked
  by frame ``t`` at path index ``t`` (the last frame persists for later
  indices; omitted/empty ``dynamic_blocked`` means static-only checks),
  non-four-neighbor adjacency, repeated coordinates, or extra points when
  ``start == goal``.
- A valid path returns ``{"valid": True, "cost": int, "steps": int}``
  where ``steps == len(path) - 1`` and ``cost`` is recomputed with the
  same entering-cell rule ``plan`` uses (sum of the costs of the cells
  entered; unit costs without ``costs``; the start cell is never
  counted), so it equals the cost ``plan`` assigns to that same path.
- ``diagnose`` defaults to false; omitting it or passing ``False`` keeps
  every key, value, exception type and validation order identical. A
  non-bool ``diagnose`` raises ``TypeError`` after all grid checks
  (including the frame-0 check) and before the path structure is
  inspected. With ``diagnose=True`` both results additionally carry
  ``error`` and ``error_index``: ``None``/``None`` for a valid path, and
  for an invalid one a single error code plus the zero-based index of
  the first offending element, with ``cost``/``steps`` still ``None``.
  The codes, in fixed precedence order, are ``start_mismatch`` (index
  0), ``goal_mismatch`` (index of the last element),
  ``start_goal_extra`` (index 1), ``repeated_coordinate`` (the second
  occurrence), ``non_adjacent`` (the later element of the pair),
  ``static_blocked`` and ``dynamic_blocked`` (the offending element);
  ties resolve to the earliest rule in this list and no partial cost is
  ever returned.

Validation (all performed before the search starts):
- ``width``/``height`` must be positive integers.
- ``start``, ``goal`` and every entry of ``blocked`` must be grid
  coordinates: sequences of exactly two integers (tuples, lists, etc.),
  normalized to tuples.
- ``costs`` may be omitted or ``None`` (unit costs). Otherwise it must be a
  sequence of ``height`` rows, each a sequence of ``width`` positive
  integers; strings/bytes, non-integer cells and booleans are rejected.
- ``trace`` must be a bool; non-bool values raise ``TypeError``.
- Type or structure violations raise ``TypeError``; non-positive
  dimensions, out-of-bounds coordinates, endpoints on obstacles, wrong
  matrix shape, or non-positive cell costs raise ``ValueError``.
- Duplicate blocked cells are merged without changing the result.
"""

import heapq
from collections.abc import Iterable, Sequence

__all__ = ["plan", "plan_any", "replay", "resume"]

# Fixed neighbor generation order: +x, -x, +y, -y.
_NEIGHBORS = ((1, 0), (-1, 0), (0, 1), (0, -1))

# Checkpoint format version written by ``snapshot=True`` and required by
# ``resume``; bump whenever the snapshot layout changes.
_SNAPSHOT_VERSION = 1


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_dimension(value, name):
    if not _is_int(value):
        raise TypeError(
            f"{name} must be an int, got {type(value).__name__}"
        )
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}")
    return value


def _normalize_point(value, name):
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(
            f"{name} must be a sequence of two integers, "
            f"got {type(value).__name__}"
        )
    if len(value) != 2:
        raise TypeError(
            f"{name} must contain exactly two coordinates, got {len(value)}"
        )
    x, y = value[0], value[1]
    if not _is_int(x) or not _is_int(y):
        raise TypeError(f"{name} coordinates must be integers, got {value!r}")
    return (x, y)


def _check_bounds(point, width, height, name):
    x, y = point
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError(
            f"{name} {point} is outside the {width}x{height} grid"
        )


def _normalize_blocked(blocked, width, height):
    if isinstance(blocked, (str, bytes)) or not isinstance(blocked, Iterable):
        raise TypeError(
            f"blocked must be an iterable of grid coordinates, "
            f"got {type(blocked).__name__}"
        )
    cells = set()
    for index, item in enumerate(blocked):
        name = f"blocked[{index}]"
        point = _normalize_point(item, name)
        _check_bounds(point, width, height, name)
        cells.add(point)  # duplicates merge into one cell
    return cells


def _normalize_costs(costs, width, height):
    if costs is None:
        return None  # unit step cost
    if isinstance(costs, (str, bytes)) or not isinstance(costs, Sequence):
        raise TypeError(
            f"costs must be a height-by-width matrix of positive integers, "
            f"got {type(costs).__name__}"
        )
    if len(costs) != height:
        raise ValueError(
            f"costs must have exactly {height} rows, got {len(costs)}"
        )
    rows = []
    for y, row in enumerate(costs):
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
            raise TypeError(
                f"costs[{y}] must be a sequence of {width} positive "
                f"integers, got {type(row).__name__}"
            )
        if len(row) != width:
            raise ValueError(
                f"costs[{y}] must have exactly {width} cells, "
                f"got {len(row)}"
            )
        parsed = []
        for x, cell in enumerate(row):
            if not _is_int(cell):
                raise TypeError(
                    f"costs[{y}][{x}] must be an int, "
                    f"got {type(cell).__name__}"
                )
            if cell <= 0:
                raise ValueError(
                    f"costs[{y}][{x}] must be a positive integer, "
                    f"got {cell}"
                )
            parsed.append(cell)
        rows.append(tuple(parsed))
    return tuple(rows)


def _normalize_dynamic_blocked(dynamic_blocked, width, height):
    if dynamic_blocked is None:
        return None  # dynamic mode disabled
    if isinstance(dynamic_blocked, (str, bytes)) or not isinstance(
        dynamic_blocked, Sequence
    ):
        raise TypeError(
            f"dynamic_blocked must be a sequence of frames, "
            f"got {type(dynamic_blocked).__name__}"
        )
    if len(dynamic_blocked) == 0:
        return None  # no frames: equivalent to not provided
    frames = []
    for t, frame in enumerate(dynamic_blocked):
        if isinstance(frame, (str, bytes)) or not isinstance(frame, Iterable):
            raise TypeError(
                f"dynamic_blocked[{t}] must be an iterable of grid "
                f"coordinates, got {type(frame).__name__}"
            )
        cells = set()
        for index, item in enumerate(frame):
            name = f"dynamic_blocked[{t}][{index}]"
            point = _normalize_point(item, name)
            _check_bounds(point, width, height, name)
            cells.add(point)  # duplicates merge into one cell
        frames.append(frozenset(cells))
    return frames


def _normalize_path(path, width, height):
    if (path is None or isinstance(path, (str, bytes))
            or not isinstance(path, Sequence)):
        raise TypeError(
            f"path must be a non-empty sequence of grid coordinates, "
            f"got {type(path).__name__}"
        )
    if len(path) == 0:
        raise TypeError("path must be a non-empty sequence of coordinates")
    points = []
    for index, item in enumerate(path):
        name = f"path[{index}]"
        point = _normalize_point(item, name)
        _check_bounds(point, width, height, name)
        points.append(point)
    return points


def _normalize_goals(goals):
    # ``goals`` is the plan_any counterpart of ``plan``'s single ``goal``.
    # This performs the outer/structure pass in exactly the position where
    # ``plan`` structurally normalizes ``goal``: strings, bytes, ``None``
    # and other non-sequences are ``TypeError`` and an empty sequence is a
    # ``ValueError``; every element is normalized to a coordinate tuple
    # (element shape/coordinate type failures are ``TypeError``). Bounds and
    # obstacle checks follow in ``plan_any`` at the same relative positions
    # ``plan`` uses for its single goal. Duplicates merge and the tuple is
    # sorted so the input order never affects the result.
    if (goals is None or isinstance(goals, (str, bytes))
            or not isinstance(goals, Sequence)):
        raise TypeError(
            f"goals must be a non-empty sequence of grid coordinates, "
            f"got {type(goals).__name__}"
        )
    if len(goals) == 0:
        raise ValueError("goals must contain at least one candidate endpoint")
    points = set()
    for index, item in enumerate(goals):
        point = _normalize_point(item, f"goals[{index}]")
        points.add(point)
    return tuple(sorted(points))


def _validate_budget(max_expanded):
    # The shared non-negative-integer budget rules used by ``plan``,
    # ``plan_any`` and ``resume``: other types raise ``TypeError`` and
    # negative values raise ``ValueError``.
    if max_expanded is not None:
        if not _is_int(max_expanded):
            raise TypeError(
                f"max_expanded must be an int, "
                f"got {type(max_expanded).__name__}"
            )
        if max_expanded < 0:
            raise ValueError(
                f"max_expanded must be a non-negative integer, "
                f"got {max_expanded}"
            )


def _validate_snapshot_flag(snapshot):
    if not isinstance(snapshot, bool):
        raise TypeError(
            f"snapshot must be a bool, got {type(snapshot).__name__}"
        )


def _validate_max_cost(max_cost):
    # The shared non-negative-integer cost-limit rules used by ``plan``
    # and ``plan_any`` (and by ``resume`` for a checkpoint that records
    # the field): other types raise ``TypeError`` and negative values
    # raise ``ValueError``.
    if max_cost is not None:
        if not _is_int(max_cost):
            raise TypeError(
                f"max_cost must be an int, "
                f"got {type(max_cost).__name__}"
            )
        if max_cost < 0:
            raise ValueError(
                f"max_cost must be a non-negative integer, "
                f"got {max_cost}"
            )


def _search_static(width, height, obstacles, start, goal, costs, trace,
                   max_expanded, snapshot=False, state=None, max_cost=None):
    """Classic static-grid A* (the ``plan`` search without frames).

    Cells are merged by coordinate: the best known ``g`` per cell is kept
    in ``g_score``, stale heap entries are skipped, and ``came_from``
    reconstructs the cheapest route once the goal is closed. With
    ``snapshot=True`` a budget stop additionally returns a checkpoint of
    the complete search state; ``state`` carries such a snapshot back in
    so ``resume`` continues the identical traversal. The snapshot needs
    the trace even when the caller did not ask for it, so nodes are
    recorded whenever ``trace`` or ``snapshot`` is true. With
    ``max_cost`` a candidate whose cumulative entering cost exceeds the
    limit is discarded instead of being queued, and ``cost_pruned``
    records whether that ever happened (it decides the failure status
    and is part of the checkpointed state).
    """

    def heuristic(point):
        return abs(point[0] - goal[0]) + abs(point[1] - goal[1])

    def step_cost(point):
        # Cost of entering ``point``; the start cell is never entered.
        if costs is None:
            return 1
        return costs[point[1]][point[0]]

    if state is None:
        # Heap entries are (f, h, x, y, point): the first four fields give
        # a total, input-order-independent ordering, so the point itself
        # is never compared.
        h0 = heuristic(start)
        open_heap = [(h0, h0, start[0], start[1], start)]
        came_from = {}
        g_score = {start: 0}
        closed = set()
        expanded_nodes = [] if trace or snapshot else None
        cost_pruned = False
    else:
        open_heap = state["open"]
        came_from = state["came_from"]
        g_score = state["g_score"]
        closed = state["closed"]
        expanded_nodes = state["expanded_nodes"]
        cost_pruned = state["cost_pruned"]
    budget_stop = False

    while open_heap:
        entry = heapq.heappop(open_heap)
        current = entry[4]
        if current in closed:
            continue  # stale heap entry; already closed with its best g
        if max_expanded is not None and len(closed) >= max_expanded:
            # A live candidate remains but the budget is spent; close
            # nothing more (this also covers start == goal with a zero
            # budget: no node is closed and no start is fabricated). The
            # popped candidate was already removed from the heap, so for a
            # snapshot it is pushed back: the checkpoint must hold every
            # still-live candidate for the search to resume exactly.
            budget_stop = True
            if snapshot:
                heapq.heappush(open_heap, entry)
            break
        closed.add(current)
        if expanded_nodes is not None:
            expanded_nodes.append(current)
        if current == goal:
            path = [current]
            while path[-1] in came_from:
                path.append(came_from[path[-1]])
            path.reverse()
            result = {
                "path": path,
                "cost": g_score[current],
                "expanded": len(closed),
            }
            if max_expanded is not None or max_cost is not None:
                result["status"] = "found"
            if trace:
                result["expanded_nodes"] = expanded_nodes
            return result
        for dx, dy in _NEIGHBORS:
            nxt = (current[0] + dx, current[1] + dy)
            if not (0 <= nxt[0] < width and 0 <= nxt[1] < height):
                continue
            if nxt in obstacles:
                continue
            new_g = g_score[current] + step_cost(nxt)
            if new_g < g_score.get(nxt, float("inf")):
                if max_cost is not None and new_g > max_cost:
                    # Cumulative entering cost over the limit: the
                    # candidate is discarded without being closed.
                    cost_pruned = True
                    continue
                g_score[nxt] = new_g
                came_from[nxt] = current
                h = heuristic(nxt)
                heapq.heappush(
                    open_heap, (new_g + h, h, nxt[0], nxt[1], nxt)
                )
    result = {"path": None, "cost": None, "expanded": len(closed)}
    if max_expanded is not None or max_cost is not None:
        if budget_stop:
            result["status"] = "budget_exhausted"
        elif cost_pruned:
            result["status"] = "cost_exhausted"
        else:
            result["status"] = "unreachable"
    if trace:
        result["expanded_nodes"] = expanded_nodes
    if snapshot and result.get("status") == "budget_exhausted":
        result["checkpoint"] = _snapshot_checkpoint(
            "plan", width, height, obstacles, start, goal, costs, None,
            len(closed), expanded_nodes,
            _static_snapshot_state(open_heap, came_from, g_score, closed),
            max_cost, cost_pruned,
        )
    return result


def _search_dynamic(width, height, obstacles, frames, start, goal, costs,
                    trace, max_expanded, snapshot=False, state=None,
                    max_cost=None):
    """History-sensitive time-expanded A*.

    Frame ``t`` constrains the cell occupied at path index ``t``; frames
    past the last one reuse the last frame. Waiting in place and repeated
    coordinates are forbidden, so whether a candidate route can be
    extended depends on the exact sequence of cells it already visited:
    two routes reaching the same ``(x, y, t)`` with different histories
    are distinct states and neither may prune the other (the route with
    the larger accumulated cost can be the only one that remains
    extendable). Each heap node therefore carries its complete path; the
    search is a best-first traversal of the feasible route tree. Goal
    closings are collected until the heap's smallest f exceeds the best
    goal cost, after which the minimum-cost goal tie-break is settled by
    the fixed f, h, x, y, t priority and then the complete path's
    lexicographic order. With ``snapshot=True`` a budget stop additionally
    returns a checkpoint of the complete route-tree state; ``state``
    carries such a snapshot back in for ``resume``. With ``max_cost`` a
    candidate whose cumulative entering cost exceeds the limit is
    discarded instead of being queued, and ``cost_pruned`` records
    whether that ever happened.
    """

    def heuristic(point):
        return abs(point[0] - goal[0]) + abs(point[1] - goal[1])

    def step_cost(point):
        # Cost of entering ``point``; the start cell is never entered.
        if costs is None:
            return 1
        return costs[point[1]][point[0]]

    last_frame = len(frames) - 1

    def frame_cells(t):
        return frames[t] if t <= last_frame else frames[last_frame]

    # Heap entries are (f, h, x, y, t, g, path, seen): f/h/x/y/t give the
    # fixed numeric priority and candidates still tied are ordered by the
    # complete coordinate path lexicographically, so ordering never depends
    # on obstacle sets, in-frame coordinate order, or traversal order. The
    # frozenset ``seen`` mirrors ``path`` for an O(1) repeat check and is
    # never compared (distinct histories always differ in ``path``).
    if state is None:
        h0 = heuristic(start)
        start_path = (start,)
        open_heap = [(h0, h0, start[0], start[1], 0, 0, start_path,
                      frozenset(start_path))]
        closed_count = 0
        expanded_nodes = [] if trace or snapshot else None
        # Best goal closing seen so far, keyed exactly as the tie-break
        # specializes at the goal: (g, t, path) (there f == g, h == 0 and
        # (x, y) is fixed). Closing a goal does not stop the search
        # immediately: an equally cheap route whose chain runs through
        # higher-h nodes might still be undeveloped. Since each move costs
        # at least 1 and the Manhattan h changes by at most 1 per move,
        # f = g + h never decreases along a route; once the smallest f on
        # the heap exceeds the best goal cost, no route of that cost (or
        # less) can remain undiscovered.
        best_key = None
        cost_pruned = False
    else:
        open_heap = state["open"]
        closed_count = state["closed_count"]
        expanded_nodes = state["expanded_nodes"]
        best_key = state["best_key"]
        cost_pruned = state["cost_pruned"]
    budget_stop = False

    while open_heap:
        if best_key is not None and open_heap[0][0] > best_key[0]:
            break
        if max_expanded is not None and closed_count >= max_expanded:
            # Live route candidates remain but the budget is spent; close
            # nothing more. A goal already closed still wins below.
            budget_stop = True
            break
        _, _, x, y, t, g, path, seen = heapq.heappop(open_heap)
        closed_count += 1  # every popped route node closes exactly once
        if expanded_nodes is not None:
            # Histories reaching the same (x, y, t) each record a triple,
            # in the order their route nodes are actually closed.
            expanded_nodes.append((x, y, t))
        if (x, y) == goal:
            goal_key = (g, t, path)
            if best_key is None or goal_key < best_key:
                best_key = goal_key
            continue  # routes end at the goal; never expanded past it
        next_t = t + 1
        frame = frame_cells(next_t)
        for dx, dy in _NEIGHBORS:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            nxt = (nx, ny)
            if nxt in obstacles or nxt in frame or nxt in seen:
                continue  # static obstacle, timed obstacle, or revisit
            new_g = g + step_cost(nxt)
            if max_cost is not None and new_g > max_cost:
                # Cumulative entering cost over the limit: the candidate
                # is discarded without being closed.
                cost_pruned = True
                continue
            new_path = path + (nxt,)
            h = heuristic(nxt)
            heapq.heappush(
                open_heap,
                (new_g + h, h, nx, ny, next_t, new_g, new_path,
                 seen | {nxt}),
            )
    if best_key is None:
        result = {"path": None, "cost": None, "expanded": closed_count}
        if max_expanded is not None or max_cost is not None:
            if budget_stop:
                result["status"] = "budget_exhausted"
            elif cost_pruned:
                result["status"] = "cost_exhausted"
            else:
                result["status"] = "unreachable"
    else:
        best_g, _, best_path = best_key
        result = {
            "path": list(best_path),
            "cost": best_g,
            "expanded": closed_count,
        }
        if max_expanded is not None or max_cost is not None:
            result["status"] = "found"
    if trace:
        result["expanded_nodes"] = expanded_nodes
    if snapshot and result.get("status") == "budget_exhausted":
        result["checkpoint"] = _snapshot_checkpoint(
            "plan", width, height, obstacles, start, goal, costs, frames,
            closed_count, expanded_nodes,
            _tree_snapshot_state(open_heap, best_key, "plan"),
            max_cost, cost_pruned,
        )
    return result


def _search_any(width, height, obstacles, frames, start, goals, costs,
                trace, max_expanded, snapshot=False, state=None,
                max_cost=None):
    """History-sensitive time-expanded A* over several candidate goals.

    This is the ``plan_any`` counterpart of ``_search_dynamic``. The only
    semantic differences are: the heuristic is the minimum Manhattan
    distance to any candidate endpoint, and a closing at any goal
    coordinate is a goal closing that terminates the route. As in
    ``_search_dynamic`` the feasibility of a route depends on its exact
    coordinate history (no waiting, no repeated coordinates, and dynamic
    frames), so every node carries its complete path and routes reaching
    the same cell through different histories are never merged; this is
    also what makes the complete-path lexicographic tie-break exact.
    Static mode is the same traversal with the time dimension fixed at 0
    and no timed frames, so the trace records plain coordinate pairs
    there, exactly as ``plan`` does. With ``snapshot=True`` a budget stop
    additionally returns a checkpoint of the complete route-tree state;
    ``state`` carries such a snapshot back in for ``resume``. With
    ``max_cost`` a candidate whose cumulative entering cost exceeds the
    limit is discarded instead of being queued, and ``cost_pruned``
    records whether that ever happened.
    """

    def heuristic(point):
        return min(
            abs(point[0] - goal[0]) + abs(point[1] - goal[1])
            for goal in goals
        )

    def step_cost(point):
        # Cost of entering ``point``; the start cell is never entered.
        if costs is None:
            return 1
        return costs[point[1]][point[0]]

    dynamic = frames is not None
    last_frame = len(frames) - 1 if dynamic else None

    def frame_cells(t):
        return frames[t] if t <= last_frame else frames[last_frame]

    # Heap entries are (f, h, x, y, t, g, path, seen), keyed exactly like
    # ``_search_dynamic``: the fixed numeric priority comes first and
    # candidates still tied are ordered by the complete coordinate path
    # lexicographically, so ordering never depends on goal order, obstacle
    # sets, in-frame coordinate order, or traversal order. Static mode
    # keeps t fixed at 0; the frozenset ``seen`` mirrors ``path`` for an
    # O(1) repeat check and is never compared.
    if state is None:
        h0 = heuristic(start)
        start_path = (start,)
        open_heap = [(h0, h0, start[0], start[1], 0, 0, start_path,
                      frozenset(start_path))]
        closed_count = 0
        expanded_nodes = [] if trace or snapshot else None
        # Best goal closing seen so far, keyed as (g, (x, y), path):
        # minimum total cost first, then the endpoint coordinate's
        # lexicographic order, then the complete route's lexicographic
        # order. Closing a goal does not stop the search immediately: an
        # equally cheap route to a smaller goal whose chain runs through
        # higher-h nodes might still be undeveloped. Every move costs at
        # least 1 and the minimum Manhattan heuristic changes by at most 1
        # per move, so f = g + h never decreases along a route; once the
        # smallest f on the heap exceeds the best goal cost, no route of
        # that cost (or less) can remain undiscovered.
        best_key = None
        cost_pruned = False
    else:
        open_heap = state["open"]
        closed_count = state["closed_count"]
        expanded_nodes = state["expanded_nodes"]
        best_key = state["best_key"]
        cost_pruned = state["cost_pruned"]
    budget_stop = False

    while open_heap:
        if best_key is not None and open_heap[0][0] > best_key[0]:
            break
        if max_expanded is not None and closed_count >= max_expanded:
            # Live route candidates remain but the budget is spent; close
            # nothing more. A goal already closed still wins below.
            budget_stop = True
            break
        _, _, x, y, t, g, path, seen = heapq.heappop(open_heap)
        closed_count += 1  # every popped route node closes exactly once
        if expanded_nodes is not None:
            # Static mode records coordinate pairs, dynamic mode the
            # (x, y, t) triple; distinct histories each record an entry in
            # actual closing order.
            expanded_nodes.append((x, y, t) if dynamic else (x, y))
        cell = (x, y)
        if cell in goals:
            goal_key = (g, cell, path)
            if best_key is None or goal_key < best_key:
                best_key = goal_key
            continue  # routes end at a goal; never expanded past one
        next_t = t + 1 if dynamic else 0
        frame = frame_cells(next_t) if dynamic else frozenset()
        for dx, dy in _NEIGHBORS:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            nxt = (nx, ny)
            if nxt in obstacles or nxt in frame or nxt in seen:
                continue  # static obstacle, timed obstacle, or revisit
            new_g = g + step_cost(nxt)
            if max_cost is not None and new_g > max_cost:
                # Cumulative entering cost over the limit: the candidate
                # is discarded without being closed.
                cost_pruned = True
                continue
            new_path = path + (nxt,)
            h = heuristic(nxt)
            heapq.heappush(
                open_heap,
                (new_g + h, h, nx, ny, next_t, new_g, new_path,
                 seen | {nxt}),
            )
    if best_key is None:
        result = {"path": None, "cost": None, "expanded": closed_count}
        if max_expanded is not None or max_cost is not None:
            if budget_stop:
                result["status"] = "budget_exhausted"
            elif cost_pruned:
                result["status"] = "cost_exhausted"
            else:
                result["status"] = "unreachable"
    else:
        best_g, _, best_path = best_key
        result = {
            "path": list(best_path),
            "cost": best_g,
            "expanded": closed_count,
        }
        if max_expanded is not None or max_cost is not None:
            result["status"] = "found"
    if trace:
        result["expanded_nodes"] = expanded_nodes
    if snapshot and result.get("status") == "budget_exhausted":
        result["checkpoint"] = _snapshot_checkpoint(
            "plan_any", width, height, obstacles, start, goals, costs,
            frames, closed_count, expanded_nodes,
            _tree_snapshot_state(open_heap, best_key, "plan_any"),
            max_cost, cost_pruned,
        )
    return result


def _snapshot_checkpoint(planner, width, height, obstacles, start, endpoints,
                         costs, frames, closed_count, expanded_nodes, state,
                         max_cost=None, cost_pruned=False):
    # The checkpoint is built from JSON-native values only (ints, strings,
    # lists, dicts, ``None``) in one fixed key order, and every list
    # derived from a set is sorted, so neither set iteration order, goal
    # order nor in-frame coordinate order can influence the result.
    checkpoint = {
        "version": _SNAPSHOT_VERSION,
        "planner": planner,
        "width": width,
        "height": height,
        "blocked": [[cell[0], cell[1]] for cell in sorted(obstacles)],
        "start": [start[0], start[1]],
    }
    if planner == "plan":
        checkpoint["goal"] = [endpoints[0], endpoints[1]]
    else:
        checkpoint["goals"] = [[goal[0], goal[1]] for goal in endpoints]
    checkpoint["costs"] = (
        [list(row) for row in costs] if costs is not None else None
    )
    checkpoint["dynamic_blocked"] = (
        [[[cell[0], cell[1]] for cell in sorted(frame)]
         for frame in frames]
        if frames is not None else None
    )
    if max_cost is not None:
        # The cost limit and whether any candidate has already been
        # discarded for exceeding it; both are needed for a resumed
        # search to reproduce the uninterrupted failure status. The
        # keys are omitted entirely when no limit is in effect, so
        # checkpoints of pre-max_cost calls keep their exact shape.
        checkpoint["max_cost"] = max_cost
        checkpoint["cost_pruned"] = cost_pruned
    checkpoint["closed"] = closed_count
    checkpoint["trace"] = [list(entry) for entry in expanded_nodes]
    checkpoint["state"] = state
    return checkpoint


def _static_snapshot_state(open_heap, came_from, g_score, closed):
    # The complete static-A* state: the heap array as it stands (restoring
    # it verbatim keeps the exact future pop order), the route
    # reconstruction links, the best-known costs and the closed set.
    return {
        "open": [[entry[0], entry[1], entry[2], entry[3]]
                 for entry in open_heap],
        "came_from": [
            [[point[0], point[1]], [parent[0], parent[1]]]
            for point, parent in sorted(came_from.items())
        ],
        "g_score": [
            [[point[0], point[1]], g]
            for point, g in sorted(g_score.items())
        ],
        "closed": [[cell[0], cell[1]] for cell in sorted(closed)],
    }


def _tree_snapshot_state(open_heap, best_key, planner):
    # The complete route-tree state: every live candidate with its full
    # coordinate history (the ``seen`` frozenset is derived from the path
    # on restore) plus the best goal closing recorded so far, keyed as
    # (g, t, path) for ``plan`` and (g, (x, y), path) for ``plan_any``.
    if best_key is None:
        best = None
    else:
        best_g, best_tie, best_path = best_key
        best = [
            best_g,
            [best_tie[0], best_tie[1]] if planner == "plan_any" else best_tie,
            [[point[0], point[1]] for point in best_path],
        ]
    return {
        "open": [
            [f, h, x, y, t, g, [[point[0], point[1]] for point in path]]
            for f, h, x, y, t, g, path, _seen in open_heap
        ],
        "best": best,
    }


def _snapshot_path(raw, name):
    # A candidate's complete coordinate history inside a checkpoint:
    # structurally a non-empty sequence of two-integer coordinates.
    if (isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence)
            or len(raw) == 0):
        raise TypeError(
            f"{name} must be a non-empty sequence of grid coordinates, "
            f"got {type(raw).__name__}"
        )
    return tuple(
        _normalize_point(item, f"{name}[{index}]")
        for index, item in enumerate(raw)
    )


def _check_heap_order(keys):
    # A genuine checkpoint stores the heap array exactly as the search
    # left it, so every parent must sort at or before its children.
    for index, key in enumerate(keys):
        for child in (2 * index + 1, 2 * index + 2):
            if child < len(keys) and keys[child] < key:
                raise ValueError(
                    "checkpoint candidates are not in heap order"
                )


def _restore_trace(raw, dynamic, unique, width, height, obstacles, frames):
    # The recorded trace: coordinate pairs in static mode, ``(x, y, t)``
    # triples in dynamic mode. Structure problems are ``TypeError``; a
    # trace that could not have been produced by the search is a
    # ``ValueError`` (out-of-bounds cells, blocked cells, negative times,
    # or duplicates where the search never records any).
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise TypeError(
            f"checkpoint trace must be a sequence of recorded nodes, "
            f"got {type(raw).__name__}"
        )
    arity = 3 if dynamic else 2
    entries = []
    for index, item in enumerate(raw):
        name = f"checkpoint trace[{index}]"
        if (isinstance(item, (str, bytes)) or not isinstance(item, Sequence)
                or len(item) != arity):
            raise TypeError(
                f"{name} must be a sequence of {arity} integers"
            )
        if not all(_is_int(value) for value in item):
            raise TypeError(f"{name} must contain only integers")
        entries.append(tuple(item))
    last_frame = len(frames) - 1 if dynamic else None
    seen = set()
    for index, entry in enumerate(entries):
        cell = (entry[0], entry[1])
        _check_bounds(cell, width, height, f"checkpoint trace[{index}]")
        if cell in obstacles:
            raise ValueError(
                f"checkpoint trace[{index}] {cell} lies on a blocked cell"
            )
        if dynamic:
            t = entry[2]
            if t < 0:
                raise ValueError(
                    f"checkpoint trace[{index}] has a negative time {t}"
                )
            frame = frames[t] if t <= last_frame else frames[last_frame]
            if cell in frame:
                raise ValueError(
                    f"checkpoint trace[{index}] {cell} is blocked at "
                    f"frame {t}"
                )
        elif unique:
            # The static single-goal search closes each cell at most once.
            if entry in seen:
                raise ValueError(
                    "checkpoint trace records a node twice"
                )
            seen.add(entry)
    return entries


def _restore_static_state(raw, width, height, obstacles, start, goal,
                          trace, max_cost=None):
    # Rebuild and fully cross-check the static-A* snapshot state.
    if not isinstance(raw, dict):
        raise TypeError(
            f"checkpoint state must be an object, got {type(raw).__name__}"
        )
    for key in ("open", "came_from", "g_score", "closed"):
        if key not in raw:
            raise TypeError(
                f"checkpoint state is missing required field {key!r}"
            )
    # ---- structural pass: shapes and types only ----
    raw_open = raw["open"]
    if isinstance(raw_open, (str, bytes)) or not isinstance(raw_open, Sequence):
        raise TypeError(
            f"checkpoint state open must be a sequence of candidates, "
            f"got {type(raw_open).__name__}"
        )
    open_entries = []
    for index, item in enumerate(raw_open):
        name = f"checkpoint state open[{index}]"
        if (isinstance(item, (str, bytes)) or not isinstance(item, Sequence)
                or len(item) != 4):
            raise TypeError(f"{name} must be a [f, h, x, y] candidate")
        if not all(_is_int(value) for value in item):
            raise TypeError(f"{name} must contain only integers")
        open_entries.append(tuple(item))
    raw_came = raw["came_from"]
    if isinstance(raw_came, (str, bytes)) or not isinstance(raw_came, Sequence):
        raise TypeError(
            f"checkpoint state came_from must be a sequence of links, "
            f"got {type(raw_came).__name__}"
        )
    came_pairs = []
    for index, item in enumerate(raw_came):
        name = f"checkpoint state came_from[{index}]"
        if (isinstance(item, (str, bytes)) or not isinstance(item, Sequence)
                or len(item) != 2):
            raise TypeError(f"{name} must be a [cell, parent] pair")
        came_pairs.append((
            _normalize_point(item[0], f"{name}[0]"),
            _normalize_point(item[1], f"{name}[1]"),
        ))
    raw_g = raw["g_score"]
    if isinstance(raw_g, (str, bytes)) or not isinstance(raw_g, Sequence):
        raise TypeError(
            f"checkpoint state g_score must be a sequence of entries, "
            f"got {type(raw_g).__name__}"
        )
    g_pairs = []
    for index, item in enumerate(raw_g):
        name = f"checkpoint state g_score[{index}]"
        if (isinstance(item, (str, bytes)) or not isinstance(item, Sequence)
                or len(item) != 2):
            raise TypeError(f"{name} must be a [cell, cost] pair")
        point = _normalize_point(item[0], f"{name}[0]")
        if not _is_int(item[1]):
            raise TypeError(
                f"{name}[1] must be an int, got {type(item[1]).__name__}"
            )
        g_pairs.append((point, item[1]))
    raw_closed = raw["closed"]
    if isinstance(raw_closed, (str, bytes)) or not isinstance(raw_closed, Sequence):
        raise TypeError(
            f"checkpoint state closed must be a sequence of cells, "
            f"got {type(raw_closed).__name__}"
        )
    closed_cells = [
        _normalize_point(item, f"checkpoint state closed[{index}]")
        for index, item in enumerate(raw_closed)
    ]
    # ---- semantic pass: the state must be one the search could reach ----
    if not open_entries:
        raise ValueError("checkpoint state has no pending candidates")
    _check_heap_order([(f, h, x, y) for f, h, x, y in open_entries])
    g_score = {}
    for point, g in g_pairs:
        _check_bounds(point, width, height, "checkpoint state g_score")
        if point in obstacles:
            raise ValueError(
                f"checkpoint state g_score {point} lies on a blocked cell"
            )
        if point in g_score:
            raise ValueError(
                "checkpoint state g_score records a cell twice"
            )
        g_score[point] = g
    if g_score.get(start) != 0:
        raise ValueError("checkpoint state does not seed the start cell")
    for point, g in g_score.items():
        if point != start and g <= 0:
            raise ValueError(
                "checkpoint state g_score holds a non-positive cost"
            )
    if max_cost is not None:
        for point, g in g_score.items():
            if g > max_cost:
                raise ValueError(
                    "checkpoint state cost exceeds its max_cost"
                )
    came_from = {}
    for point, parent in came_pairs:
        _check_bounds(point, width, height, "checkpoint state came_from")
        _check_bounds(parent, width, height, "checkpoint state came_from")
        if point in obstacles or parent in obstacles:
            raise ValueError(
                "checkpoint state came_from involves a blocked cell"
            )
        if point in came_from:
            raise ValueError(
                "checkpoint state came_from records a cell twice"
            )
        if point == start:
            raise ValueError(
                "checkpoint state came_from records the start cell"
            )
        if (abs(point[0] - parent[0]) + abs(point[1] - parent[1])) != 1:
            raise ValueError(
                "checkpoint state came_from link is not four-connected"
            )
        came_from[point] = parent
    if set(came_from) != set(g_score) - {start}:
        raise ValueError(
            "checkpoint state came_from does not match its g_score"
        )
    closed_set = set()
    for cell in closed_cells:
        _check_bounds(cell, width, height, "checkpoint state closed")
        if cell in obstacles:
            raise ValueError(
                f"checkpoint state closed {cell} lies on a blocked cell"
            )
        if cell in closed_set:
            raise ValueError(
                "checkpoint state closed records a cell twice"
            )
        closed_set.add(cell)
    if closed_set != set(trace):
        raise ValueError("checkpoint closed cells do not match its trace")
    if not closed_set <= set(g_score):
        raise ValueError("checkpoint closed cells are missing from g_score")
    for parent in came_from.values():
        if parent not in closed_set:
            raise ValueError(
                "checkpoint state came_from parent was never closed"
            )
    live = False
    for f, h, x, y in open_entries:
        cell = (x, y)
        _check_bounds(cell, width, height, "checkpoint state open")
        if cell in obstacles:
            raise ValueError(
                f"checkpoint state open {cell} lies on a blocked cell"
            )
        if h != abs(x - goal[0]) + abs(y - goal[1]):
            raise ValueError(
                "checkpoint candidate heuristic is inconsistent"
            )
        if cell not in g_score:
            raise ValueError(
                "checkpoint candidate is missing from g_score"
            )
        if f < g_score[cell] + h:
            raise ValueError(
                "checkpoint candidate priority is inconsistent"
            )
        if cell not in closed_set:
            live = True
    if not live:
        raise ValueError("checkpoint state has no live pending candidates")
    return {
        "open": [(f, h, x, y, (x, y)) for f, h, x, y in open_entries],
        "came_from": came_from,
        "g_score": g_score,
        "closed": closed_set,
        "expanded_nodes": list(trace),
    }


def _restore_tree_state(raw, planner, dynamic, width, height, obstacles,
                        frames, start, goal, goals, costs, closed_count,
                        trace, max_cost=None):
    # Rebuild and fully cross-check a route-tree snapshot state (dynamic
    # ``plan``, and ``plan_any`` in both static and dynamic mode).
    if not isinstance(raw, dict):
        raise TypeError(
            f"checkpoint state must be an object, got {type(raw).__name__}"
        )
    for key in ("open", "best"):
        if key not in raw:
            raise TypeError(
                f"checkpoint state is missing required field {key!r}"
            )

    def heuristic(point):
        if planner == "plan":
            return abs(point[0] - goal[0]) + abs(point[1] - goal[1])
        return min(
            abs(point[0] - end[0]) + abs(point[1] - end[1])
            for end in goals
        )

    def step_cost(point):
        # Cost of entering ``point``; the start cell is never entered.
        if costs is None:
            return 1
        return costs[point[1]][point[0]]

    last_frame = len(frames) - 1 if dynamic else None

    # ---- structural pass: shapes and types only ----
    raw_open = raw["open"]
    if isinstance(raw_open, (str, bytes)) or not isinstance(raw_open, Sequence):
        raise TypeError(
            f"checkpoint state open must be a sequence of candidates, "
            f"got {type(raw_open).__name__}"
        )
    entries = []
    for index, item in enumerate(raw_open):
        name = f"checkpoint state open[{index}]"
        if (isinstance(item, (str, bytes)) or not isinstance(item, Sequence)
                or len(item) != 7):
            raise TypeError(
                f"{name} must be an [f, h, x, y, t, g, path] candidate"
            )
        f, h, x, y, t, g, raw_path = item
        if not all(_is_int(value) for value in (f, h, x, y, t, g)):
            raise TypeError(f"{name} must contain only integer fields")
        entries.append((f, h, x, y, t, g,
                        _snapshot_path(raw_path, f"{name} path")))
    raw_best = raw["best"]
    best = None
    if raw_best is not None:
        if (isinstance(raw_best, (str, bytes))
                or not isinstance(raw_best, Sequence) or len(raw_best) != 3):
            raise TypeError(
                "checkpoint state best must be null or a [g, key, path] "
                "entry"
            )
        best_g, best_tie, best_path_raw = raw_best
        if not _is_int(best_g):
            raise TypeError("checkpoint state best cost must be an int")
        if planner == "plan_any":
            best_tie = _normalize_point(best_tie,
                                        "checkpoint state best goal")
        elif not _is_int(best_tie):
            raise TypeError("checkpoint state best time must be an int")
        best = (best_g, best_tie,
                _snapshot_path(best_path_raw, "checkpoint state best path"))

    # ---- semantic pass: every candidate must be a feasible route state ----
    if not entries:
        raise ValueError("checkpoint state has no pending candidates")
    _check_heap_order([
        (f, h, x, y, t, g, path) for f, h, x, y, t, g, path in entries
    ])

    def check_route(path, t, g):
        if path[0] != start:
            raise ValueError(
                "checkpoint candidate path does not begin at start"
            )
        expected_t = len(path) - 1 if dynamic else 0
        if t != expected_t:
            raise ValueError(
                "checkpoint candidate time does not match its path"
            )
        seen = set()
        total = 0
        previous = None
        for index, cell in enumerate(path):
            _check_bounds(cell, width, height, "checkpoint candidate")
            if cell in obstacles:
                raise ValueError(
                    f"checkpoint candidate {cell} lies on a blocked cell"
                )
            if cell in seen:
                raise ValueError(
                    "checkpoint candidate path repeats a coordinate"
                )
            if previous is not None:
                if (abs(cell[0] - previous[0])
                        + abs(cell[1] - previous[1])) != 1:
                    raise ValueError(
                        "checkpoint candidate path is not four-connected"
                    )
                total += step_cost(cell)
            if dynamic:
                frame = (frames[index] if index <= last_frame
                         else frames[last_frame])
                if cell in frame:
                    raise ValueError(
                        f"checkpoint candidate {cell} is blocked at "
                        f"frame {index}"
                    )
            seen.add(cell)
            previous = cell
        if g != total:
            raise ValueError(
                "checkpoint candidate cost does not match its path"
            )

    for f, h, x, y, t, g, path in entries:
        check_route(path, t, g)
        if path[-1] != (x, y):
            raise ValueError(
                "checkpoint candidate head does not match its path"
            )
        if h != heuristic((x, y)):
            raise ValueError(
                "checkpoint candidate heuristic is inconsistent"
            )
        if f != g + h:
            raise ValueError(
                "checkpoint candidate priority is inconsistent"
            )
        if max_cost is not None and g > max_cost:
            raise ValueError(
                "checkpoint candidate cost exceeds its max_cost"
            )
    if best is not None:
        best_g, best_tie, best_path = best
        check_route(best_path, len(best_path) - 1 if dynamic else 0, best_g)
        if max_cost is not None and best_g > max_cost:
            raise ValueError(
                "checkpoint best route cost exceeds its max_cost"
            )
        endpoint = best_path[-1]
        if planner == "plan_any":
            if endpoint not in goals:
                raise ValueError(
                    "checkpoint best route does not end at a candidate goal"
                )
            if best_tie != endpoint:
                raise ValueError(
                    "checkpoint best key does not match its endpoint"
                )
        else:
            if endpoint != goal:
                raise ValueError(
                    "checkpoint best route does not end at the goal"
                )
            if best_tie != len(best_path) - 1:
                raise ValueError(
                    "checkpoint best key does not match its arrival time"
                )
        if entries[0][0] > best_g:
            raise ValueError(
                "checkpoint best route should already have been returned"
            )
    return {
        "open": [
            (f, h, x, y, t, g, path, frozenset(path))
            for f, h, x, y, t, g, path in entries
        ],
        "best_key": best,
        "closed_count": closed_count,
        "expanded_nodes": list(trace),
    }


def _restore_checkpoint(checkpoint):
    # Validate a snapshot produced with ``snapshot=True`` and normalize it
    # back into the planner's internal values. Missing fields and wrong
    # types raise ``TypeError``; an unsupported version or planner, grid
    # constraints that violate the usual ``plan`` rules, or an internally
    # inconsistent state raise ``ValueError``. Everything is decided here,
    # before any searching happens.
    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"checkpoint must be a snapshot object, "
            f"got {type(checkpoint).__name__}"
        )
    for key in ("version", "planner", "width", "height", "blocked", "start",
                "costs", "dynamic_blocked", "closed", "trace", "state"):
        if key not in checkpoint:
            raise TypeError(
                f"checkpoint is missing required field {key!r}"
            )
    version = checkpoint["version"]
    if not _is_int(version):
        raise TypeError(
            f"checkpoint version must be an int, "
            f"got {type(version).__name__}"
        )
    planner = checkpoint["planner"]
    if not isinstance(planner, str):
        raise TypeError(
            f"checkpoint planner must be a string, "
            f"got {type(planner).__name__}"
        )
    if version != _SNAPSHOT_VERSION:
        raise ValueError(
            f"checkpoint version {version} is not supported"
        )
    if planner not in ("plan", "plan_any"):
        raise ValueError(
            f"checkpoint planner {planner!r} is not supported"
        )
    endpoint_key = "goal" if planner == "plan" else "goals"
    if endpoint_key not in checkpoint:
        raise TypeError(
            f"checkpoint is missing required field {endpoint_key!r}"
        )
    # Grid constraints, revalidated in exactly ``plan``'s order with the
    # same exception types.
    width = _validate_dimension(checkpoint["width"], "width")
    height = _validate_dimension(checkpoint["height"], "height")
    start = _normalize_point(checkpoint["start"], "start")
    if planner == "plan":
        goal = _normalize_point(checkpoint["goal"], "goal")
        goals = (goal,)
    else:
        goal = None
        goals = _normalize_goals(checkpoint["goals"])
    _check_bounds(start, width, height, "start")
    for point in goals:
        _check_bounds(point, width, height, "goal")
    obstacles = _normalize_blocked(checkpoint["blocked"], width, height)
    if start in obstacles:
        raise ValueError(f"start {start} lies on a blocked cell")
    for point in goals:
        if point in obstacles:
            raise ValueError(f"goal {point} lies on a blocked cell")
    costs = _normalize_costs(checkpoint["costs"], width, height)
    frames = _normalize_dynamic_blocked(
        checkpoint["dynamic_blocked"], width, height
    )
    if frames is not None and start in frames[0]:
        raise ValueError(f"start {start} is blocked at frame 0")
    closed_count = checkpoint["closed"]
    if not _is_int(closed_count):
        raise TypeError(
            f"checkpoint closed must be an int, "
            f"got {type(closed_count).__name__}"
        )
    if closed_count < 0:
        raise ValueError(
            f"checkpoint closed must be a non-negative integer, "
            f"got {closed_count}"
        )
    # The cost limit is optional for backward compatibility: a checkpoint
    # written before ``max_cost`` existed simply lacks the field and is
    # treated as unlimited. A present field follows the same rules as
    # the ``plan`` argument, and a state inconsistent with the limit is
    # rejected when the state itself is restored below.
    max_cost = checkpoint.get("max_cost")
    _validate_max_cost(max_cost)
    cost_pruned = checkpoint.get("cost_pruned", False)
    if not isinstance(cost_pruned, bool):
        raise TypeError(
            f"checkpoint cost_pruned must be a bool, "
            f"got {type(cost_pruned).__name__}"
        )
    if cost_pruned and max_cost is None:
        raise ValueError(
            "checkpoint records cost pruning without a max_cost"
        )
    dynamic = frames is not None
    # Only the static single-goal search closes each cell at most once;
    # route-tree searches may record the same cell through different
    # histories.
    unique_trace = planner == "plan" and not dynamic
    trace = _restore_trace(
        checkpoint["trace"], dynamic, unique_trace, width, height,
        obstacles, frames
    )
    if len(trace) != closed_count:
        raise ValueError(
            "checkpoint closed count does not match its trace"
        )
    if trace:
        if trace[0][:2] != start:
            raise ValueError("checkpoint trace does not begin at start")
        if dynamic and trace[0][2] != 0:
            raise ValueError("checkpoint trace does not begin at frame 0")
    if planner == "plan" and not dynamic:
        state = _restore_static_state(
            checkpoint["state"], width, height, obstacles, start, goal,
            trace, max_cost
        )
    else:
        state = _restore_tree_state(
            checkpoint["state"], planner, dynamic, width, height,
            obstacles, frames, start, goal, goals, costs, closed_count,
            trace, max_cost
        )
    state["cost_pruned"] = cost_pruned
    return {
        "planner": planner,
        "width": width,
        "height": height,
        "obstacles": obstacles,
        "start": start,
        "goal": goal,
        "goals": goals,
        "costs": costs,
        "frames": frames,
        "max_cost": max_cost,
        "state": state,
    }


def plan(width, height, blocked, start, goal, costs=None, trace=False,
         dynamic_blocked=None, max_expanded=None, snapshot=False,
         max_cost=None):
    # --- Validation: everything is checked before the search begins. ---
    width = _validate_dimension(width, "width")
    height = _validate_dimension(height, "height")
    start = _normalize_point(start, "start")
    goal = _normalize_point(goal, "goal")
    _check_bounds(start, width, height, "start")
    _check_bounds(goal, width, height, "goal")
    obstacles = _normalize_blocked(blocked, width, height)
    if start in obstacles:
        raise ValueError(f"start {start} lies on a blocked cell")
    if goal in obstacles:
        raise ValueError(f"goal {goal} lies on a blocked cell")
    costs = _normalize_costs(costs, width, height)
    if not isinstance(trace, bool):
        raise TypeError(
            f"trace must be a bool, got {type(trace).__name__}"
        )
    frames = _normalize_dynamic_blocked(dynamic_blocked, width, height)
    if frames is not None and start in frames[0]:
        raise ValueError(
            f"start {start} is blocked at frame 0"
        )
    # The budget is validated after every pre-existing check, the snapshot
    # flag follows it, and the cost limit is validated last of all,
    # before the search starts.
    _validate_budget(max_expanded)
    _validate_snapshot_flag(snapshot)
    _validate_max_cost(max_cost)
    if frames is not None:
        return _search_dynamic(
            width, height, obstacles, frames, start, goal, costs, trace,
            max_expanded, snapshot, max_cost=max_cost
        )
    return _search_static(
        width, height, obstacles, start, goal, costs, trace, max_expanded,
        snapshot, max_cost=max_cost
    )


def plan_any(width, height, blocked, start, goals, costs=None, trace=False,
             dynamic_blocked=None, max_expanded=None, snapshot=False,
             max_cost=None):
    # --- Validation: ``goals`` takes ``goal``'s exact position in       ---
    # --- ``plan``'s validation sequence; every shared check keeps its   ---
    # --- order, exception type and message boundary.                    ---
    width = _validate_dimension(width, "width")
    height = _validate_dimension(height, "height")
    start = _normalize_point(start, "start")
    goal_points = _normalize_goals(goals)
    _check_bounds(start, width, height, "start")
    for point in goal_points:
        _check_bounds(point, width, height, "goal")
    obstacles = _normalize_blocked(blocked, width, height)
    if start in obstacles:
        raise ValueError(f"start {start} lies on a blocked cell")
    for point in goal_points:
        if point in obstacles:
            raise ValueError(f"goal {point} lies on a blocked cell")
    costs = _normalize_costs(costs, width, height)
    if not isinstance(trace, bool):
        raise TypeError(
            f"trace must be a bool, got {type(trace).__name__}"
        )
    frames = _normalize_dynamic_blocked(dynamic_blocked, width, height)
    if frames is not None and start in frames[0]:
        raise ValueError(
            f"start {start} is blocked at frame 0"
        )
    # The budget stays the last pre-existing check before the search,
    # exactly as in ``plan``; the snapshot flag and the cost limit
    # follow it in the same order as ``plan``.
    _validate_budget(max_expanded)
    _validate_snapshot_flag(snapshot)
    _validate_max_cost(max_cost)
    return _search_any(
        width, height, obstacles, frames, start, goal_points, costs, trace,
        max_expanded, snapshot, max_cost=max_cost
    )


def replay(width, height, blocked, start, goal, path, costs=None,
           dynamic_blocked=None, diagnose=False):
    # --- Validation: identical to ``plan`` and fully completed before ---
    # --- the candidate path is inspected or judged in any way.        ---
    width = _validate_dimension(width, "width")
    height = _validate_dimension(height, "height")
    start = _normalize_point(start, "start")
    goal = _normalize_point(goal, "goal")
    _check_bounds(start, width, height, "start")
    _check_bounds(goal, width, height, "goal")
    obstacles = _normalize_blocked(blocked, width, height)
    if start in obstacles:
        raise ValueError(f"start {start} lies on a blocked cell")
    if goal in obstacles:
        raise ValueError(f"goal {goal} lies on a blocked cell")
    costs = _normalize_costs(costs, width, height)
    frames = _normalize_dynamic_blocked(dynamic_blocked, width, height)
    if frames is not None and start in frames[0]:
        raise ValueError(f"start {start} is blocked at frame 0")
    if not isinstance(diagnose, bool):
        raise TypeError(
            f"diagnose must be a bool, got {type(diagnose).__name__}"
        )
    points = _normalize_path(path, width, height)

    def step_cost(point):
        # Cost of entering ``point``; the start cell is never entered.
        if costs is None:
            return 1
        return costs[point[1]][point[0]]

    def invalid(error, error_index):
        # The fixed invalid structure, with the diagnostic fields only
        # present when diagnostics were requested.
        result = {"valid": False, "cost": None, "steps": None}
        if diagnose:
            result["error"] = error
            result["error_index"] = error_index
        return result

    # --- Semantic checks: failures return the fixed invalid structure; ---
    # --- no exception and no partial cost/steps are reported.         ---
    if points[0] != start:
        return invalid("start_mismatch", 0)
    if points[-1] != goal:
        return invalid("goal_mismatch", len(points) - 1)
    if start == goal:
        # The only admissible route is the single-point route.
        if len(points) != 1:
            return invalid("start_goal_extra", 1)
        result = {"valid": True, "cost": 0, "steps": 0}
        if diagnose:
            result["error"] = None
            result["error_index"] = None
        return result

    last_frame = len(frames) - 1 if frames is not None else None
    seen = set()
    total = 0
    previous = None
    for t, point in enumerate(points):
        if point in seen:
            return invalid("repeated_coordinate", t)  # second occurrence
        if previous is not None:
            if (abs(point[0] - previous[0])
                    + abs(point[1] - previous[1])) != 1:
                return invalid("non_adjacent", t)  # later point of the pair
            total += step_cost(point)
        if point in obstacles:
            return invalid("static_blocked", t)  # static obstacle
        if frames is not None:
            frame = frames[t] if t <= last_frame else frames[last_frame]
            if point in frame:
                return invalid("dynamic_blocked", t)  # blocked at frame t
        seen.add(point)
        previous = point
    result = {"valid": True, "cost": total, "steps": len(points) - 1}
    if diagnose:
        result["error"] = None
        result["error_index"] = None
    return result


def resume(checkpoint, max_expanded=None):
    # --- Validation: the snapshot is fully checked and normalized     ---
    # --- before any searching; the budget keeps ``plan``'s rules and  ---
    # --- is validated last.                                           ---
    restored = _restore_checkpoint(checkpoint)
    _validate_budget(max_expanded)
    # A resume always continues with the trace recorded (the checkpoint
    # carries it) and always snapshots again if the budget stops the
    # search once more.
    if restored["planner"] == "plan" and restored["frames"] is None:
        return _search_static(
            restored["width"], restored["height"], restored["obstacles"],
            restored["start"], restored["goal"], restored["costs"],
            True, max_expanded, snapshot=True, state=restored["state"],
            max_cost=restored["max_cost"]
        )
    if restored["planner"] == "plan":
        return _search_dynamic(
            restored["width"], restored["height"], restored["obstacles"],
            restored["frames"], restored["start"], restored["goal"],
            restored["costs"], True, max_expanded, snapshot=True,
            state=restored["state"], max_cost=restored["max_cost"]
        )
    return _search_any(
        restored["width"], restored["height"], restored["obstacles"],
        restored["frames"], restored["start"], restored["goals"],
        restored["costs"], True, max_expanded, snapshot=True,
        state=restored["state"], max_cost=restored["max_cost"]
    )
