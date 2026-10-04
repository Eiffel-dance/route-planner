"""Deterministic 2D grid A* planner.

Public entry points: ``plan(width, height, blocked, start, goal, costs=None,
trace=False, dynamic_blocked=None, max_expanded=None, snapshot=False)``,
``plan_any(width, height, blocked, start, goals, costs=None, trace=False,
dynamic_blocked=None, max_expanded=None, snapshot=False)``,
``resume(checkpoint, max_expanded=None)`` and
``replay(width, height, blocked, start, goal, path, costs=None,
dynamic_blocked=None, diagnose=False)``.

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

Pause/resume records (``snapshot`` and ``resume``):
- ``snapshot`` is a bool option of ``plan`` and ``plan_any`` defaulting to
  ``False``; omitting it leaves every key, value, exception type and
  validation order untouched. A non-bool value raises ``TypeError`` after
  the frame-0 check and immediately before the ``max_expanded`` check, so
  the budget stays the final validation before the search.
- When ``snapshot`` is true and a budgeted search stops as
  ``"budget_exhausted"``, the result additionally carries ``checkpoint``:
  a dict holding JSON-serializable values only (str/int/bool/None/list
  values; tuples and sets never appear). It records the planner kind, the
  normalized grid constraints (dimensions, static obstacles, start, the
  goal or the sorted candidate goals, costs and dynamic-frame semantics),
  the closed-node count, the trace (``null`` when ``trace`` was false) and
  the complete pending candidate state. Every collection is emitted in a
  fixed canonical order, so the checkpoint never depends on set, goal-list
  or in-frame iteration order. No ``checkpoint`` is produced for
  ``"found"``/``"unreachable"`` or when ``snapshot`` is false.
- ``resume(checkpoint, max_expanded=None)`` continues the same
  deterministic search from a saved checkpoint; the caller may persist the
  dict as JSON and pass it back unchanged. ``max_expanded`` keeps the
  total-from-start meaning (nodes already closed count toward it) and the
  same non-negative non-bool integer rules; when it is not greater than the
  already closed count no further node is closed. Omitting it runs the
  search to completion.
- The continued result is equivalent to one un-paused call: the same
  ``path``, ``cost``, ``expanded``, ``status`` (when a budget applies) and
  ``expanded_nodes`` (per the recorded trace preference); static traces
  record coordinate pairs and dynamic traces ``(x, y, t)`` triples.
  Repeated resumptions never count an already closed node twice. Stopping
  at the limit again returns a fresh ``checkpoint``; ``found`` and
  ``unreachable`` results omit it. A resumed path verifies in ``replay``
  with the same cost and step count.
- All ``resume`` validation runs before the search: a checkpoint that is
  not an object, fields of the wrong type, or missing required fields raise
  ``TypeError``; an unknown snapshot version or planner kind or any
  inconsistent grid, obstacle, endpoint, cost, frame or search-state field
  raises ``ValueError``.

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
  for an invalid one a single error code plus the zero-based index of the
  first offending element, with ``cost``/``steps`` still ``None``.
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

# Snapshot format version emitted by this build and accepted by ``resume``.
_SNAPSHOT_VERSION = 1
_PLANNER_PLAN = "plan"
_PLANNER_ANY = "plan_any"


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


# ---------------------------------------------------------------------------
# Search backends
#
# Each backend returns ``(result, state)``: ``result`` is the public result
# dict and ``state`` is the raw continuation state (tuples/sets allowed)
# when the search stopped with live candidates remaining because of the
# budget, and ``None`` otherwise. ``state`` contains only in-memory
# structures; ``_build_checkpoint`` turns it into the canonical
# JSON-serializable record and ``_restore_state`` rebuilds it on resume.
# ---------------------------------------------------------------------------


def _enter_cost(costs, point):
    # Cost of entering ``point``; the start cell is never entered.
    if costs is None:
        return 1
    return costs[point[1]][point[0]]


def _search_static(width, height, obstacles, start, goal, costs, trace,
                   max_expanded, initial=None):
    """Classical closed-set A* used by static ``plan``."""

    def heuristic(point):
        return abs(point[0] - goal[0]) + abs(point[1] - goal[1])

    if initial is None:
        h0 = heuristic(start)
        # Entries are (f, h, x, y, g_push, point): ``g_push`` is the
        # candidate g at push time (a stale entry keeps an older, larger
        # value) and is fully determined by f/h/x/y, so it never changes
        # the ordering; it is carried so a checkpoint can verify every
        # pending entry exactly.
        open_heap = [(h0, h0, start[0], start[1], 0, start)]
        came_from = {}
        g_score = {start: 0}
        closed = set()
        expanded_nodes = [] if trace else None
    else:
        open_heap = initial["heap"]
        closed = initial["closed"]
        g_score = initial["g_score"]
        came_from = initial["came_from"]
        expanded_nodes = initial["expanded_nodes"]

    budget_stop = False
    while open_heap:
        if max_expanded is not None and len(closed) >= max_expanded:
            # The budget is spent. A normal iteration would keep popping
            # roots that belong to already-closed nodes (its stale-entry
            # ``continue``) until it either popped a live one (budget
            # exhausted) or emptied the heap (unreachable). Do exactly
            # that discrimination here without touching a live root, so
            # the pending frontier stays complete for a checkpoint; this
            # also covers start == goal with a zero budget, where no node
            # is closed and no start is fabricated.
            while open_heap and open_heap[0][5] in closed:
                heapq.heappop(open_heap)
            if open_heap:
                budget_stop = True
            break
        _, _, _, _, _, current = heapq.heappop(open_heap)
        if current in closed:
            continue  # stale heap entry; already closed with its best g
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
            if max_expanded is not None:
                result["status"] = "found"
            if trace:
                result["expanded_nodes"] = expanded_nodes
            return result, None
        for dx, dy in _NEIGHBORS:
            nxt = (current[0] + dx, current[1] + dy)
            if not (0 <= nxt[0] < width and 0 <= nxt[1] < height):
                continue
            if nxt in obstacles:
                continue
            new_g = g_score[current] + _enter_cost(costs, nxt)
            if new_g < g_score.get(nxt, float("inf")):
                g_score[nxt] = new_g
                came_from[nxt] = current
                h = heuristic(nxt)
                heapq.heappush(
                    open_heap, (new_g + h, h, nxt[0], nxt[1], new_g, nxt)
                )
    result = {"path": None, "cost": None, "expanded": len(closed)}
    if max_expanded is not None:
        result["status"] = (
            "budget_exhausted" if budget_stop else "unreachable"
        )
    if trace:
        result["expanded_nodes"] = expanded_nodes
    state = None
    if budget_stop:
        state = {
            "heap": open_heap,
            "closed": closed,
            "g_score": g_score,
            "came_from": came_from,
            "expanded": len(closed),
            "expanded_nodes": expanded_nodes,
        }
    return result, state


def _search_dynamic(width, height, obstacles, frames, start, goal, costs,
                    trace, max_expanded, initial=None):
    """History-sensitive time-expanded A* used by dynamic ``plan``.

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
    lexicographic order.
    """

    def heuristic(point):
        return abs(point[0] - goal[0]) + abs(point[1] - goal[1])

    last_frame = len(frames) - 1

    def frame_cells(t):
        return frames[t] if t <= last_frame else frames[last_frame]

    if initial is None:
        h0 = heuristic(start)
        # Heap entries are (f, h, x, y, t, g, path, seen): f/h/x/y/t give
        # the fixed numeric priority and candidates still tied are ordered
        # by the complete coordinate path lexicographically. The frozenset
        # ``seen`` mirrors ``path`` for an O(1) repeat check and is never
        # compared (distinct histories always differ in ``path``).
        start_path = (start,)
        open_heap = [(h0, h0, start[0], start[1], 0, 0, start_path,
                      frozenset(start_path))]
        closed_count = 0
        expanded_nodes = [] if trace else None
    else:
        open_heap = initial["heap"]
        closed_count = initial["expanded"]
        expanded_nodes = initial["expanded_nodes"]

    # Best goal closing seen so far, keyed exactly as the tie-break
    # specializes at the goal: (g, t, path) (there f == g, h == 0 and
    # (x, y) is fixed). Closing a goal does not stop the search
    # immediately: an equally cheap route whose chain runs through
    # higher-h nodes might still be undeveloped. Since each move costs at
    # least 1 and the Manhattan h changes by at most 1 per move, f = g + h
    # never decreases along a route; once the smallest f on the heap
    # exceeds the best goal cost, no route of that cost (or less) can
    # remain undiscovered.
    best_key = None
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
            new_g = g + _enter_cost(costs, nxt)
            new_path = path + (nxt,)
            h = heuristic(nxt)
            heapq.heappush(
                open_heap,
                (new_g + h, h, nx, ny, next_t, new_g, new_path,
                 seen | {nxt}),
            )
    if best_key is None:
        result = {"path": None, "cost": None, "expanded": closed_count}
        if max_expanded is not None:
            result["status"] = (
                "budget_exhausted" if budget_stop else "unreachable"
            )
    else:
        best_g, _, best_path = best_key
        result = {
            "path": list(best_path),
            "cost": best_g,
            "expanded": closed_count,
        }
        if max_expanded is not None:
            result["status"] = "found"
    if trace:
        result["expanded_nodes"] = expanded_nodes
    state = None
    # A checkpoint exists only for a genuine budget stop with no goal
    # closed yet: when ``best_key`` is set the result is ``found``.
    if budget_stop and best_key is None:
        state = {
            "heap": open_heap,
            "expanded": closed_count,
            "expanded_nodes": expanded_nodes,
        }
    return result, state


def _search_any(width, height, obstacles, frames, start, goals, costs,
                trace, max_expanded, initial=None):
    """History-sensitive route-tree A* used by ``plan_any``.

    The only semantic differences from ``_search_dynamic`` are: the
    heuristic is the minimum Manhattan distance to any candidate endpoint,
    and a closing at any goal coordinate terminates the route. As in
    ``_search_dynamic`` the feasibility of a route depends on its exact
    coordinate history (no waiting, no repeated coordinates, and dynamic
    frames), so every node carries its complete path and routes reaching
    the same cell through different histories are never merged; this is
    also what makes the complete-path lexicographic tie-break exact.
    Static mode is the same traversal with the time dimension fixed at 0
    and no timed frames, so the trace records plain coordinate pairs
    there, exactly as ``plan`` does.
    """

    def heuristic(point):
        return min(
            abs(point[0] - goal[0]) + abs(point[1] - goal[1])
            for goal in goals
        )

    dynamic = frames is not None
    last_frame = len(frames) - 1 if dynamic else None

    def frame_cells(t):
        return frames[t] if t <= last_frame else frames[last_frame]

    if initial is None:
        h0 = heuristic(start)
        start_path = (start,)
        open_heap = [(h0, h0, start[0], start[1], 0, 0, start_path,
                      frozenset(start_path))]
        closed_count = 0
        expanded_nodes = [] if trace else None
    else:
        open_heap = initial["heap"]
        closed_count = initial["expanded"]
        expanded_nodes = initial["expanded_nodes"]

    # Best goal closing seen so far, keyed as (g, (x, y), path): minimum
    # total cost first, then the endpoint coordinate's lexicographic order,
    # then the complete route's lexicographic order. Closing a goal does
    # not stop the search immediately: an equally cheap route to a smaller
    # goal whose chain runs through higher-h nodes might still be
    # undeveloped. Every move costs at least 1 and the minimum Manhattan
    # heuristic changes by at most 1 per move, so f = g + h never
    # decreases along a route; once the smallest f on the heap exceeds the
    # best goal cost, no route of that cost (or less) can remain
    # undiscovered.
    best_key = None
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
            new_g = g + _enter_cost(costs, nxt)
            new_path = path + (nxt,)
            h = heuristic(nxt)
            heapq.heappush(
                open_heap,
                (new_g + h, h, nx, ny, next_t, new_g, new_path,
                 seen | {nxt}),
            )
    if best_key is None:
        result = {"path": None, "cost": None, "expanded": closed_count}
        if max_expanded is not None:
            result["status"] = (
                "budget_exhausted" if budget_stop else "unreachable"
            )
    else:
        best_g, _, best_path = best_key
        result = {
            "path": list(best_path),
            "cost": best_g,
            "expanded": closed_count,
        }
        if max_expanded is not None:
            result["status"] = "found"
    if trace:
        result["expanded_nodes"] = expanded_nodes
    state = None
    if budget_stop and best_key is None:
        state = {
            "heap": open_heap,
            "expanded": closed_count,
            "expanded_nodes": expanded_nodes,
        }
    return result, state


# ---------------------------------------------------------------------------
# Checkpoint serialization
# ---------------------------------------------------------------------------


def _build_checkpoint(kind, width, height, obstacles, start, endpoint,
                      costs, frames, trace, state):
    """Turn a raw continuation state into a canonical JSON-only dict.

    ``endpoint`` is the single goal tuple for ``plan`` or the sorted
    candidate-goal tuple for ``plan_any``. Every collection is emitted in a
    fixed order, so the bytes are independent of set, goal-list or
    in-frame iteration order.
    """
    dynamic = frames is not None
    # Static ``plan`` is the closed-set A*; every other traversal is a
    # route-tree search whose heap entries carry full paths.
    grid_mode = kind == _PLANNER_PLAN and not dynamic

    cp = {}
    cp["version"] = _SNAPSHOT_VERSION
    cp["kind"] = kind
    cp["dynamic"] = dynamic
    cp["trace"] = bool(trace)
    cp["width"] = width
    cp["height"] = height
    cp["blocked"] = [[x, y] for x, y in sorted(obstacles)]
    cp["start"] = [start[0], start[1]]
    if kind == _PLANNER_PLAN:
        cp["goal"] = [endpoint[0], endpoint[1]]
    else:
        cp["goals"] = [[x, y] for x, y in sorted(endpoint)]
    cp["costs"] = (
        None if costs is None
        else [[cell for cell in row] for row in costs]
    )
    cp["frames"] = (
        None if frames is None
        else [[[x, y] for x, y in sorted(frame)] for frame in frames]
    )
    cp["expanded"] = state["expanded"]
    nodes = state["expanded_nodes"]
    cp["expanded_nodes"] = (
        None if nodes is None else [[coord for coord in node]
                                    for node in nodes]
    )
    if grid_mode:
        # Canonical heap layout independent of insertion history: sorting
        # the entries and heapifying yields one fixed array for the same
        # pending multiset, so a checkpoint after chunked resumes is
        # byte-identical to one from a single budgeted call at that point.
        pending = sorted(state["heap"])
        cp["open"] = [
            [f, h, x, y, g_push, [point[0], point[1]]]
            for f, h, x, y, g_push, point in pending
        ]
        cp["closed"] = [[x, y] for x, y in sorted(state["closed"])]
        g_score = state["g_score"]
        cp["g_score"] = [
            [[x, y], g_score[(x, y)]] for x, y in sorted(g_score)
        ]
        came_from = state["came_from"]
        cp["came_from"] = [
            [[x, y], [parent[0], parent[1]]]
            for (x, y), parent in sorted(came_from.items())
        ]
    else:
        # Canonical heap layout (see the static branch above). Distinct
        # entries always differ in their complete ``path``, so the trailing
        # frozenset ``seen`` never participates in ordering.
        pending = sorted(state["heap"])
        cp["open"] = [
            [f, h, x, y, t, g,
             [[px, py] for px, py in path],
             [[px, py] for px, py in sorted(seen)]]
            for f, h, x, y, t, g, path, seen in pending
        ]
    return cp


def _parse_checkpoint(checkpoint):
    """Validate and decode a checkpoint before any search runs.

    Structural/type problems raise ``TypeError``; unknown versions/kinds and
    inconsistent values raise ``ValueError``. Returns the normalized problem
    description plus the decoded ``initial`` search state.
    """
    if not isinstance(checkpoint, dict):
        raise TypeError(
            "checkpoint must be an object, got "
            f"{type(checkpoint).__name__}"
        )
    data = checkpoint

    def field(key):
        if key not in data:
            raise TypeError(
                f"checkpoint is missing required field {key!r}"
            )
        return data[key]

    def as_int(value, what):
        if not _is_int(value):
            raise TypeError(
                f"checkpoint field {what} must be an int, got "
                f"{type(value).__name__}"
            )
        return value

    def as_bool(value, what):
        if not isinstance(value, bool):
            raise TypeError(
                f"checkpoint field {what} must be a bool, got "
                f"{type(value).__name__}"
            )
        return value

    def as_point(value, what):
        if not isinstance(value, list):
            raise TypeError(
                f"checkpoint field {what} must be a [x, y] list, got "
                f"{type(value).__name__}"
            )
        if len(value) != 2:
            raise TypeError(
                f"checkpoint field {what} must have exactly two "
                f"coordinates, got {len(value)}"
            )
        x, y = value
        if not _is_int(x) or not _is_int(y):
            raise TypeError(
                f"checkpoint field {what} coordinates must be ints, "
                f"got {value!r}"
            )
        return (x, y)

    def bounded(point, what):
        if not (0 <= point[0] < width and 0 <= point[1] < height):
            raise ValueError(
                f"checkpoint field {what} {point} is outside the "
                f"{width}x{height} grid"
            )
        return point

    version = as_int(field("version"), "version")
    if version != _SNAPSHOT_VERSION:
        raise ValueError(
            f"unsupported checkpoint version {version}; expected "
            f"{_SNAPSHOT_VERSION}"
        )
    kind = field("kind")
    if not isinstance(kind, str):
        raise TypeError(
            "checkpoint field 'kind' must be a string, got "
            f"{type(kind).__name__}"
        )
    if kind not in (_PLANNER_PLAN, _PLANNER_ANY):
        raise ValueError(f"unknown checkpoint planner kind {kind!r}")
    dynamic = as_bool(field("dynamic"), "dynamic")
    trace = as_bool(field("trace"), "trace")
    width = as_int(field("width"), "width")
    height = as_int(field("height"), "height")
    if width <= 0 or height <= 0:
        raise ValueError(
            "checkpoint grid dimensions must be positive integers, got "
            f"{width}x{height}"
        )

    raw_blocked = field("blocked")
    if not isinstance(raw_blocked, list):
        raise TypeError(
            "checkpoint field 'blocked' must be a list, got "
            f"{type(raw_blocked).__name__}"
        )
    obstacles = set()
    for index, item in enumerate(raw_blocked):
        point = bounded(as_point(item, f"blocked[{index}]"),
                        f"blocked[{index}]")
        obstacles.add(point)

    start = bounded(as_point(field("start"), "start"), "start")
    if start in obstacles:
        raise ValueError(f"checkpoint start {start} lies on a blocked cell")

    goal = None
    goals = None
    if kind == _PLANNER_PLAN:
        goal = bounded(as_point(field("goal"), "goal"), "goal")
        if goal in obstacles:
            raise ValueError(f"checkpoint goal {goal} lies on a blocked cell")
    else:
        raw_goals = field("goals")
        if not isinstance(raw_goals, list):
            raise TypeError(
                "checkpoint field 'goals' must be a list, got "
                f"{type(raw_goals).__name__}"
            )
        if len(raw_goals) == 0:
            raise ValueError(
                "checkpoint field 'goals' must contain at least one "
                "candidate endpoint"
            )
        goal_set = set()
        for index, item in enumerate(raw_goals):
            point = bounded(as_point(item, f"goals[{index}]"),
                            f"goals[{index}]")
            if point in obstacles:
                raise ValueError(
                    f"checkpoint goal {point} lies on a blocked cell"
                )
            goal_set.add(point)
        goals = tuple(sorted(goal_set))

    raw_costs = field("costs")
    if raw_costs is None:
        costs = None
    else:
        if not isinstance(raw_costs, list):
            raise TypeError(
                "checkpoint field 'costs' must be a list or null, got "
                f"{type(raw_costs).__name__}"
            )
        if len(raw_costs) != height:
            raise ValueError(
                f"checkpoint costs must have exactly {height} rows, got "
                f"{len(raw_costs)}"
            )
        rows = []
        for y, row in enumerate(raw_costs):
            if not isinstance(row, list):
                raise TypeError(
                    f"checkpoint costs[{y}] must be a list, got "
                    f"{type(row).__name__}"
                )
            if len(row) != width:
                raise ValueError(
                    f"checkpoint costs[{y}] must have exactly {width} "
                    f"cells, got {len(row)}"
                )
            parsed_row = []
            for x, cell in enumerate(row):
                if not _is_int(cell):
                    raise TypeError(
                        f"checkpoint costs[{y}][{x}] must be an int, got "
                        f"{type(cell).__name__}"
                    )
                if cell <= 0:
                    raise ValueError(
                        f"checkpoint costs[{y}][{x}] must be a positive "
                        f"integer, got {cell}"
                    )
                parsed_row.append(cell)
            rows.append(tuple(parsed_row))
        costs = tuple(rows)

    raw_frames = field("frames")
    frames = None
    if raw_frames is not None:
        if not isinstance(raw_frames, list):
            raise TypeError(
                "checkpoint field 'frames' must be a list or null, got "
                f"{type(raw_frames).__name__}"
            )
    if dynamic:
        if raw_frames is None:
            raise ValueError(
                "checkpoint 'dynamic' is true but 'frames' is null"
            )
        if len(raw_frames) == 0:
            raise ValueError(
                "a dynamic checkpoint must contain at least one frame"
            )
        frames = []
        for t, frame in enumerate(raw_frames):
            if not isinstance(frame, list):
                raise TypeError(
                    f"checkpoint frames[{t}] must be a list, got "
                    f"{type(frame).__name__}"
                )
            cells = set()
            for index, item in enumerate(frame):
                point = bounded(as_point(item, f"frames[{t}][{index}]"),
                                f"frames[{t}][{index}]")
                cells.add(point)
            frames.append(frozenset(cells))
    elif raw_frames is not None:
        raise ValueError(
            "checkpoint 'dynamic' is false but 'frames' is not null"
        )
    if frames is not None and start in frames[0]:
        raise ValueError("checkpoint start is blocked at frame 0")

    expanded = as_int(field("expanded"), "expanded")
    if expanded < 0:
        raise ValueError(
            f"checkpoint field 'expanded' must be non-negative, got "
            f"{expanded}"
        )

    # ----- Trace -----------------------------------------------------------
    raw_nodes = field("expanded_nodes")
    if raw_nodes is not None and not isinstance(raw_nodes, list):
        raise TypeError(
            "checkpoint field 'expanded_nodes' must be a list or null, got "
            f"{type(raw_nodes).__name__}"
        )
    expanded_nodes = None
    if trace:
        if raw_nodes is None:
            raise ValueError(
                "checkpoint 'trace' is true but 'expanded_nodes' is null"
            )
        arity = 3 if dynamic else 2
        nodes = []
        for index, item in enumerate(raw_nodes):
            if not isinstance(item, list) or len(item) != arity:
                raise TypeError(
                    f"checkpoint expanded_nodes[{index}] must be a list of "
                    f"{arity} ints"
                )
            if not all(_is_int(coord) for coord in item):
                raise TypeError(
                    f"checkpoint expanded_nodes[{index}] coordinates must "
                    "be ints"
                )
            node = tuple(item)
            x, y = node[0], node[1]
            if not (0 <= x < width and 0 <= y < height):
                raise ValueError(
                    f"checkpoint expanded_nodes[{index}] {node} is outside "
                    "the grid"
                )
            if (x, y) in obstacles:
                raise ValueError(
                    f"checkpoint expanded_nodes[{index}] {node} is a static "
                    "obstacle"
                )
            if dynamic:
                t = node[2]
                if t < 0:
                    raise ValueError(
                        f"checkpoint expanded_nodes[{index}] has a negative "
                        f"time {t}"
                    )
                last_frame = len(frames) - 1
                if (x, y) in (frames[t] if t <= last_frame
                              else frames[last_frame]):
                    raise ValueError(
                        f"checkpoint expanded_nodes[{index}] {node} is "
                        "blocked at its frame"
                    )
            nodes.append(node)
        if len(nodes) != expanded:
            raise ValueError(
                f"checkpoint expanded_nodes length {len(nodes)} does not "
                f"match expanded {expanded}"
            )
        expanded_nodes = nodes
    elif raw_nodes is not None:
        raise ValueError(
            "checkpoint 'trace' is false but 'expanded_nodes' is not null"
        )

    def enter_cost(point):
        return _enter_cost(costs, point)

    def heuristic_to(point):
        if kind == _PLANNER_PLAN:
            return abs(point[0] - goal[0]) + abs(point[1] - goal[1])
        return min(
            abs(point[0] - candidate[0]) + abs(point[1] - candidate[1])
            for candidate in goals
        )

    def check_route_path(raw_path, what):
        if not isinstance(raw_path, list) or len(raw_path) == 0:
            raise TypeError(
                f"checkpoint field {what} must be a non-empty list"
            )
        route = []
        for index, item in enumerate(raw_path):
            point = bounded(as_point(item, f"{what}[{index}]"),
                            f"{what}[{index}]")
            if point in obstacles:
                raise ValueError(
                    f"checkpoint {what}[{index}] {point} is a static "
                    "obstacle"
                )
            if index == 0:
                if point != start:
                    raise ValueError(
                        f"checkpoint {what} must begin at start {start}, "
                        f"got {point}"
                    )
            else:
                previous = route[-1]
                if (abs(point[0] - previous[0])
                        + abs(point[1] - previous[1])) != 1:
                    raise ValueError(
                        f"checkpoint {what}[{index}] is not four-neighbor "
                        "adjacent"
                    )
                if point in route:
                    raise ValueError(
                        f"checkpoint {what}[{index}] repeats a coordinate"
                    )
            if dynamic:
                t = index
                last_frame = len(frames) - 1
                frame = frames[t] if t <= last_frame else frames[last_frame]
                if point in frame:
                    raise ValueError(
                        f"checkpoint {what}[{index}] {point} is blocked at "
                        f"frame {t}"
                    )
            route.append(point)
        return tuple(route)

    # ----- Pending candidates ---------------------------------------------
    raw_open = field("open")
    if not isinstance(raw_open, list):
        raise TypeError(
            "checkpoint field 'open' must be a list, got "
            f"{type(raw_open).__name__}"
        )
    if len(raw_open) == 0:
        raise ValueError(
            "checkpoint 'open' is empty but the search was reported as "
            "budget_exhausted"
        )

    grid_mode = kind == _PLANNER_PLAN and not dynamic
    heap = []

    if grid_mode:
        closed = set()
        raw_closed = field("closed")
        if not isinstance(raw_closed, list):
            raise TypeError(
                "checkpoint field 'closed' must be a list, got "
                f"{type(raw_closed).__name__}"
            )
        for index, item in enumerate(raw_closed):
            point = bounded(as_point(item, f"closed[{index}]"),
                            f"closed[{index}]")
            if point in obstacles:
                raise ValueError(
                    f"checkpoint closed[{index}] {point} is a static "
                    "obstacle"
                )
            if point in closed:
                raise ValueError(
                    f"checkpoint closed[{index}] {point} is listed twice"
                )
            closed.add(point)
        if len(closed) != expanded:
            raise ValueError(
                f"checkpoint closed count {len(closed)} does not match "
                f"expanded {expanded}"
            )
        if trace and {node for node in expanded_nodes} != closed:
            raise ValueError(
                "checkpoint expanded_nodes does not match the closed set"
            )

        raw_g = field("g_score")
        if not isinstance(raw_g, list):
            raise TypeError(
                "checkpoint field 'g_score' must be a list, got "
                f"{type(raw_g).__name__}"
            )
        g_score = {}
        for index, pair in enumerate(raw_g):
            if not isinstance(pair, list) or len(pair) != 2:
                raise TypeError(
                    f"checkpoint g_score[{index}] must be a [point, g] pair"
                )
            point = bounded(as_point(pair[0], f"g_score[{index}]"),
                            f"g_score[{index}]")
            gv = as_int(pair[1], f"g_score[{index}] g")
            if gv < 0:
                raise ValueError(
                    f"checkpoint g_score[{index}] g must be non-negative"
                )
            if point in obstacles:
                raise ValueError(
                    f"checkpoint g_score[{index}] {point} is blocked"
                )
            if point in g_score:
                raise ValueError(
                    f"checkpoint g_score[{index}] {point} is listed twice"
                )
            g_score[point] = gv
        if g_score.get(start) != 0:
            raise ValueError(
                "checkpoint g_score must map the start to a cost of 0"
            )
        if not closed <= set(g_score):
            raise ValueError(
                "checkpoint closed nodes are missing from g_score"
            )

        raw_cf = field("came_from")
        if not isinstance(raw_cf, list):
            raise TypeError(
                "checkpoint field 'came_from' must be a list, got "
                f"{type(raw_cf).__name__}"
            )
        came_from = {}
        for index, pair in enumerate(raw_cf):
            if not isinstance(pair, list) or len(pair) != 2:
                raise TypeError(
                    f"checkpoint came_from[{index}] must be a "
                    "[child, parent] pair"
                )
            child = bounded(as_point(pair[0], f"came_from[{index}]"),
                            f"came_from[{index}]")
            parent = bounded(as_point(pair[1], f"came_from[{index}]"),
                             f"came_from[{index}]")
            if child == start:
                raise ValueError(
                    "checkpoint came_from must not list the start as a child"
                )
            if child in came_from:
                raise ValueError(
                    f"checkpoint came_from[{index}] {child} is listed twice"
                )
            came_from[child] = parent
        if set(came_from) != set(g_score) - {start}:
            raise ValueError(
                "checkpoint came_from edges do not match g_score"
            )
        for child, parent in came_from.items():
            if parent not in g_score:
                raise ValueError(
                    f"checkpoint came_from parent {parent} of {child} is "
                    "missing from g_score"
                )
            if (abs(child[0] - parent[0])
                    + abs(child[1] - parent[1])) != 1:
                raise ValueError(
                    f"checkpoint came_from edge {parent} -> {child} is not "
                    "four-neighbor adjacent"
                )
            if g_score[child] != g_score[parent] + enter_cost(child):
                raise ValueError(
                    f"checkpoint g_score for {child} disagrees with its "
                    "came_from edge"
                )

        for index, entry in enumerate(raw_open):
            what = f"open[{index}]"
            if not isinstance(entry, list) or len(entry) != 6:
                raise TypeError(
                    f"checkpoint {what} must have six fields"
                )
            fv, hv, ex, ey, g_push, raw_point = entry
            fv = as_int(fv, f"{what} f")
            hv = as_int(hv, f"{what} h")
            ex = as_int(ex, f"{what} x")
            ey = as_int(ey, f"{what} y")
            g_push = as_int(g_push, f"{what} g")
            point = bounded(as_point(raw_point, what), what)
            if point in obstacles:
                raise ValueError(f"checkpoint {what} point is blocked")
            if (ex, ey) != point:
                raise ValueError(
                    f"checkpoint {what} coordinate does not match its point"
                )
            if point not in g_score:
                raise ValueError(
                    f"checkpoint {what} point is missing from g_score"
                )
            if g_push < 0:
                raise ValueError(f"checkpoint {what} g must be non-negative")
            # Every pending entry was produced by the search with this exact
            # g at push time; a stale entry carries an older, larger g than
            # the current best, so ``g_push`` must equal f - h and be at
            # least the node's current g_score.
            if g_push < g_score[point]:
                raise ValueError(
                    f"checkpoint {what} g is below the current g_score"
                )
            if hv != heuristic_to(point):
                raise ValueError(
                    f"checkpoint {what} h disagrees with the heuristic"
                )
            if fv != g_push + hv:
                raise ValueError(
                    f"checkpoint {what} f disagrees with g + h"
                )
            heap.append((fv, hv, ex, ey, g_push, point))
        # Canonicalize the heap ordering; the serialized list order must
        # never affect the resumed traversal.
        heapq.heapify(heap)
    else:
        for index, entry in enumerate(raw_open):
            what = f"open[{index}]"
            if not isinstance(entry, list) or len(entry) != 8:
                raise TypeError(
                    f"checkpoint {what} must have eight fields"
                )
            fv, hv, ex, ey, tv, gv, raw_path, raw_seen = entry
            fv = as_int(fv, f"{what} f")
            hv = as_int(hv, f"{what} h")
            ex = as_int(ex, f"{what} x")
            ey = as_int(ey, f"{what} y")
            tv = as_int(tv, f"{what} t")
            gv = as_int(gv, f"{what} g")
            if tv < 0 or gv < 0:
                raise ValueError(f"checkpoint {what} t/g must be non-negative")
            route = check_route_path(raw_path, f"{what} path")
            if not dynamic and tv != 0:
                raise ValueError(
                    f"checkpoint {what} is static but carries t={tv}"
                )
            if dynamic and tv != len(route) - 1:
                raise ValueError(
                    f"checkpoint {what} t does not match its path length"
                )
            if not isinstance(raw_seen, list):
                raise TypeError(
                    f"checkpoint {what} seen must be a list, got "
                    f"{type(raw_seen).__name__}"
                )
            seen = set()
            for s_index, item in enumerate(raw_seen):
                point = bounded(as_point(item, f"{what} seen[{s_index}]"),
                                f"{what} seen[{s_index}]")
                if point in seen:
                    raise ValueError(
                        f"checkpoint {what} seen lists {point} twice"
                    )
                seen.add(point)
            if seen != set(route):
                raise ValueError(
                    f"checkpoint {what} seen does not match its path"
                )
            last = route[-1]
            if (ex, ey) != last:
                raise ValueError(
                    f"checkpoint {what} coordinate does not match its path "
                    "end"
                )
            expected_g = sum(enter_cost(point) for point in route[1:])
            if gv != expected_g:
                raise ValueError(
                    f"checkpoint {what} g disagrees with its path cost"
                )
            if hv != heuristic_to(last):
                raise ValueError(
                    f"checkpoint {what} h disagrees with the heuristic"
                )
            if fv != gv + hv:
                raise ValueError(
                    f"checkpoint {what} f disagrees with g + h"
                )
            heap.append((fv, hv, ex, ey, tv, gv, route, frozenset(seen)))
        # Canonicalize the heap ordering; the serialized list order must
        # never affect the resumed traversal. Distinct entries always differ
        # in their complete ``path`` (tuple element 6), so the frozenset
        # ``seen`` is never compared, exactly as in the live search.
        heapq.heapify(heap)

    expected_fields = {
        "version", "kind", "dynamic", "trace", "width", "height",
        "blocked", "start", "costs", "frames", "expanded",
        "expanded_nodes", "open",
    }
    expected_fields.add("goal" if kind == _PLANNER_PLAN else "goals")
    if grid_mode:
        expected_fields.update({"closed", "g_score", "came_from"})
    unexpected = set(data) - expected_fields
    if unexpected:
        raise ValueError(
            "checkpoint contains unexpected fields: "
            + ", ".join(sorted(unexpected))
        )

    endpoint = goal if kind == _PLANNER_PLAN else goals
    initial = {
        "heap": heap,
        "expanded": expanded,
        "expanded_nodes": expanded_nodes,
    }
    if grid_mode:
        initial["closed"] = closed
        initial["g_score"] = g_score
        initial["came_from"] = came_from
    return {
        "kind": kind,
        "dynamic": dynamic,
        "trace": trace,
        "width": width,
        "height": height,
        "obstacles": obstacles,
        "start": start,
        "endpoint": endpoint,
        "costs": costs,
        "frames": frames,
        "initial": initial,
    }


def _validate_budget(max_expanded):
    if max_expanded is None:
        return None
    if not _is_int(max_expanded):
        raise TypeError(
            "max_expanded must be an int, got "
            f"{type(max_expanded).__name__}"
        )
    if max_expanded < 0:
        raise ValueError(
            f"max_expanded must be a non-negative integer, got "
            f"{max_expanded}"
        )
    return max_expanded


def plan(width, height, blocked, start, goal, costs=None, trace=False,
         dynamic_blocked=None, max_expanded=None, snapshot=False):
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
    # ``snapshot`` is checked after every pre-existing check and before the
    # budget, so the budget remains the final gate before the search.
    if not isinstance(snapshot, bool):
        raise TypeError(
            f"snapshot must be a bool, got {type(snapshot).__name__}"
        )
    _validate_budget(max_expanded)
    if frames is not None:
        result, state = _search_dynamic(
            width, height, obstacles, frames, start, goal, costs, trace,
            max_expanded
        )
    else:
        result, state = _search_static(
            width, height, obstacles, start, goal, costs, trace,
            max_expanded
        )
    if snapshot and state is not None:
        result["checkpoint"] = _build_checkpoint(
            _PLANNER_PLAN, width, height, obstacles, start, goal, costs,
            frames, trace, state
        )
    return result


def plan_any(width, height, blocked, start, goals, costs=None, trace=False,
             dynamic_blocked=None, max_expanded=None, snapshot=False):
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
    if not isinstance(snapshot, bool):
        raise TypeError(
            f"snapshot must be a bool, got {type(snapshot).__name__}"
        )
    _validate_budget(max_expanded)
    result, state = _search_any(
        width, height, obstacles, frames, start, goal_points, costs, trace,
        max_expanded
    )
    if snapshot and state is not None:
        result["checkpoint"] = _build_checkpoint(
            _PLANNER_ANY, width, height, obstacles, start, goal_points,
            costs, frames, trace, state
        )
    return result


def resume(checkpoint, max_expanded=None):
    """Continue a paused deterministic search from a saved ``checkpoint``.

    See the module docstring on "Pause/resume records". Every validation
    error is raised here, before the search runs; a resumed traversal is
    bit-for-bit equivalent to letting the original budgeted call run with
    the new total ``max_expanded``.
    """
    parsed = _parse_checkpoint(checkpoint)
    _validate_budget(max_expanded)

    kind = parsed["kind"]
    initial = parsed["initial"]
    if kind == _PLANNER_PLAN:
        if parsed["dynamic"]:
            result, state = _search_dynamic(
                parsed["width"], parsed["height"], parsed["obstacles"],
                parsed["frames"], parsed["start"], parsed["endpoint"],
                parsed["costs"], parsed["trace"], max_expanded,
                initial=initial
            )
        else:
            result, state = _search_static(
                parsed["width"], parsed["height"], parsed["obstacles"],
                parsed["start"], parsed["endpoint"], parsed["costs"],
                parsed["trace"], max_expanded, initial=initial
            )
    else:
        result, state = _search_any(
            parsed["width"], parsed["height"], parsed["obstacles"],
            parsed["frames"], parsed["start"], parsed["endpoint"],
            parsed["costs"], parsed["trace"], max_expanded, initial=initial
        )
    if state is not None:
        result["checkpoint"] = _build_checkpoint(
            kind, parsed["width"], parsed["height"], parsed["obstacles"],
            parsed["start"], parsed["endpoint"], parsed["costs"],
            parsed["frames"], parsed["trace"], state
        )
    return result


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
