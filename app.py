"""Deterministic 2D grid A* planner.

Public entry points: ``plan(width, height, blocked, start, goal, costs=None,
trace=False, dynamic_blocked=None)`` and ``replay(width, height, blocked,
start, goal, path, costs=None, dynamic_blocked=None)``.

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

Offline replay: ``replay(width, height, blocked, start, goal, path,
costs=None, dynamic_blocked=None)`` re-checks a saved candidate path
against the same rules without searching. Grid arguments are validated
exactly as in ``plan`` (same exceptions, same order) before the path is
examined. ``path`` must be a non-empty sequence of two-integer
coordinates (``TypeError`` for strings/bytes/``None``/non-sequences/bad
elements, ``ValueError`` for an empty path or out-of-bounds coordinates).
Semantically invalid paths (wrong endpoints, static or timed obstacle
hits, non-four-neighborhood moves, repeated coordinates, extra points
when ``start == goal``) return ``{"valid": False, "cost": None,
"steps": None}`` instead of raising. Valid paths return
``{"valid": True, "cost": int, "steps": int}`` with ``cost`` recomputed
by the same entering-cell rule as ``plan`` and ``steps == len(path) - 1``.
No ``expanded``/``expanded_nodes`` fields are produced.
"""

import heapq
from collections.abc import Iterable, Sequence

__all__ = ["plan", "replay"]

# Fixed neighbor generation order: +x, -x, +y, -y.
_NEIGHBORS = ((1, 0), (-1, 0), (0, 1), (0, -1))


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


def _search_dynamic(width, height, obstacles, frames, start, goal, costs,
                    trace):
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
    lexicographic order.
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

    h0 = heuristic(start)
    if start == goal:
        # Single-point, zero-cost result; frame 0 was checked upstream.
        result = {"path": [start], "cost": 0, "expanded": 1}
        if trace:
            result["expanded_nodes"] = [(start[0], start[1], 0)]
        return result

    # Heap entries are (f, h, x, y, t, g, path, seen): f/h/x/y/t give the
    # fixed numeric priority and candidates still tied are ordered by the
    # complete coordinate path lexicographically, so ordering never depends
    # on obstacle sets, in-frame coordinate order, or traversal order. The
    # frozenset ``seen`` mirrors ``path`` for an O(1) repeat check and is
    # never compared (distinct histories always differ in ``path``).
    start_path = (start,)
    open_heap = [(h0, h0, start[0], start[1], 0, 0, start_path,
                  frozenset(start_path))]
    closed_count = 0
    expanded_nodes = [] if trace else None
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

    while open_heap:
        if best_key is not None and open_heap[0][0] > best_key[0]:
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
            new_path = path + (nxt,)
            h = heuristic(nxt)
            heapq.heappush(
                open_heap,
                (new_g + h, h, nx, ny, next_t, new_g, new_path,
                 seen | {nxt}),
            )
    if best_key is None:
        result = {"path": None, "cost": None, "expanded": closed_count}
    else:
        best_g, _, best_path = best_key
        result = {
            "path": list(best_path),
            "cost": best_g,
            "expanded": closed_count,
        }
    if trace:
        result["expanded_nodes"] = expanded_nodes
    return result


def plan(width, height, blocked, start, goal, costs=None, trace=False,
         dynamic_blocked=None):
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
    if frames is not None:
        if start in frames[0]:
            raise ValueError(
                f"start {start} is blocked at frame 0"
            )
        return _search_dynamic(
            width, height, obstacles, frames, start, goal, costs, trace
        )

    def heuristic(point):
        return abs(point[0] - goal[0]) + abs(point[1] - goal[1])

    def step_cost(point):
        # Cost of entering ``point``; the start cell is never entered.
        if costs is None:
            return 1
        return costs[point[1]][point[0]]

    # Heap entries are (f, h, x, y, point): the first four fields give a
    # total, input-order-independent ordering, so the point itself is never
    # compared.
    h0 = heuristic(start)
    open_heap = [(h0, h0, start[0], start[1], start)]
    came_from = {}
    g_score = {start: 0}
    closed = set()
    expanded_nodes = [] if trace else None

    while open_heap:
        _, _, _, _, current = heapq.heappop(open_heap)
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
                g_score[nxt] = new_g
                came_from[nxt] = current
                h = heuristic(nxt)
                heapq.heappush(
                    open_heap, (new_g + h, h, nxt[0], nxt[1], nxt)
                )
    result = {"path": None, "cost": None, "expanded": len(closed)}
    if trace:
        result["expanded_nodes"] = expanded_nodes
    return result


def replay(width, height, blocked, start, goal, path, costs=None,
           dynamic_blocked=None):
    """Offline verification of a candidate path against the plan rules.

    The grid arguments (``width``, ``height``, ``blocked``, ``start``,
    ``goal``, ``costs``, ``dynamic_blocked``) are validated exactly as in
    ``plan`` — same checks, same ``TypeError``/``ValueError`` boundaries,
    same order — and all of that validation runs before the path itself is
    examined. No search is performed and no ``expanded``/``expanded_nodes``
    fields are produced.

    ``path`` must be a non-empty sequence of grid coordinates; strings,
    bytes, ``None``, non-sequences and elements that are not two integers
    raise ``TypeError``, an empty path raises ``ValueError``, and
    out-of-bounds coordinates raise ``ValueError``.

    Semantically invalid paths do not raise: they return the fixed
    structure ``{"valid": False, "cost": None, "steps": None}``. A path is
    semantically invalid when it does not start at ``start``, does not end
    at ``goal``, enters a static obstacle or the dynamic obstacle frame for
    its time index (frames past the last one reuse the last frame; no
    frames means static-only checking), moves outside the four-neighborhood,
    repeats a coordinate, or has extra points when ``start == goal``.

    A valid path returns ``{"valid": True, "cost": int, "steps": int}``
    where ``cost`` is recomputed with the same entering-cell rule as
    ``plan`` (unit cost without ``costs``; the start cell is never
    counted) and ``steps`` is the number of moves, ``len(path) - 1``.
    """
    # --- Grid validation: identical to plan, before the path is read. ---
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
        raise ValueError(
            f"start {start} is blocked at frame 0"
        )

    # --- Path structure validation. ---
    if isinstance(path, (str, bytes)) or not isinstance(path, Sequence):
        raise TypeError(
            f"path must be a non-empty sequence of grid coordinates, "
            f"got {type(path).__name__}"
        )
    if len(path) == 0:
        raise ValueError("path must not be empty")
    points = []
    for index, item in enumerate(path):
        name = f"path[{index}]"
        point = _normalize_point(item, name)
        _check_bounds(point, width, height, name)
        points.append(point)

    invalid = {"valid": False, "cost": None, "steps": None}

    # --- Semantic checks: any violation yields the fixed invalid result. ---
    if points[0] != start or points[-1] != goal:
        return invalid
    if start == goal and len(points) > 1:
        return invalid  # extra points on a zero-move route
    if len(set(points)) != len(points):
        return invalid  # repeated coordinate
    if frames is not None:
        last_frame = len(frames) - 1
        for t, point in enumerate(points):
            if point in obstacles:
                return invalid
            frame = frames[t] if t <= last_frame else frames[last_frame]
            if point in frame:
                return invalid
    else:
        for point in points:
            if point in obstacles:
                return invalid
    for a, b in zip(points, points[1:]):
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1:
            return invalid  # not a four-neighborhood move

    if costs is None:
        cost = len(points) - 1
    else:
        cost = sum(costs[y][x] for x, y in points[1:])
    return {"valid": True, "cost": cost, "steps": len(points) - 1}
