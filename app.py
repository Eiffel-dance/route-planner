"""Deterministic 2D grid A* planner.

Public entry point: ``plan(width, height, blocked, start, goal, costs=None,
trace=False, dynamic_blocked=None)``.

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

Dynamic obstacles (``dynamic_blocked``):
- ``None`` or an empty sequence disables the dynamic mode entirely; every
  result is then identical to the static planner described above.
- Otherwise it must be a sequence of frames; each frame is an iterable of
  grid coordinates blocked at that time step. Frame 0 constrains ``start``;
  the coordinate at path index ``t`` must avoid both the static obstacles
  and frame ``t``. Indices past the last frame reuse the last frame.
- The search runs over spacetime states ``(x, y, t)``: each move still goes
  to a four-neighbor (no waiting in place) and no coordinate may appear
  twice in a path. Costs and the Manhattan heuristic are unchanged.
- Tie-breaking compares f, then h, then x, y, t, so results do not depend
  on the iteration order of any frame or coordinate set.
- On success ``path`` is still a sequence of coordinate tuples; on failure
  ``expanded`` reports the number of spacetime states actually closed.
- With ``trace=True`` the dynamic mode records ``expanded_nodes`` as
  ``(x, y, t)`` triples in closing order; ``trace=False`` still omits the
  field. The static mode keeps its coordinate-pair trace.
- Validation: the outer value must be a sequence and every frame an
  iterable of coordinates; strings, bytes and non-two-integer coordinates
  raise ``TypeError``, out-of-bounds coordinates raise ``ValueError``, and
  duplicate coordinates within a frame are merged. A ``start`` forbidden
  by frame 0 raises ``ValueError``. All checks run before the search.
"""

import heapq
from collections.abc import Iterable, Sequence

__all__ = ["plan"]

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
            "dynamic_blocked must be a sequence of frames, each an "
            f"iterable of grid coordinates, "
            f"got {type(dynamic_blocked).__name__}"
        )
    frames = []
    for index, frame in enumerate(dynamic_blocked):
        name = f"dynamic_blocked[{index}]"
        if isinstance(frame, (str, bytes)) or not isinstance(frame, Iterable):
            raise TypeError(
                f"{name} must be an iterable of grid coordinates, "
                f"got {type(frame).__name__}"
            )
        cells = set()
        for item_index, item in enumerate(frame):
            item_name = f"{name}[{item_index}]"
            point = _normalize_point(item, item_name)
            _check_bounds(point, width, height, item_name)
            cells.add(point)  # duplicates merge into one cell
        frames.append(frozenset(cells))
    if not frames:
        return None  # empty sequence: dynamic mode disabled
    return frames


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
    dynamic_frames = _normalize_dynamic_blocked(
        dynamic_blocked, width, height
    )
    if dynamic_frames is not None and start in dynamic_frames[0]:
        raise ValueError(f"start {start} is blocked at frame 0")

    def heuristic(point):
        return abs(point[0] - goal[0]) + abs(point[1] - goal[1])

    def step_cost(point):
        # Cost of entering ``point``; the start cell is never entered.
        if costs is None:
            return 1
        return costs[point[1]][point[0]]

    if dynamic_frames is not None:
        return _plan_dynamic(
            width, height, obstacles, start, goal, step_cost, heuristic,
            trace, dynamic_frames,
        )

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


def _plan_dynamic(width, height, obstacles, start, goal, step_cost,
                  heuristic, trace, dynamic_frames):
    """A* over spacetime states ``(x, y, t)`` with per-frame obstacles.

    Moves go to four-neighbors only (no waiting); a coordinate already
    closed at any time is never entered again, so paths contain no repeated
    coordinates. Frame ``min(t, last)`` constrains the coordinate at path
    index ``t``.
    """
    last_frame = len(dynamic_frames) - 1

    def blocked_at(point, t):
        frame = dynamic_frames[t] if t <= last_frame else dynamic_frames[last_frame]
        return point in frame

    # Heap entries are (f, h, x, y, t, state): the first five fields give a
    # total, input-order-independent ordering, so the state is never
    # compared.
    h0 = heuristic(start)
    start_state = (start[0], start[1], 0)
    open_heap = [(h0, h0, start[0], start[1], 0, start_state)]
    came_from = {start_state: None}
    g_score = {start_state: 0}
    closed = set()
    closed_coords = set()
    expanded_nodes = [] if trace else None

    while open_heap:
        _, _, _, _, _, current = heapq.heappop(open_heap)
        current_xy = (current[0], current[1])
        if current in closed or current_xy in closed_coords:
            continue  # stale entry: state or coordinate already closed
        closed.add(current)
        closed_coords.add(current_xy)
        if expanded_nodes is not None:
            expanded_nodes.append(current)
        if current_xy == goal:
            path = [current_xy]
            state = current
            while came_from[state] is not None:
                state = came_from[state]
                path.append((state[0], state[1]))
            path.reverse()
            result = {
                "path": path,
                "cost": g_score[current],
                "expanded": len(closed),
            }
            if trace:
                result["expanded_nodes"] = expanded_nodes
            return result
        nt = current[2] + 1
        for dx, dy in _NEIGHBORS:
            nxt = (current[0] + dx, current[1] + dy)
            if not (0 <= nxt[0] < width and 0 <= nxt[1] < height):
                continue
            if nxt in obstacles or nxt in closed_coords:
                continue
            if blocked_at(nxt, nt):
                continue
            new_g = g_score[current] + step_cost(nxt)
            nxt_state = (nxt[0], nxt[1], nt)
            if new_g < g_score.get(nxt_state, float("inf")):
                g_score[nxt_state] = new_g
                came_from[nxt_state] = current
                h = heuristic(nxt)
                heapq.heappush(
                    open_heap,
                    (new_g + h, h, nxt[0], nxt[1], nt, nxt_state),
                )
    result = {"path": None, "cost": None, "expanded": len(closed)}
    if trace:
        result["expanded_nodes"] = expanded_nodes
    return result
