import heapq
from collections.abc import Sequence

_NEIGHBORS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_dimension(name, value):
    if not _is_int(value):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}")
    return value


def _normalize_coord(name, value):
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(
            f"{name} must be a sequence of two integers, got {type(value).__name__}"
        )
    if len(value) != 2:
        raise TypeError(f"{name} must have exactly 2 elements, got {len(value)}")
    x, y = value
    if not _is_int(x) or not _is_int(y):
        raise TypeError(f"{name} elements must be ints, got {value!r}")
    return (x, y)


def _check_bounds(name, coord, width, height):
    x, y = coord
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError(
            f"{name} {coord} is outside the grid {width}x{height}"
        )


def plan(width, height, blocked, start, goal):
    # Validate everything before the search starts so a bad input can never
    # produce a partial result.
    width = _validate_dimension("width", width)
    height = _validate_dimension("height", height)
    start = _normalize_coord("start", start)
    goal = _normalize_coord("goal", goal)
    _check_bounds("start", start, width, height)
    _check_bounds("goal", goal, width, height)

    obstacles = set()
    for raw in blocked:
        coord = _normalize_coord("blocked entry", raw)
        _check_bounds("blocked entry", coord, width, height)
        obstacles.add(coord)  # duplicates merge without changing the result

    if start in obstacles:
        raise ValueError(f"start {start} lies on a blocked cell")
    if goal in obstacles:
        raise ValueError(f"goal {goal} lies on a blocked cell")

    def h(p):
        return abs(p[0] - goal[0]) + abs(p[1] - goal[1])

    # Heap entries are (f, h, x, y): ties break on h, then lexicographic
    # (x, y). This order is fixed and public, so the result never depends on
    # the iteration order of the blocked input.
    g = {start: 0}
    came = {}
    closed = set()
    expanded = 0
    heap = [(h(start), h(start), start[0], start[1])]
    while heap:
        _, _, x, y = heapq.heappop(heap)
        cur = (x, y)
        if cur in closed:  # stale heap entry, already closed
            continue
        closed.add(cur)
        expanded += 1
        if cur == goal:
            path = [cur]
            while path[-1] in came:
                path.append(came[path[-1]])
            path.reverse()
            return {"path": path, "cost": g[cur], "expanded": expanded}
        for dx, dy in _NEIGHBORS:
            nxt = (x + dx, y + dy)
            if (
                nxt in closed
                or nxt in obstacles
                or not (0 <= nxt[0] < width and 0 <= nxt[1] < height)
            ):
                continue
            new = g[cur] + 1
            if new < g.get(nxt, float("inf")):
                g[nxt] = new
                came[nxt] = cur
                hn = h(nxt)
                heapq.heappush(heap, (new + hn, hn, nxt[0], nxt[1]))
    return {"path": None, "cost": None, "expanded": expanded}
