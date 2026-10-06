"""Deterministic 2D grid A* planner.

Public entry points: ``plan(width, height, blocked, start, goal, costs=None,
trace=False, dynamic_blocked=None, max_expanded=None, snapshot=False,
max_cost=None, allow_wait=False, dynamic_costs=None)``,
``plan_any(width, height, blocked,
start, goals, costs=None, trace=False, dynamic_blocked=None,
max_expanded=None, snapshot=False, max_cost=None, allow_wait=False,
dynamic_costs=None)``,
``plan_k(width, height, blocked, start, goal, k, costs=None,
dynamic_blocked=None, max_expanded=None, max_cost=None,
allow_wait=False, dynamic_costs=None)``,
``plan_batch(width, height, blocked, requests, costs=None, trace=False,
dynamic_blocked=None, max_expanded=None, snapshot=False,
max_cost=None, allow_wait=False, dynamic_costs=None)``,
``plan_multi_start(width, height, blocked, starts, goal, costs=None,
trace=False, dynamic_blocked=None, max_expanded=None, snapshot=False,
max_cost=None, allow_wait=False, dynamic_costs=None)``,
``replay(width, height, blocked, start, goal, path, costs=None,
dynamic_blocked=None, diagnose=False, allow_wait=False,
dynamic_costs=None)``,
``resume(checkpoint, max_expanded=None)``,
``distance_field(width, height, blocked, goal, costs=None, trace=False)``
and
``distance_field_any(width, height, blocked, goals, costs=None,
trace=False)``.

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
  exception type and validation order of the previous behavior untouched.
  Otherwise it must be a non-negative integer that is not a bool: other
  types raise ``TypeError`` and negative values raise ``ValueError``. The
  check runs after all existing grid, costs, ``trace``, ``dynamic_blocked``,
  ``max_expanded`` and ``snapshot`` validation and before the search
  starts.
- The limit caps the accumulated entering-cell cost of a route, measured
  from the start (the start cell still costs 0 and each entered cell costs
  its ``costs`` value, or 1 without ``costs``). A candidate whose
  accumulated cost would exceed the limit is discarded; a candidate at
  exactly the limit still competes. The limit applies identically in
  static mode, dynamic mode and ``plan_any``, and never changes expansion
  order, tie-breaks, the minimum-cost choice, the multi-goal endpoint
  comparison or trace entries.
- When ``max_cost`` is provided the result always carries ``status``:
  ``"found"`` on success, and on failure ``"cost_exhausted"`` when at
  least one candidate was discarded for exceeding the limit, otherwise
  ``"unreachable"`` (``path``/``cost`` are ``None`` and ``expanded``
  counts the actually closed nodes). When ``max_expanded`` is also given,
  a budget stop still takes priority with ``"budget_exhausted"``.
  ``trace`` keeps recording only actually closed nodes, and the
  ``start == goal`` single-point zero-cost result and the zero-budget
  rules are unchanged.
- With ``snapshot=True`` the checkpoint records the ``max_cost`` value
  when one was provided (and no new field when it was not). ``resume``
  treats a checkpoint without the field as ``None``, validates a present
  field with the same rules (``TypeError`` for wrong types, ``ValueError``
  for negative values or a state inconsistent with the limit) before any
  searching, and a resumed search is equivalent to one uninterrupted call
  with the same limit. ``replay`` accepts no ``max_cost`` parameter.

Waiting in place (``allow_wait``):
- ``plan``, ``plan_any``, ``plan_batch``, ``plan_multi_start``, ``plan_k``
  and ``replay`` accept a final optional boolean ``allow_wait`` that
  defaults to ``False``. ``False`` -- or an omitted, ``None`` or empty
  ``dynamic_blocked`` -- keeps every key, value, exception type,
  validation order, four-neighborhood move, no-repeated-coordinate rule
  and tie-break of the previous behavior untouched. A non-bool
  ``allow_wait`` raises ``TypeError`` after the ``dynamic_blocked``
  checks (the frame-0 check included) and before the budget and snapshot
  checks; every other exception type and order is unchanged.
- With ``allow_wait=True`` a route may stay in its current cell from
  time ``t`` to time ``t + 1`` provided the cell is not blocked at frame
  ``t + 1``. Waiting adds no entering-cell cost (``cost`` still sums
  only the cells entered by moves, and ``max_cost`` caps only those) and
  is only generated while ``t + 1`` lies within the provided frames:
  after the last frame the last frame is still reused for moves but no
  further waits are generated. Consecutive waits repeat the same
  coordinate in the path; once a cell is left it may never be
  re-entered. ``start == goal`` still yields only the single-point
  zero-cost route.
- The minimum-cost choice, the endpoint selection and the complete-path
  lexicographic tie-break follow the existing rules, with the priority
  still f, then h, then the coordinate, then the time and finally the
  complete path, so results never depend on obstacle-set or in-frame
  iteration order. Wait states count toward ``expanded`` one per closing
  and the trace records their ``(x, y, t)`` triples; ``max_expanded``
  truncates them like any other state and the ``status`` values and
  priorities are unchanged. ``plan_k``, ``plan_batch``, multi-goal and
  multi-start results follow the same waiting semantics, and a
  successful path verifies in ``replay`` with the same cost and step
  count, the waits included in ``steps``.
- ``replay`` with ``allow_wait=True`` accepts consecutive stays only
  within the provided dynamic frames; stays in static mode or without
  frames, stays after the final frame and non-consecutive revisits all
  return the usual invalid structure. With ``diagnose=True`` a stay
  after the final frame reports the dedicated error
  ``wait_after_final_frame``.
- A budget-exhausted checkpoint records ``allow_wait`` when it was true
  and ``resume`` of such a checkpoint is equivalent to one uninterrupted
  call with the same setting. Checkpoints written without the field
  restore it as ``False``; a present non-bool field raises ``TypeError``
  and a field inconsistent with the recorded internal state raises
  ``ValueError``.

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

Time-varying costs (``dynamic_costs``):
- ``plan``, ``plan_any``, ``plan_batch``, ``plan_multi_start``,
  ``plan_k`` and ``replay`` accept a final optional
  ``dynamic_costs=None`` parameter. Omitting it, passing ``None`` or an
  empty sequence keeps every result key, exception boundary, expansion
  order, path choice and the legacy ``costs`` semantics of the previous
  behavior untouched; every other entry point keeps its signature.
- Otherwise ``dynamic_costs`` must be a sequence of height-by-width
  matrices of positive integers, one matrix per time frame. An outer
  value or frame that is not an acceptable sequence, or a cell that is
  not a non-bool integer, raises ``TypeError``; a wrong row or column
  count, a non-positive cost, or providing both ``costs`` and
  ``dynamic_costs`` raises ``ValueError``. All checks run after every
  pre-existing validation (in ``plan_batch`` before any request is
  searched) and before the search starts.
- When enabled, entering a coordinate at path index ``t`` costs that
  cell's value in frame ``t``; the start cell is still never charged and
  frames beyond the last one reuse the last frame. Waiting follows the
  existing rules and never produces an entering-cell cost. The search
  returns the globally minimum total ``cost`` route under the static and
  dynamic obstacle constraints; equal costs are still arbitrated by the
  existing endpoint and complete-path lexicographic rules, and
  ``expanded``/``expanded_nodes`` count only actually closed space-time
  states: distinct histories are never merged and no out-of-bounds,
  blocked or illegally repeated nodes are generated.
- ``max_cost`` caps the dynamic accumulated cost with the usual
  ``cost_exhausted`` and budget-priority rules; ``plan_any``'s winning
  endpoint rules, the trace shape, the ``start == goal`` result and
  unreachable results are unchanged. ``plan_batch`` answers each
  request exactly as an independent ``plan`` call with the same frames;
  ``plan_multi_start`` picks its winner by the same total-cost and
  complete-path tie-breaks and records ``(x, y, t)`` trace triples in
  dynamic mode; ``plan_k`` ranks its unique routes by the dynamic
  accumulated cost with the usual truncation and status rules.
  ``replay`` recomputes ``cost``
  with the same frames, so a legal path matches the planning result;
  the diagnostic error types and the ``steps`` rule are unchanged.
- With ``snapshot=True`` the checkpoint records ``dynamic_costs`` as a
  JSON-serializable structure (the field is omitted when they were not
  provided) together with the waiting setting, the accumulated state and
  the cost limit when one was provided; a resumed search matches one
  uninterrupted call field by field, the next checkpoint included. A
  checkpoint whose ``dynamic_costs`` is corrupt raises ``TypeError`` or
  ``ValueError`` and never silently degrades. ``plan_k`` accepts no
  snapshot option.

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

Top-k planning (``plan_k``):
- ``plan_k(width, height, blocked, start, goal, k, costs=None,
  dynamic_blocked=None, max_expanded=None, max_cost=None,
  allow_wait=False, dynamic_costs=None)`` returns
  several distinct feasible routes on the same grid in one call. It
  first performs exactly ``plan``'s shared validation for the
  dimensions, coordinates, static obstacles, positive integer cost
  matrix and dynamic frames (including the frame-0 start check), and
  only then checks ``k``: a non-bool positive integer is required,
  wrong types raise ``TypeError`` and non-positive values raise
  ``ValueError``. The optional ``max_expanded`` and ``max_cost``
  limits are checked after all of that (the ``k`` check included),
  still before the search starts, with the same non-negative,
  non-bool integer rules ``plan`` uses (other types raise
  ``TypeError``, negative values raise ``ValueError``). The optional
  ``dynamic_costs`` frames are validated after all of those checks,
  still before the search starts, with the shared time-varying cost
  rules (providing both ``costs`` and ``dynamic_costs`` raises
  ``ValueError``). Every error is
  decided before the search starts, and ``plan_k`` accepts no
  ``trace`` or ``snapshot`` argument.
- The search keeps four-neighborhood moves, the zero start cost, the
  entering-cell cost accumulation, the persistent last frame, the ban on
  waiting in place and on repeated coordinates, and the static/frame
  feasibility rules. Routes that reach the same cell through different
  coordinate histories are distinct candidates and are never merged, in
  both static and dynamic mode, so different complete coordinate
  sequences always remain separate results.
- It returns ``{"paths": [...], "costs": [...], "expanded": int}``: at
  most ``k`` unique complete routes ordered by ascending total cost and,
  on ties, by the lexicographic order of the complete coordinate
  sequence; ``costs`` corresponds to ``paths`` entry by entry. The first
  entry is the route ``plan`` would return. When no route is feasible
  both lists are empty and ``expanded`` still reports the route
  candidates actually closed to determine the answer: one count per
  complete history, and never counting a candidate filtered at
  generation or left pending. The traversal stops the moment the k-th
  goal route has been closed, so a small k closes strictly fewer
  candidates than a larger one. ``start == goal`` yields only the
  single-point zero-cost route, with one expansion. Every returned route
  verifies offline in ``replay`` with the same cost and step count, in
  both static and dynamic mode; the result never depends on obstacle-set
  iteration order, in-frame coordinate order or the traversal order.
- ``max_cost`` caps a route's accumulated entering-cell cost exactly as
  in ``plan``: the start still costs 0, a candidate exactly at the
  limit still competes and a candidate whose accumulated cost would
  exceed it is discarded the moment it is generated. It never changes
  the tie-break between legal routes. ``max_expanded`` caps the number
  of complete route candidates actually closed (exactly what
  ``expanded`` counts); once the limit is reached no further candidate
  is closed, and the limit only ever truncates the traversal -- it can
  never replace an already determined higher-ranked route. When either
  limit is provided the result additionally carries ``status``:
  ``"found"`` when all ``k`` routes were closed or the search exhausted
  naturally with at least one route; ``"budget_exhausted"`` when fewer
  than ``k`` routes were found and the budget stopped the search while
  candidates were still pending (this wins when both limits fire);
  ``"cost_exhausted"`` when fewer than ``k`` routes were found, no
  budget stop occurred and at least one candidate was discarded for the
  cost limit; ``"unreachable"`` when no route exists with neither a
  budget stop nor a cost discard. The routes already closed (and their
  costs, in rank order) are still returned when a limit ends the search
  early, as empty lists when none were found, and ``expanded`` is the
  actual closed count. Omitting both limits keeps every legacy key and
  value exactly unchanged.

Batch planning (``plan_batch``):
- ``plan_batch(width, height, blocked, requests, costs=None,
  trace=False, dynamic_blocked=None, max_expanded=None, snapshot=False,
  max_cost=None, allow_wait=False, dynamic_costs=None)`` answers
  several independent single-goal queries on
  one shared grid in a single call. It returns ``{"results": [...]}``
  with exactly the result ``plan`` would return for each query, in the
  order of ``requests``; duplicate requests are kept and each gets its
  own result.
- The shared arguments are validated with ``plan``'s rules and order,
  with ``requests`` taking the position of ``start``/``goal`` in that
  sequence. ``requests`` must be a non-empty sequence: ``None``,
  strings, bytes and other non-sequences raise ``TypeError`` and an
  empty sequence raises ``ValueError``. Each element must be a
  two-element sequence ``[start, goal]`` of grid coordinates: element
  shape or coordinate type failures raise ``TypeError``; out-of-bounds
  coordinates, endpoints on static obstacles and a start blocked at
  dynamic frame 0 raise ``ValueError``. Every check is decided before
  any search starts, so an invalid batch never returns partial results.
- Each query is an independent ``plan`` search over the shared grid:
  ``path``, ``cost`` and ``expanded`` are produced per request, and the
  ``max_expanded`` budget and the ``max_cost`` limit apply per query
  with no state shared across queries. The optional ``dynamic_costs``
  frames follow the shared time-varying cost rules and are validated
  after every other check, before any request is searched; each request
  is then answered exactly as an independent ``plan`` call with the
  same frames, and a per-request checkpoint records them. ``trace``,
  ``status``,
  ``expanded_nodes``, ``checkpoint``, dynamic frames, ``start == goal``,
  unreachable and ``cost_exhausted`` results all follow ``plan``
  exactly, and a returned ``checkpoint`` can be handed to ``resume``
  directly. A successful path ends at its request's goal and verifies
  offline in ``replay`` with the same cost and step count. The result
  never depends on obstacle-set iteration order, in-frame coordinate
  order or the permutation of ``requests``.

Multi-start planning (``plan_multi_start``):
- ``plan_multi_start(width, height, blocked, starts, goal, costs=None,
  trace=False, dynamic_blocked=None, max_expanded=None, snapshot=False,
  max_cost=None, allow_wait=False, dynamic_costs=None)`` plans from
  several candidate starts to one shared
  goal in a single call and returns one deterministic route, in exactly
  the result structure of ``plan``; the path begins at the winning start
  and ends at ``goal``. ``starts`` takes ``start``'s position in
  ``plan``'s validation sequence and every shared check keeps its order,
  exception type and message boundary.
- ``starts`` must be a non-empty sequence of grid coordinates: ``None``,
  strings, bytes and other non-sequences raise ``TypeError`` and an
  empty sequence raises ``ValueError``. Each start follows the
  coordinate structure and integer rules (bad shape or coordinate types
  raise ``TypeError``); out-of-bounds coordinates and starts on static
  obstacles raise ``ValueError``. Repeated starts merge and the tuple is
  sorted, so the input ordering never changes any result. A start
  blocked by dynamic frame 0 raises ``ValueError``; every check runs
  before the search starts.
- In static mode every start is a ``g == 0`` root and each coordinate is
  closed at most once: a cell keeps only its best known ``(cost, path)``
  pair (smallest accumulated entering-cell cost, then the
  lexicographically smallest complete coordinate sequence -- an ordering
  preserved under common extensions, so a dominated candidate can never
  win later). In dynamic mode the persistent last frame, the ban on
  waiting and on repeated coordinates apply, and routes reaching the
  same ``(x, y, t)`` through different coordinate histories are distinct
  states that are never merged. Time-varying ``dynamic_costs`` frames
  (validated after every other check, before the search) make the
  search time-expanded the same way obstacle frames do: entering a
  coordinate at path index ``t`` costs that frame's cell value, the
  last frame persists, and the trace records ``(x, y, t)`` triples.
- The winning route has the smallest total ``cost``; ties are decided by
  the lexicographic order of the complete coordinate path, so the result
  never depends on the input order of ``starts``. Expansion order is f,
  then h, then ``(x, y)`` (plus ``t`` in dynamic mode), and finally the
  complete path. ``cost`` counts only the cells entered (the winning
  start costs 0); ``expanded`` counts only closed candidates; when no
  route exists ``path`` and ``cost`` are ``None``.
- With ``trace=True`` static mode records the unique coordinate closing
  sequence and dynamic mode the ``(x, y, t)`` closing sequence, and
  ``expanded == len(expanded_nodes)``. A ``starts`` entry equal to
  ``goal`` yields the single-point zero-cost route under the same
  start-equals-goal and zero-budget rules as ``plan``.
- ``max_expanded`` and ``max_cost`` follow ``plan``'s ``status`` and
  budget-priority rules: a candidate whose accumulated cost would
  exceed ``max_cost`` is discarded without counting toward ``expanded``,
  and neither limit ever changes the unlimited route or the tie-breaks.
  With ``snapshot=True`` a budget stop returns a JSON-serializable
  checkpoint recording the sorted starts, the grid constraints, the
  dynamic frames, the time-varying cost frames when they were provided,
  the cost limit when one was provided and the complete
  pending state; ``resume`` continues it with the cumulative-budget,
  no-double-counting and only-budget-exhausted-snapshots rules of the
  other planners, and invalid checkpoints follow ``resume``'s usual
  ``TypeError``/``ValueError`` boundaries.

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
  candidate goals, costs, dynamic frames and the cost limit when one was
  provided), the closed-node count, the
  trace recorded so far and the complete pending-candidate state. Every
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

Distance field (``distance_field``):
- ``distance_field(width, height, blocked, goal, costs=None,
  trace=False)`` is an offline analysis entry point for the static grid:
  it accepts only the dimensions, the static obstacles, a single goal and
  optionally ``costs`` and ``trace`` -- no start, no dynamic obstacles, no
  search budget, no cost limit and no snapshot option. The dimensions,
  goal coordinate, obstacle and costs checks (structure, integer types,
  bounds, duplicate merging, goal-on-obstacle) follow ``plan``'s public
  rules and order for those shared arguments and all run before the
  search starts; ``trace`` must be a bool (``TypeError`` otherwise).
- It returns ``{"distances": ..., "expanded": int}`` where ``distances``
  is a height-by-width matrix indexed ``distances[y][x]``: ``None`` for
  obstacles and for cells that cannot reach the goal, ``0`` for the goal
  itself, and for every other reachable cell the minimum sum of the
  entering-cell costs along a four-neighborhood route from that cell to
  the goal (unit steps without ``costs``; with ``costs`` each move adds
  the cost of the cell entered, exactly as ``plan`` accumulates ``cost``,
  so a reachable ``plan`` start yields the same value ``plan`` reports
  as ``cost``). No reachable distance is ever negative and the matrix
  always matches the grid dimensions; the result is JSON-serializable.
- ``expanded`` counts exactly the reachable cells determined and closed
  once each, the goal included. The closing order is fully specified:
  smallest current distance first, then the ``(x, y)`` coordinate in
  lexicographic order, so neither the obstacle-set iteration order nor
  the costs layout traversal influences the matrix, the count or the
  trace. With ``trace=True`` the result additionally carries
  ``expanded_nodes``, the coordinate pairs in that closing order, and
  ``expanded == len(expanded_nodes)``; the field is omitted entirely
  when ``trace`` is false or omitted.

Multi-goal distance field (``distance_field_any``):
- ``distance_field_any(width, height, blocked, goals, costs=None,
  trace=False)`` is the multi-goal counterpart of ``distance_field``:
  one offline analysis returns the minimum-cost field to any of the
  candidate endpoints in a single pass. It accepts only the dimensions,
  the static obstacles, a non-empty ``goals`` sequence and optionally
  ``costs`` and ``trace`` -- no start, no dynamic obstacles, no search
  budget, no cost limit and no snapshot option, and it never produces a
  ``status`` field. The dimensions, obstacles, costs and trace checks
  follow ``distance_field``'s public rules and order, and ``goals`` is
  validated exactly like ``plan_any``'s: ``None``, strings, bytes and
  other non-sequences raise ``TypeError``, an empty sequence raises
  ``ValueError``, elements that are not exactly two non-bool integers
  raise ``TypeError``, and out-of-bounds coordinates or endpoints on
  static obstacles raise ``ValueError``. The goals checks occupy
  ``goal``'s position in ``distance_field``'s validation sequence,
  duplicates merge, and every check runs before the search starts, so
  the input permutation never changes any result.
- It returns ``{"distances": ..., "targets": ..., "expanded": int}``.
  ``distances`` is the same height-by-width matrix as
  ``distance_field``: ``None`` for obstacles and for cells that cannot
  reach any endpoint, ``0`` for every endpoint, and otherwise the
  minimum over the endpoints of the accumulated entering-cell cost
  along a four-neighborhood route (unit steps without ``costs``; with
  ``costs`` each move adds the cost of the cell entered, the start
  cell's own cost never counted, exactly as ``plan``/``plan_any`` and
  ``distance_field`` accumulate). A reachable ``plan_any`` start yields
  the same value ``plan_any`` reports as ``cost``.
- ``targets`` is a matrix of the same shape: a reachable cell records
  the coordinate tuple of the endpoint that wins it, every other
  position is ``None`` (obstacles and unreachable cells, exactly where
  ``distances`` is ``None``); endpoints record themselves. When several
  endpoints tie at the same minimum distance the winner is the endpoint
  with the lexicographically smallest ``(x, y)`` coordinate, so the
  choice never depends on goal order. The whole result is
  JSON-serializable.
- ``expanded`` counts every reachable cell closed for the first time,
  all endpoints included, exactly once each. The closing order is
  smallest distance first, then the ``(x, y)`` coordinate
  lexicographically. With ``trace=True`` the result additionally
  carries ``expanded_nodes`` -- the coordinate pairs in that order,
  ``expanded == len(expanded_nodes)`` -- and the key is omitted
  entirely when ``trace`` is false or omitted.

Validation (all performed before the search starts):
- ``width``/``height`` must be positive integers.
- ``start``, ``goal`` and every entry of ``blocked`` must be grid
  coordinates: sequences of exactly two integers (tuples, lists, etc.),
  normalized to tuples.
- ``goals`` (``plan_any`` and ``distance_field_any``) must be a
  non-empty sequence of grid coordinates following the same rules,
  normalized, deduplicated and order-independent.
- ``requests`` (``plan_batch``) must be a non-empty sequence of
  two-element ``[start, goal]`` coordinate pairs following the same
  coordinate rules, with the input order and duplicates preserved.
- ``costs`` may be omitted or ``None`` (unit costs). Otherwise it must be a
  sequence of ``height`` rows, each a sequence of ``width`` positive
  integers; strings/bytes, non-integer cells and booleans are rejected.
- ``dynamic_costs`` (``plan``, ``plan_any``, ``plan_batch``,
  ``plan_multi_start``, ``plan_k`` and ``replay``) may be
  omitted, ``None`` or an empty sequence. Otherwise it must be a
  sequence of frames, each frame a height-by-width matrix following the
  ``costs`` rules per frame; providing both ``costs`` and
  ``dynamic_costs`` raises ``ValueError``.
- ``trace`` must be a bool; non-bool values raise ``TypeError``.
- Type or structure violations raise ``TypeError``; non-positive
  dimensions, out-of-bounds coordinates, endpoints on obstacles, wrong
  matrix shape, or non-positive cell costs raise ``ValueError``.
- Duplicate blocked cells are merged without changing the result.
"""

import heapq
from collections.abc import Iterable, Sequence

__all__ = ["plan", "plan_any", "plan_k", "plan_batch", "plan_multi_start",
           "replay", "resume", "distance_field", "distance_field_any"]

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


def _normalize_dynamic_costs(dynamic_costs, width, height):
    # ``dynamic_costs`` is the time-varying counterpart of ``costs``: a
    # sequence of height-by-width positive integer matrices, one per time
    # frame. ``None`` or an empty sequence disables it (equivalent to not
    # provided). The outer value and every frame must be sequences
    # (strings/bytes and other non-sequences raise ``TypeError``); a frame
    # with the wrong row count, a row with the wrong cell count, or a
    # non-positive cell cost raises ``ValueError``; a cell that is not a
    # non-bool integer raises ``TypeError``. All checks run before the
    # search starts.
    if dynamic_costs is None:
        return None  # time-varying costs disabled
    if isinstance(dynamic_costs, (str, bytes)) or not isinstance(
        dynamic_costs, Sequence
    ):
        raise TypeError(
            f"dynamic_costs must be a sequence of cost frames, "
            f"got {type(dynamic_costs).__name__}"
        )
    if len(dynamic_costs) == 0:
        return None  # no frames: equivalent to not provided
    frames = []
    for t, frame in enumerate(dynamic_costs):
        if isinstance(frame, (str, bytes)) or not isinstance(frame, Sequence):
            raise TypeError(
                f"dynamic_costs[{t}] must be a height-by-width matrix of "
                f"positive integers, got {type(frame).__name__}"
            )
        if len(frame) != height:
            raise ValueError(
                f"dynamic_costs[{t}] must have exactly {height} rows, "
                f"got {len(frame)}"
            )
        rows = []
        for y, row in enumerate(frame):
            if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
                raise TypeError(
                    f"dynamic_costs[{t}][{y}] must be a sequence of "
                    f"{width} positive integers, got {type(row).__name__}"
                )
            if len(row) != width:
                raise ValueError(
                    f"dynamic_costs[{t}][{y}] must have exactly {width} "
                    f"cells, got {len(row)}"
                )
            parsed = []
            for x, cell in enumerate(row):
                if not _is_int(cell):
                    raise TypeError(
                        f"dynamic_costs[{t}][{y}][{x}] must be an int, "
                        f"got {type(cell).__name__}"
                    )
                if cell <= 0:
                    raise ValueError(
                        f"dynamic_costs[{t}][{y}][{x}] must be a positive "
                        f"integer, got {cell}"
                    )
                parsed.append(cell)
            rows.append(tuple(parsed))
        frames.append(tuple(rows))
    return tuple(frames)


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


def _normalize_starts(starts):
    # ``starts`` is the plan_multi_start counterpart of ``plan``'s single
    # ``start``. This performs the outer/structure pass in exactly the
    # position where ``plan`` structurally normalizes ``start``: strings,
    # bytes, ``None`` and other non-sequences are ``TypeError`` and an
    # empty sequence is a ``ValueError``; every element is normalized to
    # a coordinate tuple (element shape/coordinate type failures are
    # ``TypeError``). Bounds and obstacle checks follow in
    # ``plan_multi_start`` at the same relative positions ``plan`` uses
    # for its single start. Duplicates merge and the tuple is sorted so
    # the input order never affects the result.
    if (starts is None or isinstance(starts, (str, bytes))
            or not isinstance(starts, Sequence)):
        raise TypeError(
            f"starts must be a non-empty sequence of grid coordinates, "
            f"got {type(starts).__name__}"
        )
    if len(starts) == 0:
        raise ValueError("starts must contain at least one start")
    points = set()
    for index, item in enumerate(starts):
        point = _normalize_point(item, f"starts[{index}]")
        points.add(point)
    return tuple(sorted(points))


def _normalize_requests(requests):
    # ``requests`` is the plan_batch counterpart of ``plan``'s
    # ``start``/``goal`` pair. This performs the outer/structure pass in
    # exactly the position where ``plan`` structurally normalizes its
    # endpoints: strings, bytes, ``None`` and other non-sequences are
    # ``TypeError`` and an empty sequence is a ``ValueError``. Every
    # element must be a [start, goal] pair of grid coordinates (element
    # shape or coordinate type failures are ``TypeError``); bounds and
    # obstacle checks follow in ``plan_batch`` at the same relative
    # positions ``plan`` uses for its endpoints. The input order and
    # duplicate requests are preserved: every element gets its own
    # independent result.
    if (requests is None or isinstance(requests, (str, bytes))
            or not isinstance(requests, Sequence)):
        raise TypeError(
            f"requests must be a non-empty sequence of [start, goal] "
            f"coordinate pairs, got {type(requests).__name__}"
        )
    if len(requests) == 0:
        raise ValueError("requests must contain at least one query")
    pairs = []
    for index, item in enumerate(requests):
        name = f"requests[{index}]"
        if isinstance(item, (str, bytes)) or not isinstance(item, Sequence):
            raise TypeError(
                f"{name} must be a [start, goal] pair of grid "
                f"coordinates, got {type(item).__name__}"
            )
        if len(item) != 2:
            raise TypeError(
                f"{name} must contain exactly two coordinates, "
                f"got {len(item)}"
            )
        start = _normalize_point(item[0], f"{name} start")
        goal = _normalize_point(item[1], f"{name} goal")
        pairs.append((start, goal))
    return pairs


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


def _validate_allow_wait(allow_wait):
    # The shared waiting-flag rule used by every planner and ``replay``:
    # a non-bool raises ``TypeError`` after the ``dynamic_blocked`` checks
    # and before the budget and snapshot checks.
    if not isinstance(allow_wait, bool):
        raise TypeError(
            f"allow_wait must be a bool, got {type(allow_wait).__name__}"
        )


def _validate_max_cost(max_cost):
    # The shared non-negative-integer cost-limit rules used by ``plan``
    # and ``plan_any`` (and by ``resume`` for a checkpoint field): other
    # types raise ``TypeError`` and negative values raise ``ValueError``.
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


def _status_for(budget_stop, cost_limited):
    # The failure status when a limit is in play: a budget stop always
    # wins, then a cost-limit discard, otherwise plain unreachability.
    if budget_stop:
        return "budget_exhausted"
    if cost_limited:
        return "cost_exhausted"
    return "unreachable"


def _search_static(width, height, obstacles, start, goal, costs, trace,
                   max_expanded, max_cost=None, snapshot=False,
                   allow_wait=False, state=None):
    """Classic static-grid A* (the ``plan`` search without frames).

    Cells are merged by coordinate: the best known ``g`` per cell is kept
    in ``g_score``, stale heap entries are skipped, and ``came_from``
    reconstructs the cheapest route once the goal is closed. With
    ``max_cost`` a candidate whose accumulated entering-cell cost would
    exceed the limit is discarded instead of queued, and the search
    remembers whether any discard happened (``cost_limited``) so a
    failure can be reported as ``cost_exhausted``; the flag is part of
    the snapshot state so a resumed search stays equivalent to one
    uninterrupted call. With
    ``snapshot=True`` a budget stop additionally returns a checkpoint of
    the complete search state; ``state`` carries such a snapshot back in
    so ``resume`` continues the identical traversal. The snapshot needs
    the trace even when the caller did not ask for it, so nodes are
    recorded whenever ``trace`` or ``snapshot`` is true.
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
        cost_limited = False
    else:
        open_heap = state["open"]
        came_from = state["came_from"]
        g_score = state["g_score"]
        closed = state["closed"]
        expanded_nodes = state["expanded_nodes"]
        cost_limited = state["cost_limited"]
    budget_stop = False
    has_status = max_expanded is not None or max_cost is not None

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
            if has_status:
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
                    # The candidate's accumulated cost exceeds the limit:
                    # discard it and remember that a discard happened.
                    cost_limited = True
                    continue
                g_score[nxt] = new_g
                came_from[nxt] = current
                h = heuristic(nxt)
                heapq.heappush(
                    open_heap, (new_g + h, h, nxt[0], nxt[1], nxt)
                )
    result = {"path": None, "cost": None, "expanded": len(closed)}
    if has_status:
        result["status"] = _status_for(budget_stop, cost_limited)
    if trace:
        result["expanded_nodes"] = expanded_nodes
    if snapshot and result.get("status") == "budget_exhausted":
        result["checkpoint"] = _snapshot_checkpoint(
            "plan", width, height, obstacles, start, goal, costs, None,
            max_cost, len(closed), expanded_nodes,
            _static_snapshot_state(open_heap, came_from, g_score, closed,
                                   cost_limited, max_cost),
            allow_wait,
        )
    return result


def _search_dynamic(width, height, obstacles, frames, start, goal, costs,
                    trace, max_expanded, max_cost=None, snapshot=False,
                    allow_wait=False, state=None, cost_frames=None):
    """History-sensitive time-expanded A*.

    Frame ``t`` constrains the cell occupied at path index ``t``; frames
    past the last one reuse the last frame. ``frames`` may be ``None``
    when ``cost_frames`` is given: time-varying costs alone already make
    the search space time-expanded. With ``cost_frames`` the price of
    entering a cell at time ``t`` is that frame's cell value (frames past
    the last one reuse the last frame); the start cell is never counted.
    Waiting in place (unless ``allow_wait`` is set) and repeated
    coordinates are forbidden, so
    whether a candidate route can be
    extended depends on the exact sequence of cells it already visited:
    two routes reaching the same ``(x, y, t)`` with different histories
    are distinct states and neither may prune the other (the route with
    the larger accumulated cost can be the only one that remains
    extendable). Each heap node therefore carries its complete path; the
    search is a best-first traversal of the feasible route tree. Goal
    closings are collected until the heap's smallest f exceeds the best
    goal cost, after which the minimum-cost goal tie-break is settled by
    the fixed f, h, x, y, t priority and then the complete path's
    lexicographic order. With ``max_cost`` a candidate route whose
    accumulated entering-cell cost would exceed the limit is discarded
    (``cost_limited`` records that any discard happened, so a failure can
    be reported as ``cost_exhausted``; the flag is part of the snapshot
    state). With ``allow_wait`` a route may also stay in its current cell
    for one frame at no entering-cell cost, generated only while the next
    time step still lies within the provided frames and the cell is free
    there; consecutive waits repeat the coordinate in the path, and once
    a cell is left it may never be re-entered. With ``snapshot=True`` a
    budget stop additionally
    returns a checkpoint of the complete route-tree state; ``state``
    carries such a snapshot back in for ``resume``.
    """

    def heuristic(point):
        return abs(point[0] - goal[0]) + abs(point[1] - goal[1])

    last_frame = len(frames) - 1 if frames is not None else None
    last_cost_frame = len(cost_frames) - 1 if cost_frames is not None else None

    def step_cost(point, t):
        # Cost of entering ``point`` at time ``t``; the start cell is
        # never entered. Time-varying frames take the value of frame
        # ``t`` (the last frame persists); otherwise the static ``costs``
        # matrix or unit steps apply.
        if cost_frames is not None:
            frame = (cost_frames[t] if t <= last_cost_frame
                     else cost_frames[last_cost_frame])
            return frame[point[1]][point[0]]
        if costs is None:
            return 1
        return costs[point[1]][point[0]]

    def frame_cells(t):
        if frames is None:
            return frozenset()
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
        cost_limited = False
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
    else:
        open_heap = state["open"]
        closed_count = state["closed_count"]
        expanded_nodes = state["expanded_nodes"]
        best_key = state["best_key"]
        cost_limited = state["cost_limited"]
    budget_stop = False
    has_status = max_expanded is not None or max_cost is not None

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
            new_g = g + step_cost(nxt, next_t)
            if max_cost is not None and new_g > max_cost:
                # The candidate's accumulated cost exceeds the limit:
                # discard it and remember that a discard happened.
                cost_limited = True
                continue
            new_path = path + (nxt,)
            h = heuristic(nxt)
            heapq.heappush(
                open_heap,
                (new_g + h, h, nx, ny, next_t, new_g, new_path,
                 seen | {nxt}),
            )
        if (allow_wait and frames is not None and next_t <= last_frame
                and (x, y) not in frame):
            # Waiting in place: the route stays one more frame at no
            # entering-cell cost. Only generated while the next time step
            # still lies within the provided frames (after the last frame
            # moves reuse it but no waits are generated), and only while
            # the cell itself is free at that next frame. The coordinate
            # is already in ``seen``, so the repeat check is untouched.
            wait_path = path + ((x, y),)
            h = heuristic((x, y))
            heapq.heappush(
                open_heap,
                (g + h, h, x, y, next_t, g, wait_path, seen),
            )
    if best_key is None:
        result = {"path": None, "cost": None, "expanded": closed_count}
        if has_status:
            result["status"] = _status_for(budget_stop, cost_limited)
    else:
        best_g, _, best_path = best_key
        result = {
            "path": list(best_path),
            "cost": best_g,
            "expanded": closed_count,
        }
        if has_status:
            result["status"] = "found"
    if trace:
        result["expanded_nodes"] = expanded_nodes
    if snapshot and result.get("status") == "budget_exhausted":
        result["checkpoint"] = _snapshot_checkpoint(
            "plan", width, height, obstacles, start, goal, costs, frames,
            max_cost, closed_count, expanded_nodes,
            _tree_snapshot_state(open_heap, best_key, "plan",
                                 cost_limited, max_cost),
            allow_wait, cost_frames,
        )
    return result


def _search_any(width, height, obstacles, frames, start, goals, costs,
                trace, max_expanded, max_cost=None, snapshot=False,
                allow_wait=False, state=None, cost_frames=None):
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
    there, exactly as ``plan`` does; providing ``cost_frames`` (with or
    without obstacle frames) makes the search time-expanded, so the time
    dimension advances and the trace records ``(x, y, t)`` triples. With ``max_cost`` a candidate route
    whose accumulated entering-cell cost would exceed the limit is
    discarded (``cost_limited`` records that any discard happened, so a
    failure can be reported as ``cost_exhausted``; the flag is part of
    the snapshot state). With ``allow_wait`` a dynamic route may also stay
    in its current cell for one frame at no entering-cell cost, generated
    only while the next time step still lies within the provided frames
    and the cell is free there; consecutive waits repeat the coordinate
    in the path, and once a cell is left it may never be re-entered.
    With ``snapshot=True`` a budget stop
    additionally returns a checkpoint of the complete route-tree state;
    ``state`` carries such a snapshot back in for ``resume``.
    """

    def heuristic(point):
        return min(
            abs(point[0] - goal[0]) + abs(point[1] - goal[1])
            for goal in goals
        )

    dynamic = frames is not None or cost_frames is not None
    last_frame = len(frames) - 1 if frames is not None else None
    last_cost_frame = len(cost_frames) - 1 if cost_frames is not None else None

    def step_cost(point, t):
        # Cost of entering ``point`` at time ``t``; the start cell is
        # never entered. Time-varying frames take the value of frame
        # ``t`` (the last frame persists); otherwise the static ``costs``
        # matrix or unit steps apply.
        if cost_frames is not None:
            frame = (cost_frames[t] if t <= last_cost_frame
                     else cost_frames[last_cost_frame])
            return frame[point[1]][point[0]]
        if costs is None:
            return 1
        return costs[point[1]][point[0]]

    def frame_cells(t):
        if frames is None:
            return frozenset()
        return frames[t] if t <= last_frame else frames[last_frame]

    # Heap entries are (f, h, x, y, t, g, path, seen), keyed exactly like
    # ``_search_dynamic``: the fixed numeric priority comes first and
    # candidates still tied are ordered by the complete coordinate path
    # lexicographically, so ordering never depends on goal order, obstacle
    # sets, in-frame coordinate order, or traversal order. Static mode
    # (neither obstacle frames nor cost frames) keeps t fixed at 0; the
    # frozenset ``seen`` mirrors ``path`` for an O(1) repeat check and is
    # never compared.
    if state is None:
        h0 = heuristic(start)
        start_path = (start,)
        open_heap = [(h0, h0, start[0], start[1], 0, 0, start_path,
                      frozenset(start_path))]
        closed_count = 0
        expanded_nodes = [] if trace or snapshot else None
        cost_limited = False
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
    else:
        open_heap = state["open"]
        closed_count = state["closed_count"]
        expanded_nodes = state["expanded_nodes"]
        best_key = state["best_key"]
        cost_limited = state["cost_limited"]
    budget_stop = False
    has_status = max_expanded is not None or max_cost is not None

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
            new_g = g + step_cost(nxt, next_t)
            if max_cost is not None and new_g > max_cost:
                # The candidate's accumulated cost exceeds the limit:
                # discard it and remember that a discard happened.
                cost_limited = True
                continue
            new_path = path + (nxt,)
            h = heuristic(nxt)
            heapq.heappush(
                open_heap,
                (new_g + h, h, nx, ny, next_t, new_g, new_path,
                 seen | {nxt}),
            )
        if (allow_wait and frames is not None and next_t <= last_frame
                and cell not in frame):
            # Waiting in place: one more frame at no entering-cell cost,
            # only within the provided frames and while the cell itself
            # is free at the next frame. The coordinate is already in
            # ``seen``, so the repeat check is untouched.
            wait_path = path + (cell,)
            h = heuristic(cell)
            heapq.heappush(
                open_heap,
                (g + h, h, x, y, next_t, g, wait_path, seen),
            )
    if best_key is None:
        result = {"path": None, "cost": None, "expanded": closed_count}
        if has_status:
            result["status"] = _status_for(budget_stop, cost_limited)
    else:
        best_g, _, best_path = best_key
        result = {
            "path": list(best_path),
            "cost": best_g,
            "expanded": closed_count,
        }
        if has_status:
            result["status"] = "found"
    if trace:
        result["expanded_nodes"] = expanded_nodes
    if snapshot and result.get("status") == "budget_exhausted":
        result["checkpoint"] = _snapshot_checkpoint(
            "plan_any", width, height, obstacles, start, goals, costs,
            frames, max_cost, closed_count, expanded_nodes,
            _tree_snapshot_state(open_heap, best_key, "plan_any",
                                 cost_limited, max_cost),
            allow_wait, cost_frames,
        )
    return result


def _search_multi_static(width, height, obstacles, starts, goal, costs,
                         trace, max_expanded, max_cost=None, snapshot=False,
                         allow_wait=False, state=None):
    """Multi-source static A* over the (cost, path) lexicographic objective.

    This is the ``plan_multi_start`` search without frames. Every start is
    a root with ``g == 0`` and cells are merged by coordinate: each cell
    keeps only its best known ``(g, path)`` pair -- smallest accumulated
    entering-cell cost first, then the lexicographically smallest complete
    coordinate sequence. That ordering is preserved when two candidate
    prefixes are extended by the same suffix, so a dominated candidate can
    never win later and each coordinate is closed at most once (the first
    pop whose entry still matches the cell's best pair). A cell whose best
    pair improves after it was closed is relaxed again from the improved
    entry without being closed a second time, which keeps the recorded
    best pairs exact everywhere. The goal is never expanded past: a route
    ends the moment it reaches the goal. The goal's best pair only counts
    once the goal has actually been closed (exactly as the route-tree
    searches only record a goal closing), and the search stops once the
    heap's smallest f exceeds the closed goal's cost: f never decreases
    along a route, so no pending candidate can still reach the goal at
    that cost or less. The result is therefore the minimum-cost route,
    ties decided by the complete coordinate path lexicographically, and
    ``expanded`` counts every first-time closing (the unique coordinate
    closing sequence recorded by ``trace``). With ``max_cost`` a candidate
    whose accumulated entering-cell cost would exceed the limit is
    discarded instead of queued (``cost_limited`` records that any discard
    happened, so a failure can be reported as ``cost_exhausted``; the flag
    is part of the snapshot state). With ``snapshot=True`` a budget stop
    additionally returns a checkpoint of the complete search state;
    ``state`` carries such a snapshot back in so ``resume`` continues the
    identical traversal. The snapshot needs the trace even when the caller
    did not ask for it, so nodes are recorded whenever ``trace`` or
    ``snapshot`` is true.
    """

    def heuristic(point):
        return abs(point[0] - goal[0]) + abs(point[1] - goal[1])

    def step_cost(point):
        # Cost of entering ``point``; the start cell is never entered.
        if costs is None:
            return 1
        return costs[point[1]][point[0]]

    # Heap entries are (f, h, x, y, path, g): the fixed numeric priority
    # comes first and candidates still tied (same cell, same f) are ordered
    # by the complete coordinate path lexicographically, so ordering never
    # depends on the input order of ``starts`` or the traversal order. The
    # trailing ``g`` is never compared (equal keys up to ``path`` imply
    # equal ``g``).
    if state is None:
        open_heap = []
        best = {}
        for start in starts:
            h0 = heuristic(start)
            start_path = (start,)
            best[start] = (0, start_path)
            open_heap.append((h0, h0, start[0], start[1], start_path, 0))
        heapq.heapify(open_heap)
        closed = set()
        expanded_nodes = [] if trace or snapshot else None
        cost_limited = False
    else:
        open_heap = state["open"]
        best = state["best"]
        closed = state["closed"]
        expanded_nodes = state["expanded_nodes"]
        cost_limited = state["cost_limited"]
    budget_stop = False
    has_status = max_expanded is not None or max_cost is not None

    while open_heap:
        goal_best = best.get(goal) if goal in closed else None
        if goal_best is not None and open_heap[0][0] > goal_best[0]:
            break
        if max_expanded is not None and len(closed) >= max_expanded:
            # Live candidates remain but the budget is spent; close
            # nothing more (this also covers a start equal to the goal
            # with a zero budget: no node is closed). A goal already
            # closed still wins below.
            budget_stop = True
            break
        _f, _h, x, y, path, g = heapq.heappop(open_heap)
        cell = (x, y)
        if best.get(cell) != (g, path):
            continue  # stale entry; a better route superseded it
        if cell not in closed:
            closed.add(cell)
            if expanded_nodes is not None:
                expanded_nodes.append(cell)
        if cell == goal:
            continue  # routes end at the goal; never expanded past it
        for dx, dy in _NEIGHBORS:
            nxt = (x + dx, y + dy)
            if not (0 <= nxt[0] < width and 0 <= nxt[1] < height):
                continue
            if nxt in obstacles:
                continue
            new_g = g + step_cost(nxt)
            if max_cost is not None and new_g > max_cost:
                # The candidate's accumulated cost exceeds the limit:
                # discard it and remember that a discard happened.
                cost_limited = True
                continue
            new_path = path + (nxt,)
            known = best.get(nxt)
            if (known is None or new_g < known[0]
                    or (new_g == known[0] and new_path < known[1])):
                best[nxt] = (new_g, new_path)
                h = heuristic(nxt)
                heapq.heappush(
                    open_heap, (new_g + h, h, nxt[0], nxt[1], new_path, new_g)
                )
    goal_best = best.get(goal) if goal in closed else None
    if goal_best is None:
        result = {"path": None, "cost": None, "expanded": len(closed)}
        if has_status:
            result["status"] = _status_for(budget_stop, cost_limited)
    else:
        best_g, best_path = goal_best
        result = {
            "path": list(best_path),
            "cost": best_g,
            "expanded": len(closed),
        }
        if has_status:
            result["status"] = "found"
    if trace:
        result["expanded_nodes"] = expanded_nodes
    if snapshot and result.get("status") == "budget_exhausted":
        result["checkpoint"] = _snapshot_checkpoint_multi(
            width, height, obstacles, starts, goal, costs, None,
            max_cost, len(closed), expanded_nodes,
            _multi_static_snapshot_state(open_heap, best, closed,
                                         cost_limited, max_cost),
            allow_wait,
        )
    return result


def _search_multi_dynamic(width, height, obstacles, frames, starts, goal,
                          costs, trace, max_expanded, max_cost=None,
                          snapshot=False, allow_wait=False, state=None,
                          cost_frames=None):
    """History-sensitive time-expanded multi-source A*.

    This is the ``plan_multi_start`` counterpart of ``_search_dynamic``:
    every start is a root at ``t == 0`` with ``g == 0`` and the same
    route-tree rules apply (the persistent last frame, no waiting unless
    ``allow_wait`` is set, no
    repeated coordinates, and routes reaching the same ``(x, y, t)``
    through different coordinate histories are distinct states that are
    never merged). ``frames`` may be ``None`` when ``cost_frames`` is
    given: time-varying costs alone already make the search space
    time-expanded. With ``cost_frames`` the price of entering a cell at
    time ``t`` is that frame's cell value (frames past the last one
    reuse the last frame); the start cell is never counted. Goal
    closings are collected until the heap's smallest f
    exceeds the best goal cost; the winner is the minimum-cost route, ties
    decided by the complete coordinate path lexicographically. With
    ``max_cost`` a candidate route whose accumulated entering-cell cost
    would exceed the limit is discarded (``cost_limited`` records that any
    discard happened, so a failure can be reported as ``cost_exhausted``;
    the flag is part of the snapshot state). With ``allow_wait`` a route
    may also stay in its current cell for one frame at no entering-cell
    cost, generated only while the next time step still lies within the
    provided frames and the cell is free there; consecutive waits repeat
    the coordinate in the path, and once a cell is left it may never be
    re-entered. With ``snapshot=True`` a
    budget stop additionally returns a checkpoint of the complete
    route-tree state; ``state`` carries such a snapshot back in for
    ``resume``.
    """

    def heuristic(point):
        return abs(point[0] - goal[0]) + abs(point[1] - goal[1])

    last_frame = len(frames) - 1 if frames is not None else None
    last_cost_frame = (len(cost_frames) - 1
                       if cost_frames is not None else None)

    def step_cost(point, t):
        # Cost of entering ``point`` at time ``t``; the start cell is
        # never entered. Time-varying frames take the value of frame
        # ``t`` (the last frame persists); otherwise the static ``costs``
        # matrix or unit steps apply.
        if cost_frames is not None:
            frame = (cost_frames[t] if t <= last_cost_frame
                     else cost_frames[last_cost_frame])
            return frame[point[1]][point[0]]
        if costs is None:
            return 1
        return costs[point[1]][point[0]]

    def frame_cells(t):
        if frames is None:
            return frozenset()
        return frames[t] if t <= last_frame else frames[last_frame]

    # Heap entries are (f, h, x, y, t, g, path, seen), keyed exactly like
    # ``_search_dynamic``: the fixed numeric priority comes first and
    # candidates still tied are ordered by the complete coordinate path
    # lexicographically, so ordering never depends on the input order of
    # ``starts``, the obstacle sets, in-frame coordinate order, or the
    # traversal order. The frozenset ``seen`` mirrors ``path`` for an O(1)
    # repeat check and is never compared.
    if state is None:
        open_heap = []
        for start in starts:
            h0 = heuristic(start)
            start_path = (start,)
            open_heap.append((h0, h0, start[0], start[1], 0, 0, start_path,
                              frozenset(start_path)))
        heapq.heapify(open_heap)
        closed_count = 0
        expanded_nodes = [] if trace or snapshot else None
        cost_limited = False
        # Best goal closing seen so far, keyed as (g, path): minimum total
        # cost first, then the complete route's lexicographic order.
        # Closing a goal does not stop the search immediately: an equally
        # cheap route whose chain runs through higher-h nodes might still
        # be undeveloped. Since each move costs at least 1 and the
        # Manhattan h changes by at most 1 per move, f = g + h never
        # decreases along a route; once the smallest f on the heap exceeds
        # the best goal cost, no route of that cost (or less) can remain
        # undiscovered.
        best_key = None
    else:
        open_heap = state["open"]
        closed_count = state["closed_count"]
        expanded_nodes = state["expanded_nodes"]
        best_key = state["best_key"]
        cost_limited = state["cost_limited"]
    budget_stop = False
    has_status = max_expanded is not None or max_cost is not None

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
            goal_key = (g, path)
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
            new_g = g + step_cost(nxt, next_t)
            if max_cost is not None and new_g > max_cost:
                # The candidate's accumulated cost exceeds the limit:
                # discard it and remember that a discard happened.
                cost_limited = True
                continue
            new_path = path + (nxt,)
            h = heuristic(nxt)
            heapq.heappush(
                open_heap,
                (new_g + h, h, nx, ny, next_t, new_g, new_path,
                 seen | {nxt}),
            )
        if (allow_wait and frames is not None and next_t <= last_frame
                and (x, y) not in frame):
            # Waiting in place: one more frame at no entering-cell cost,
            # only within the provided frames and while the cell itself
            # is free at the next frame. The coordinate is already in
            # ``seen``, so the repeat check is untouched.
            wait_path = path + ((x, y),)
            h = heuristic((x, y))
            heapq.heappush(
                open_heap,
                (g + h, h, x, y, next_t, g, wait_path, seen),
            )
    if best_key is None:
        result = {"path": None, "cost": None, "expanded": closed_count}
        if has_status:
            result["status"] = _status_for(budget_stop, cost_limited)
    else:
        best_g, best_path = best_key
        result = {
            "path": list(best_path),
            "cost": best_g,
            "expanded": closed_count,
        }
        if has_status:
            result["status"] = "found"
    if trace:
        result["expanded_nodes"] = expanded_nodes
    if snapshot and result.get("status") == "budget_exhausted":
        result["checkpoint"] = _snapshot_checkpoint_multi(
            width, height, obstacles, starts, goal, costs, frames,
            max_cost, closed_count, expanded_nodes,
            _multi_tree_snapshot_state(open_heap, best_key, cost_limited,
                                       max_cost),
            allow_wait, cost_frames,
        )
    return result


def _search_k(width, height, obstacles, frames, start, goal, costs, k,
              max_expanded=None, max_cost=None, allow_wait=False,
              cost_frames=None):
    """Best-first traversal of the feasible route tree keeping the k best.

    This is the ``plan_k`` search. As in ``_search_dynamic`` and
    ``_search_any`` the feasibility of a route depends on its exact
    coordinate history (no waiting unless ``allow_wait`` is set, no
    repeated coordinates, and dynamic
    frames), so every node carries its complete path and routes reaching
    the same cell through different histories are never merged; this also
    makes every complete coordinate sequence a distinct candidate and
    keeps static-mode ordering consistent with ``plan``'s path
    lexicographic tie-break. With ``allow_wait`` a route may also stay in
    its current cell for one frame at no entering-cell cost, generated
    only while the next time step still lies within the provided frames
    and the cell is free there; consecutive waits repeat the coordinate
    in the path, and once a cell is left it may never be re-entered.
    With ``cost_frames`` the price of entering a cell at time ``t`` is
    that frame's cell value (frames past the last one reuse the last
    frame); time-varying costs alone already make the traversal
    time-expanded, and the start cell is never counted.

    Heap entries are ``(f, path, x, y, t, g, seen)``: the total priority
    is ``f`` first and then the complete coordinate path lexicographically,
    which is exactly the ordering ``plan_k`` reports (total cost, and at
    the goal ``f == g``). Every move costs at least 1 and the Manhattan
    heuristic changes by at most 1 per move, so ``f`` never decreases
    along a route and a child path extends its parent's; the traversal
    therefore closes goal routes in reported ranking order, and it stops
    the moment the k-th goal route has been closed: every still-pending
    candidate (and every extension of one) can only reach the goal at an
    equal-or-larger rank. ``expanded`` counts only the route candidates
    actually popped and closed, once per complete history; candidates
    filtered at generation and candidates left pending are never counted.

    Optional limits: ``max_cost`` discards at generation time every
    candidate whose accumulated entering-cell cost would exceed it (a
    candidate exactly at the limit still competes; the start costs 0 and
    is never checked), remembering whether any discard happened so a
    short result can be reported as ``cost_exhausted``. ``max_expanded``
    caps the number of route candidates closed: once the count reaches
    the limit the loop stops with live candidates still pending, which a
    short result reports as ``budget_exhausted`` (a budget stop wins over
    a cost-limit stop). Neither limit ever changes the ordering or the
    routes already determined: a candidate is only dropped or left
    pending, so the returned prefix is exactly the corresponding prefix
    of the unlimited result. When either limit is given the result
    carries ``status``; without them the shape is exactly the legacy
    ``{"paths", "costs", "expanded"}``.
    """

    def heuristic(point):
        return abs(point[0] - goal[0]) + abs(point[1] - goal[1])

    dynamic = frames is not None or cost_frames is not None
    last_frame = len(frames) - 1 if frames is not None else None
    last_cost_frame = (len(cost_frames) - 1
                       if cost_frames is not None else None)

    def step_cost(point, t):
        # Cost of entering ``point`` at time ``t``; the start cell is
        # never entered. Time-varying frames take the value of frame
        # ``t`` (the last frame persists); otherwise the static ``costs``
        # matrix or unit steps apply.
        if cost_frames is not None:
            frame = (cost_frames[t] if t <= last_cost_frame
                     else cost_frames[last_cost_frame])
            return frame[point[1]][point[0]]
        if costs is None:
            return 1
        return costs[point[1]][point[0]]

    def frame_cells(t):
        if frames is None:
            return frozenset()
        return frames[t] if t <= last_frame else frames[last_frame]

    # ``path`` is unique per candidate (revisits are forbidden and waits
    # only extend a route's own tail), so ``(f, path)`` is already a total
    # order and the later
    # fields are never compared. The frozenset ``seen`` mirrors ``path``
    # for an O(1) repeat check.
    h0 = heuristic(start)
    start_path = (start,)
    open_heap = [(h0, start_path, start[0], start[1], 0, 0,
                  frozenset(start_path))]
    expanded = 0
    found_paths = []
    found_costs = []
    cost_limited = False  # any candidate discarded for exceeding max_cost
    budget_stop = False
    has_status = max_expanded is not None or max_cost is not None

    while open_heap and len(found_paths) < k:
        if max_expanded is not None and expanded >= max_expanded:
            # Live route candidates remain but the close budget is spent;
            # close nothing more. Any goal routes already closed keep
            # their rank below.
            budget_stop = True
            break
        _, path, x, y, t, g, seen = heapq.heappop(open_heap)
        expanded += 1  # every popped route candidate closes exactly once
        cell = (x, y)
        if cell == goal:
            # Routes end at the goal; never expanded past one. Goal
            # closings leave the heap in the exact ranking the result
            # uses: total cost first, then the complete sequence order.
            found_paths.append(list(path))
            found_costs.append(g)
            continue
        next_t = t + 1 if dynamic else 0
        frame = frame_cells(next_t) if dynamic else frozenset()
        for dx, dy in _NEIGHBORS:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            nxt = (nx, ny)
            if nxt in obstacles or nxt in frame or nxt in seen:
                continue  # static obstacle, timed obstacle, or revisit
            new_g = g + step_cost(nxt, next_t)
            if max_cost is not None and new_g > max_cost:
                # The candidate's accumulated cost exceeds the limit:
                # discard it the moment it is generated and remember
                # that a discard happened. The start (g == 0) is never
                # generated here and so is never affected.
                cost_limited = True
                continue
            new_path = path + (nxt,)
            h = heuristic(nxt)
            heapq.heappush(
                open_heap,
                (new_g + h, new_path, nx, ny, next_t, new_g,
                 seen | {nxt}),
            )
        if (allow_wait and frames is not None and next_t <= last_frame
                and cell not in frame):
            # Waiting in place: one more frame at no entering-cell cost,
            # only within the provided frames and while the cell itself
            # is free at the next frame. The coordinate is already in
            # ``seen``, so the repeat check is untouched.
            wait_path = path + (cell,)
            h = heuristic(cell)
            heapq.heappush(
                open_heap,
                (g + h, wait_path, x, y, next_t, g, seen),
            )
    result = {"paths": found_paths, "costs": found_costs, "expanded": expanded}
    if has_status:
        if len(found_paths) >= k:
            result["status"] = "found"
        elif budget_stop:
            # The budget cut the search short with candidates still
            # pending; this takes priority over the cost-limit status.
            result["status"] = "budget_exhausted"
        elif found_paths:
            # The heap drained naturally (no budget truncation) and at
            # least one ranked route was closed, just fewer than k.
            result["status"] = "found"
        elif cost_limited:
            # No route was found, but at least one candidate had to be
            # dropped for exceeding the cost limit while no budget cut
            # the traversal short.
            result["status"] = "cost_exhausted"
        else:
            result["status"] = "unreachable"
    return result


def _search_distance_field(width, height, obstacles, goal, costs, trace):
    """Reverse Dijkstra from the goal over the static grid.

    This is the ``distance_field`` search. Cells are merged by coordinate:
    the best known distance per cell is kept in ``best``, stale heap
    entries are skipped, and every reachable cell is closed exactly once
    (the goal included). Because a move's price is the cost of the cell
    entered, the distance of a cell is the minimum over its neighbors of
    ``distance(neighbor) + cost(neighbor)``; expanding a closed cell
    therefore relaxes each neighbor with the closed cell's own entering
    cost, and a reachable ``plan`` start yields exactly the cost ``plan``
    reports. Heap entries are ``(distance, x, y, point)``: the distance
    and the coordinate give a total, input-order-independent closing
    order, so the point itself is never compared.
    """

    def step_cost(point):
        # Cost of entering ``point``.
        if costs is None:
            return 1
        return costs[point[1]][point[0]]

    distances = [[None] * width for _ in range(height)]
    open_heap = [(0, goal[0], goal[1], goal)]
    best = {goal: 0}
    closed = set()
    expanded_nodes = [] if trace else None

    while open_heap:
        dist, x, y, current = heapq.heappop(open_heap)
        if current in closed:
            continue  # stale heap entry; already closed with its best dist
        closed.add(current)
        distances[y][x] = dist
        if expanded_nodes is not None:
            expanded_nodes.append(current)
        step = step_cost(current)
        for dx, dy in _NEIGHBORS:
            nxt = (current[0] + dx, current[1] + dy)
            if not (0 <= nxt[0] < width and 0 <= nxt[1] < height):
                continue
            if nxt in obstacles or nxt in closed:
                continue
            new_dist = dist + step
            if new_dist < best.get(nxt, float("inf")):
                best[nxt] = new_dist
                heapq.heappush(
                    open_heap, (new_dist, nxt[0], nxt[1], nxt)
                )
    result = {"distances": distances, "expanded": len(closed)}
    if trace:
        result["expanded_nodes"] = expanded_nodes
    return result


def _search_distance_field_any(width, height, obstacles, goals, costs,
                               trace):
    """Reverse multi-source Dijkstra from all goals over the static grid.

    This is the ``distance_field_any`` search and the multi-goal
    counterpart of ``_search_distance_field``: every goal is seeded at
    distance 0 and one pass settles the minimum distance to *any*
    endpoint. Cells are merged by coordinate: the best known distance
    per cell is kept in ``best``, stale heap entries are skipped, and
    every reachable cell is closed exactly once (every goal included).
    As in the single-goal field, expanding a closed cell relaxes each
    neighbor with the closed cell's own entering cost, so a reachable
    ``plan_any`` start yields exactly the cost ``plan_any`` reports. The
    endpoint that supplies a cell's winning distance is recorded in
    ``owner`` -- goals own themselves -- and on an exact distance tie
    the endpoint with the lexicographically smallest coordinate wins,
    exactly the endpoint tie-break ``plan_any`` uses. Heap entries are
    ``(distance, x, y, point)``: distance then the coordinate give the
    total, input-order-independent closing order, and the owner is
    carried only by the entry's own field and never compared.
    """

    def step_cost(point):
        # Cost of entering ``point``.
        if costs is None:
            return 1
        return costs[point[1]][point[0]]

    goal_set = frozenset(goals)
    distances = [[None] * width for _ in range(height)]
    targets = [[None] * width for _ in range(height)]
    # Every goal starts at distance 0 and owns itself. Equal-distance
    # goals are popped in (x, y) order purely from the heap key, so
    # seeding order never influences the traversal.
    open_heap = [(0, goal[0], goal[1], goal) for goal in goals]
    heapq.heapify(open_heap)
    best = {goal: 0 for goal in goals}
    owner = {goal: goal for goal in goals}
    closed = set()
    expanded_nodes = [] if trace else None

    while open_heap:
        dist, x, y, current = heapq.heappop(open_heap)
        if current in closed:
            continue  # stale heap entry; already closed with its best dist
        closed.add(current)
        current_owner = owner[current]
        distances[y][x] = dist
        targets[y][x] = current_owner
        if expanded_nodes is not None:
            expanded_nodes.append(current)
        step = step_cost(current)
        for dx, dy in _NEIGHBORS:
            nxt = (current[0] + dx, current[1] + dy)
            if not (0 <= nxt[0] < width and 0 <= nxt[1] < height):
                continue
            if nxt in obstacles or nxt in closed:
                continue
            new_dist = dist + step
            old_dist = best.get(nxt)
            if old_dist is None:
                best[nxt] = new_dist
                owner[nxt] = current_owner
                heapq.heappush(
                    open_heap, (new_dist, nxt[0], nxt[1], nxt)
                )
            elif new_dist == old_dist:
                # An equal-distance route from a lexicographically
                # smaller endpoint wins the cell; the owner only changes
                # before the cell is closed.
                if current_owner < owner[nxt]:
                    owner[nxt] = current_owner
            elif new_dist < old_dist:
                best[nxt] = new_dist
                owner[nxt] = current_owner
                heapq.heappush(
                    open_heap, (new_dist, nxt[0], nxt[1], nxt)
                )
    result = {
        "distances": distances,
        "targets": targets,
        "expanded": len(closed),
    }
    if trace:
        result["expanded_nodes"] = expanded_nodes
    return result


def _snapshot_checkpoint(planner, width, height, obstacles, start, endpoints,
                         costs, frames, max_cost, closed_count,
                         expanded_nodes, state, allow_wait=False,
                         cost_frames=None):
    # The checkpoint is built from JSON-native values only (ints, strings,
    # lists, dicts, ``None``) in one fixed key order, and every list
    # derived from a set is sorted, so neither set iteration order, goal
    # order nor in-frame coordinate order can influence the result. The
    # ``max_cost`` field is only present when a cost limit was provided,
    # ``dynamic_costs`` is only present when time-varying costs were
    # provided, and ``allow_wait`` is only recorded when waiting was
    # enabled, so checkpoints of calls that do not use them keep their
    # exact previous shape.
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
    if cost_frames is not None:
        checkpoint["dynamic_costs"] = [
            [list(row) for row in frame] for frame in cost_frames
        ]
    if max_cost is not None:
        checkpoint["max_cost"] = max_cost
    if allow_wait:
        checkpoint["allow_wait"] = True
    checkpoint["closed"] = closed_count
    checkpoint["trace"] = [list(entry) for entry in expanded_nodes]
    checkpoint["state"] = state
    return checkpoint


def _static_snapshot_state(open_heap, came_from, g_score, closed,
                           cost_limited, max_cost):
    # The complete static-A* state: the heap array as it stands (restoring
    # it verbatim keeps the exact future pop order), the route
    # reconstruction links, the best-known costs and the closed set. The
    # cost-discard flag is only recorded when a cost limit is in play.
    state = {
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
    if max_cost is not None:
        state["cost_limited"] = cost_limited
    return state


def _tree_snapshot_state(open_heap, best_key, planner, cost_limited,
                         max_cost):
    # The complete route-tree state: every live candidate with its full
    # coordinate history (the ``seen`` frozenset is derived from the path
    # on restore) plus the best goal closing recorded so far, keyed as
    # (g, t, path) for ``plan`` and (g, (x, y), path) for ``plan_any``.
    # The cost-discard flag is only recorded when a cost limit is in play.
    if best_key is None:
        best = None
    else:
        best_g, best_tie, best_path = best_key
        best = [
            best_g,
            [best_tie[0], best_tie[1]] if planner == "plan_any" else best_tie,
            [[point[0], point[1]] for point in best_path],
        ]
    state = {
        "open": [
            [f, h, x, y, t, g, [[point[0], point[1]] for point in path]]
            for f, h, x, y, t, g, path, _seen in open_heap
        ],
        "best": best,
    }
    if max_cost is not None:
        state["cost_limited"] = cost_limited
    return state


def _snapshot_checkpoint_multi(width, height, obstacles, starts, goal, costs,
                               frames, max_cost, closed_count, expanded_nodes,
                               state, allow_wait=False, cost_frames=None):
    # The ``plan_multi_start`` checkpoint: like ``_snapshot_checkpoint``
    # but with the sorted start tuple in place of the single start. Only
    # JSON-native values are used and every set-derived list is sorted, so
    # neither obstacle-set iteration order, start order nor in-frame
    # coordinate order can influence the result. The ``max_cost`` field is
    # only present when a cost limit was provided, ``dynamic_costs`` is
    # only present when time-varying costs were provided, and
    # ``allow_wait`` is only recorded when waiting was enabled.
    checkpoint = {
        "version": _SNAPSHOT_VERSION,
        "planner": "plan_multi_start",
        "width": width,
        "height": height,
        "blocked": [[cell[0], cell[1]] for cell in sorted(obstacles)],
        "starts": [[start[0], start[1]] for start in starts],
        "goal": [goal[0], goal[1]],
    }
    checkpoint["costs"] = (
        [list(row) for row in costs] if costs is not None else None
    )
    checkpoint["dynamic_blocked"] = (
        [[[cell[0], cell[1]] for cell in sorted(frame)]
         for frame in frames]
        if frames is not None else None
    )
    if cost_frames is not None:
        checkpoint["dynamic_costs"] = [
            [list(row) for row in frame] for frame in cost_frames
        ]
    if max_cost is not None:
        checkpoint["max_cost"] = max_cost
    if allow_wait:
        checkpoint["allow_wait"] = True
    checkpoint["closed"] = closed_count
    checkpoint["trace"] = [list(entry) for entry in expanded_nodes]
    checkpoint["state"] = state
    return checkpoint


def _multi_static_snapshot_state(open_heap, best, closed, cost_limited,
                                 max_cost):
    # The complete multi-source static state: the heap array as it stands
    # (restoring it verbatim keeps the exact future pop order), the best
    # known (g, path) pair per cell and the closed set. The cost-discard
    # flag is only recorded when a cost limit is in play.
    state = {
        "open": [
            [f, h, x, y, g, [[point[0], point[1]] for point in path]]
            for f, h, x, y, path, g in open_heap
        ],
        "best": [
            [[cell[0], cell[1]], g, [[point[0], point[1]] for point in path]]
            for cell, (g, path) in sorted(best.items())
        ],
        "closed": [[cell[0], cell[1]] for cell in sorted(closed)],
    }
    if max_cost is not None:
        state["cost_limited"] = cost_limited
    return state


def _multi_tree_snapshot_state(open_heap, best_key, cost_limited, max_cost):
    # The complete multi-start route-tree state: every live candidate with
    # its full coordinate history (the ``seen`` frozenset is derived from
    # the path on restore) plus the best goal closing recorded so far,
    # keyed as (g, path). The cost-discard flag is only recorded when a
    # cost limit is in play.
    if best_key is None:
        best = None
    else:
        best_g, best_path = best_key
        best = [best_g, [[point[0], point[1]] for point in best_path]]
    state = {
        "open": [
            [f, h, x, y, t, g, [[point[0], point[1]] for point in path]]
            for f, h, x, y, t, g, path, _seen in open_heap
        ],
        "best": best,
    }
    if max_cost is not None:
        state["cost_limited"] = cost_limited
    return state


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
    last_frame = len(frames) - 1 if frames is not None else None
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
            if frames is not None:
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


def _restore_cost_limited(raw, max_cost):
    # The optional cost-discard flag inside a snapshot state: absent (the
    # only possibility in checkpoints written before cost limits existed,
    # and the shape produced whenever no limit is in play) means no
    # discard happened yet. A present flag must be a bool, and a recorded
    # discard is inconsistent with a checkpoint that carries no limit.
    cost_limited = raw.get("cost_limited", False)
    if not isinstance(cost_limited, bool):
        raise TypeError(
            f"checkpoint state cost_limited must be a bool, "
            f"got {type(cost_limited).__name__}"
        )
    if cost_limited and max_cost is None:
        raise ValueError(
            "checkpoint state records a cost discard without a cost limit"
        )
    return cost_limited


def _restore_static_state(raw, width, height, obstacles, start, goal,
                          trace, max_cost):
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
    cost_limited = _restore_cost_limited(raw, max_cost)
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
                    "checkpoint state g_score exceeds the cost limit"
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
        "cost_limited": cost_limited,
    }


def _restore_tree_state(raw, planner, dynamic, width, height, obstacles,
                        frames, start, goal, goals, costs, closed_count,
                        trace, max_cost, allow_wait=False, cost_frames=None):
    # Rebuild and fully cross-check a route-tree snapshot state (dynamic
    # ``plan``, and ``plan_any`` in both static and dynamic mode). Here
    # ``dynamic`` means the search is time-expanded, which timed obstacle
    # frames or time-varying cost frames both cause; only the obstacle
    # frames constrain cells and waits.
    if not isinstance(raw, dict):
        raise TypeError(
            f"checkpoint state must be an object, got {type(raw).__name__}"
        )
    for key in ("open", "best"):
        if key not in raw:
            raise TypeError(
                f"checkpoint state is missing required field {key!r}"
            )
    cost_limited = _restore_cost_limited(raw, max_cost)

    def heuristic(point):
        if planner == "plan":
            return abs(point[0] - goal[0]) + abs(point[1] - goal[1])
        return min(
            abs(point[0] - end[0]) + abs(point[1] - end[1])
            for end in goals
        )

    last_cost_frame = (len(cost_frames) - 1
                       if cost_frames is not None else None)

    def step_cost(point, t):
        # Cost of entering ``point`` at path index ``t``; the start cell
        # is never entered. Time-varying frames take the value of frame
        # ``t`` (the last frame persists); otherwise the static ``costs``
        # matrix or unit steps apply.
        if cost_frames is not None:
            frame = (cost_frames[t] if t <= last_cost_frame
                     else cost_frames[last_cost_frame])
            return frame[point[1]][point[0]]
        if costs is None:
            return 1
        return costs[point[1]][point[0]]

    last_frame = len(frames) - 1 if frames is not None else None

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
            wait = previous is not None and cell == previous
            if wait:
                # A consecutive repeat is only a legal wait when the
                # checkpoint recorded ``allow_wait`` and the stay still
                # lies within the provided frames.
                if not (allow_wait and frames is not None):
                    raise ValueError(
                        "checkpoint candidate path repeats a coordinate"
                    )
                if index > last_frame:
                    raise ValueError(
                        "checkpoint candidate waits beyond the final frame"
                    )
            elif cell in seen:
                raise ValueError(
                    "checkpoint candidate path repeats a coordinate"
                )
            elif previous is not None:
                if (abs(cell[0] - previous[0])
                        + abs(cell[1] - previous[1])) != 1:
                    raise ValueError(
                        "checkpoint candidate path is not four-connected"
                    )
                total += step_cost(cell, index)
            if frames is not None:
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
        if max_cost is not None and g > max_cost:
            raise ValueError(
                "checkpoint candidate cost exceeds the cost limit"
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
    if best is not None:
        best_g, best_tie, best_path = best
        check_route(best_path, len(best_path) - 1 if dynamic else 0, best_g)
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
        "cost_limited": cost_limited,
    }


def _restore_multi_static_state(raw, width, height, obstacles, starts, goal,
                                costs, trace, max_cost):
    # Rebuild and fully cross-check the multi-source static snapshot state.
    if not isinstance(raw, dict):
        raise TypeError(
            f"checkpoint state must be an object, got {type(raw).__name__}"
        )
    for key in ("open", "best", "closed"):
        if key not in raw:
            raise TypeError(
                f"checkpoint state is missing required field {key!r}"
            )
    cost_limited = _restore_cost_limited(raw, max_cost)

    def heuristic(point):
        return abs(point[0] - goal[0]) + abs(point[1] - goal[1])

    def step_cost(point):
        # Cost of entering ``point``; the start cell is never entered.
        if costs is None:
            return 1
        return costs[point[1]][point[0]]

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
                or len(item) != 6):
            raise TypeError(
                f"{name} must be an [f, h, x, y, g, path] candidate"
            )
        f, h, x, y, g, raw_path = item
        if not all(_is_int(value) for value in (f, h, x, y, g)):
            raise TypeError(f"{name} must contain only integer fields")
        open_entries.append((f, h, x, y, g,
                             _snapshot_path(raw_path, f"{name} path")))
    raw_best = raw["best"]
    if isinstance(raw_best, (str, bytes)) or not isinstance(raw_best, Sequence):
        raise TypeError(
            f"checkpoint state best must be a sequence of entries, "
            f"got {type(raw_best).__name__}"
        )
    best_entries = []
    for index, item in enumerate(raw_best):
        name = f"checkpoint state best[{index}]"
        if (isinstance(item, (str, bytes)) or not isinstance(item, Sequence)
                or len(item) != 3):
            raise TypeError(f"{name} must be a [cell, g, path] entry")
        cell = _normalize_point(item[0], f"{name}[0]")
        if not _is_int(item[1]):
            raise TypeError(
                f"{name}[1] must be an int, got {type(item[1]).__name__}"
            )
        best_entries.append(
            (cell, item[1], _snapshot_path(item[2], f"{name} path"))
        )
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
    _check_heap_order([
        (f, h, x, y, path) for f, h, x, y, g, path in open_entries
    ])

    def check_route(path, g):
        if path[0] not in starts:
            raise ValueError(
                "checkpoint candidate path does not begin at a start"
            )
        seen = set()
        total = 0
        previous = None
        for cell in path:
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
            seen.add(cell)
            previous = cell
        if g != total:
            raise ValueError(
                "checkpoint candidate cost does not match its path"
            )
        if max_cost is not None and g > max_cost:
            raise ValueError(
                "checkpoint candidate cost exceeds the cost limit"
            )

    best = {}
    for cell, g, path in best_entries:
        check_route(path, g)
        if path[-1] != cell:
            raise ValueError(
                "checkpoint state best head does not match its path"
            )
        if cell in best:
            raise ValueError(
                "checkpoint state best records a cell twice"
            )
        best[cell] = (g, path)
    for start in starts:
        if best.get(start) != (0, (start,)):
            raise ValueError("checkpoint state does not seed a start cell")
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
    if not closed_set <= set(best):
        raise ValueError("checkpoint closed cells are missing from best")
    live = False
    for f, h, x, y, g, path in open_entries:
        check_route(path, g)
        cell = (x, y)
        if path[-1] != cell:
            raise ValueError(
                "checkpoint candidate head does not match its path"
            )
        if h != heuristic(cell):
            raise ValueError(
                "checkpoint candidate heuristic is inconsistent"
            )
        if f != g + h:
            raise ValueError(
                "checkpoint candidate priority is inconsistent"
            )
        known = best.get(cell)
        if known is None:
            raise ValueError(
                "checkpoint candidate is missing from best"
            )
        if (g, path) < known:
            raise ValueError(
                "checkpoint candidate improves on its recorded best"
            )
        if (g, path) == known:
            live = True
    if not live:
        raise ValueError("checkpoint state has no live pending candidates")
    if goal in closed_set and open_entries[0][0] > best[goal][0]:
        raise ValueError(
            "checkpoint best route should already have been returned"
        )
    return {
        "open": [(f, h, x, y, path, g) for f, h, x, y, g, path in open_entries],
        "best": best,
        "closed": closed_set,
        "expanded_nodes": list(trace),
        "cost_limited": cost_limited,
    }


def _restore_multi_tree_state(raw, width, height, obstacles, frames, starts,
                              goal, costs, closed_count, trace, max_cost,
                              allow_wait=False, cost_frames=None):
    # Rebuild and fully cross-check the multi-start route-tree snapshot
    # state (dynamic ``plan_multi_start``). Timed obstacle frames or
    # time-varying cost frames both make the search time-expanded; only
    # the obstacle frames constrain cells and waits.
    if not isinstance(raw, dict):
        raise TypeError(
            f"checkpoint state must be an object, got {type(raw).__name__}"
        )
    for key in ("open", "best"):
        if key not in raw:
            raise TypeError(
                f"checkpoint state is missing required field {key!r}"
            )
    cost_limited = _restore_cost_limited(raw, max_cost)

    def heuristic(point):
        return abs(point[0] - goal[0]) + abs(point[1] - goal[1])

    last_cost_frame = (len(cost_frames) - 1
                       if cost_frames is not None else None)

    def step_cost(point, t):
        # Cost of entering ``point`` at path index ``t``; the start cell
        # is never entered. Time-varying frames take the value of frame
        # ``t`` (the last frame persists); otherwise the static ``costs``
        # matrix or unit steps apply.
        if cost_frames is not None:
            frame = (cost_frames[t] if t <= last_cost_frame
                     else cost_frames[last_cost_frame])
            return frame[point[1]][point[0]]
        if costs is None:
            return 1
        return costs[point[1]][point[0]]

    last_frame = len(frames) - 1 if frames is not None else None

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
                or not isinstance(raw_best, Sequence) or len(raw_best) != 2):
            raise TypeError(
                "checkpoint state best must be null or a [g, path] entry"
            )
        best_g, best_path_raw = raw_best
        if not _is_int(best_g):
            raise TypeError("checkpoint state best cost must be an int")
        best = (best_g,
                _snapshot_path(best_path_raw, "checkpoint state best path"))

    # ---- semantic pass: every candidate must be a feasible route state ----
    if not entries:
        raise ValueError("checkpoint state has no pending candidates")
    _check_heap_order([
        (f, h, x, y, t, g, path) for f, h, x, y, t, g, path in entries
    ])

    def check_route(path, t, g):
        if path[0] not in starts:
            raise ValueError(
                "checkpoint candidate path does not begin at a start"
            )
        if t != len(path) - 1:
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
            wait = previous is not None and cell == previous
            if wait:
                # A consecutive repeat is only a legal wait when the
                # checkpoint recorded ``allow_wait`` and the stay still
                # lies within the provided frames.
                if not (allow_wait and frames is not None):
                    raise ValueError(
                        "checkpoint candidate path repeats a coordinate"
                    )
                if index > last_frame:
                    raise ValueError(
                        "checkpoint candidate waits beyond the final frame"
                    )
            elif cell in seen:
                raise ValueError(
                    "checkpoint candidate path repeats a coordinate"
                )
            elif previous is not None:
                if (abs(cell[0] - previous[0])
                        + abs(cell[1] - previous[1])) != 1:
                    raise ValueError(
                        "checkpoint candidate path is not four-connected"
                    )
                total += step_cost(cell, index)
            if frames is not None:
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
        if max_cost is not None and g > max_cost:
            raise ValueError(
                "checkpoint candidate cost exceeds the cost limit"
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
    if best is not None:
        best_g, best_path = best
        check_route(best_path, len(best_path) - 1, best_g)
        if best_path[-1] != goal:
            raise ValueError(
                "checkpoint best route does not end at the goal"
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
        "cost_limited": cost_limited,
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
        if key == "start":
            # ``plan_multi_start`` snapshots record ``starts`` in the same
            # position; every other planner keeps the single ``start``.
            if checkpoint.get("planner") == "plan_multi_start":
                key = "starts"
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
    if planner not in ("plan", "plan_any", "plan_multi_start"):
        raise ValueError(
            f"checkpoint planner {planner!r} is not supported"
        )
    endpoint_key = "goals" if planner == "plan_any" else "goal"
    if endpoint_key not in checkpoint:
        raise TypeError(
            f"checkpoint is missing required field {endpoint_key!r}"
        )
    # Grid constraints, revalidated in exactly ``plan``'s order with the
    # same exception types.
    width = _validate_dimension(checkpoint["width"], "width")
    height = _validate_dimension(checkpoint["height"], "height")
    if planner == "plan_multi_start":
        starts = _normalize_starts(checkpoint["starts"])
        start = None
    else:
        start = _normalize_point(checkpoint["start"], "start")
        starts = (start,)
    if planner == "plan_any":
        goal = None
        goals = _normalize_goals(checkpoint["goals"])
    else:
        goal = _normalize_point(checkpoint["goal"], "goal")
        goals = (goal,)
    for point in starts:
        _check_bounds(point, width, height, "start")
    for point in goals:
        _check_bounds(point, width, height, "goal")
    obstacles = _normalize_blocked(checkpoint["blocked"], width, height)
    for point in starts:
        if point in obstacles:
            raise ValueError(f"start {point} lies on a blocked cell")
    for point in goals:
        if point in obstacles:
            raise ValueError(f"goal {point} lies on a blocked cell")
    costs = _normalize_costs(checkpoint["costs"], width, height)
    frames = _normalize_dynamic_blocked(
        checkpoint["dynamic_blocked"], width, height
    )
    if frames is not None:
        for point in starts:
            if point in frames[0]:
                raise ValueError(f"start {point} is blocked at frame 0")
    # The cost limit is optional: checkpoints written before it existed
    # (and checkpoints of searches without a limit) simply omit the
    # field, which restores as ``None``. A present field follows the same
    # rules as ``plan``'s ``max_cost``.
    max_cost = checkpoint.get("max_cost")
    if max_cost is not None:
        if not _is_int(max_cost):
            raise TypeError(
                f"checkpoint max_cost must be an int, "
                f"got {type(max_cost).__name__}"
            )
        if max_cost < 0:
            raise ValueError(
                f"checkpoint max_cost must be a non-negative integer, "
                f"got {max_cost}"
            )
    # The waiting flag is optional: checkpoints written before it existed
    # (and checkpoints of searches without waiting) simply omit the field,
    # which restores as ``False``. A present field must be a bool, and the
    # candidate routes inside the state are cross-checked against it (a
    # stay without the flag, or one beyond the final frame, is a
    # ``ValueError`` from the state validation below).
    allow_wait = checkpoint.get("allow_wait", False)
    if not isinstance(allow_wait, bool):
        raise TypeError(
            f"checkpoint allow_wait must be a bool, "
            f"got {type(allow_wait).__name__}"
        )
    # The time-varying cost frames are optional: checkpoints written
    # before they existed (and checkpoints of searches without them)
    # simply omit the field, which restores as ``None``. A present field
    # follows the same rules as ``plan``'s ``dynamic_costs`` (structural
    # problems raise ``TypeError``, wrong shape or non-positive costs
    # raise ``ValueError``); recording both ``costs`` and
    # ``dynamic_costs``, or recording them for a planner that never
    # accepts them, is an inconsistent snapshot and raises ``ValueError``.
    cost_frames = _normalize_dynamic_costs(
        checkpoint.get("dynamic_costs"), width, height
    )
    if cost_frames is not None:
        if costs is not None:
            raise ValueError(
                "checkpoint records both costs and dynamic_costs"
            )
        if planner not in ("plan", "plan_any", "plan_multi_start"):
            raise ValueError(
                f"checkpoint planner {planner!r} does not accept "
                f"dynamic_costs"
            )
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
    dynamic = frames is not None or cost_frames is not None
    # Only the static merged searches close each cell at most once;
    # route-tree searches may record the same cell through different
    # histories.
    unique_trace = planner in ("plan", "plan_multi_start") and not dynamic
    trace = _restore_trace(
        checkpoint["trace"], dynamic, unique_trace, width, height,
        obstacles, frames
    )
    if len(trace) != closed_count:
        raise ValueError(
            "checkpoint closed count does not match its trace"
        )
    if trace:
        if planner == "plan_multi_start":
            if trace[0][:2] not in starts:
                raise ValueError(
                    "checkpoint trace does not begin at a start"
                )
        elif trace[0][:2] != start:
            raise ValueError("checkpoint trace does not begin at start")
        if dynamic and trace[0][2] != 0:
            raise ValueError("checkpoint trace does not begin at frame 0")
    if planner == "plan" and not dynamic:
        state = _restore_static_state(
            checkpoint["state"], width, height, obstacles, start, goal,
            trace, max_cost
        )
    elif planner == "plan_multi_start":
        if dynamic:
            state = _restore_multi_tree_state(
                checkpoint["state"], width, height, obstacles, frames,
                starts, goal, costs, closed_count, trace, max_cost,
                allow_wait, cost_frames
            )
        else:
            state = _restore_multi_static_state(
                checkpoint["state"], width, height, obstacles, starts,
                goal, costs, trace, max_cost
            )
    else:
        state = _restore_tree_state(
            checkpoint["state"], planner, dynamic, width, height,
            obstacles, frames, start, goal, goals, costs, closed_count,
            trace, max_cost, allow_wait, cost_frames
        )
    return {
        "planner": planner,
        "width": width,
        "height": height,
        "obstacles": obstacles,
        "start": start,
        "starts": starts,
        "goal": goal,
        "goals": goals,
        "costs": costs,
        "frames": frames,
        "cost_frames": cost_frames,
        "max_cost": max_cost,
        "allow_wait": allow_wait,
        "state": state,
    }


def plan(width, height, blocked, start, goal, costs=None, trace=False,
         dynamic_blocked=None, max_expanded=None, snapshot=False,
         max_cost=None, allow_wait=False, dynamic_costs=None):
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
    # The waiting flag is validated right after the dynamic-frame checks;
    # the budget is validated after every pre-existing check, then the
    # snapshot flag, and the new cost limit is validated last of all,
    # before the search starts. The time-varying cost frames follow every
    # pre-existing check; providing both ``costs`` and ``dynamic_costs``
    # is a ``ValueError``.
    _validate_allow_wait(allow_wait)
    _validate_budget(max_expanded)
    _validate_snapshot_flag(snapshot)
    _validate_max_cost(max_cost)
    cost_frames = _normalize_dynamic_costs(dynamic_costs, width, height)
    if cost_frames is not None and costs is not None:
        raise ValueError(
            "costs and dynamic_costs cannot both be provided"
        )
    if frames is not None or cost_frames is not None:
        return _search_dynamic(
            width, height, obstacles, frames, start, goal, costs, trace,
            max_expanded, max_cost, snapshot, allow_wait,
            cost_frames=cost_frames
        )
    return _search_static(
        width, height, obstacles, start, goal, costs, trace, max_expanded,
        max_cost, snapshot, allow_wait
    )


def plan_any(width, height, blocked, start, goals, costs=None, trace=False,
             dynamic_blocked=None, max_expanded=None, snapshot=False,
             max_cost=None, allow_wait=False, dynamic_costs=None):
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
    # The waiting flag is validated right after the dynamic-frame checks;
    # the budget stays the last pre-existing check before the search,
    # exactly as in ``plan``; the snapshot flag and then the new cost
    # limit follow it. The time-varying cost frames come last of all;
    # providing both ``costs`` and ``dynamic_costs`` is a ``ValueError``.
    _validate_allow_wait(allow_wait)
    _validate_budget(max_expanded)
    _validate_snapshot_flag(snapshot)
    _validate_max_cost(max_cost)
    cost_frames = _normalize_dynamic_costs(dynamic_costs, width, height)
    if cost_frames is not None and costs is not None:
        raise ValueError(
            "costs and dynamic_costs cannot both be provided"
        )
    return _search_any(
        width, height, obstacles, frames, start, goal_points, costs, trace,
        max_expanded, max_cost, snapshot, allow_wait,
        cost_frames=cost_frames
    )


def plan_batch(width, height, blocked, requests, costs=None, trace=False,
               dynamic_blocked=None, max_expanded=None, snapshot=False,
               max_cost=None, allow_wait=False, dynamic_costs=None):
    # --- Validation: ``plan``'s shared checks in their usual order,    ---
    # --- with ``requests`` occupying the position of ``start``/``goal``---
    # --- in that sequence. Every check is decided before any search    ---
    # --- begins, so an invalid batch never returns partial results.    ---
    width = _validate_dimension(width, "width")
    height = _validate_dimension(height, "height")
    pairs = _normalize_requests(requests)
    for index, (start, goal) in enumerate(pairs):
        _check_bounds(start, width, height, f"requests[{index}] start")
        _check_bounds(goal, width, height, f"requests[{index}] goal")
    obstacles = _normalize_blocked(blocked, width, height)
    for index, (start, goal) in enumerate(pairs):
        if start in obstacles:
            raise ValueError(
                f"requests[{index}] start {start} lies on a blocked cell"
            )
        if goal in obstacles:
            raise ValueError(
                f"requests[{index}] goal {goal} lies on a blocked cell"
            )
    costs = _normalize_costs(costs, width, height)
    if not isinstance(trace, bool):
        raise TypeError(
            f"trace must be a bool, got {type(trace).__name__}"
        )
    frames = _normalize_dynamic_blocked(dynamic_blocked, width, height)
    if frames is not None:
        for index, (start, _) in enumerate(pairs):
            if start in frames[0]:
                raise ValueError(
                    f"requests[{index}] start {start} is blocked at "
                    f"frame 0"
                )
    # The waiting flag, budget, snapshot flag and cost limit keep
    # ``plan``'s positions: after every grid and requests check and before
    # any search starts. The time-varying cost frames follow every
    # pre-existing check, still before any request is searched; providing
    # both ``costs`` and ``dynamic_costs`` is a ``ValueError``.
    _validate_allow_wait(allow_wait)
    _validate_budget(max_expanded)
    _validate_snapshot_flag(snapshot)
    _validate_max_cost(max_cost)
    cost_frames = _normalize_dynamic_costs(dynamic_costs, width, height)
    if cost_frames is not None and costs is not None:
        raise ValueError(
            "costs and dynamic_costs cannot both be provided"
        )
    # Each request is an independent ``plan`` search over the shared,
    # already normalized grid: ``path``/``cost``/``expanded`` are
    # produced per query and the budget and cost limit apply per query,
    # so no search state is ever shared between requests. The input
    # order and duplicate requests are preserved in the results.
    results = []
    for start, goal in pairs:
        if frames is not None or cost_frames is not None:
            results.append(_search_dynamic(
                width, height, obstacles, frames, start, goal, costs,
                trace, max_expanded, max_cost, snapshot, allow_wait,
                cost_frames=cost_frames
            ))
        else:
            results.append(_search_static(
                width, height, obstacles, start, goal, costs, trace,
                max_expanded, max_cost, snapshot, allow_wait
            ))
    return {"results": results}


def plan_multi_start(width, height, blocked, starts, goal, costs=None,
                     trace=False, dynamic_blocked=None, max_expanded=None,
                     snapshot=False, max_cost=None, allow_wait=False,
                     dynamic_costs=None):
    # --- Validation: ``starts`` takes ``start``'s exact position in     ---
    # --- ``plan``'s validation sequence; every shared check keeps its   ---
    # --- order, exception type and message boundary.                    ---
    width = _validate_dimension(width, "width")
    height = _validate_dimension(height, "height")
    start_points = _normalize_starts(starts)
    goal = _normalize_point(goal, "goal")
    for point in start_points:
        _check_bounds(point, width, height, "start")
    _check_bounds(goal, width, height, "goal")
    obstacles = _normalize_blocked(blocked, width, height)
    for point in start_points:
        if point in obstacles:
            raise ValueError(f"start {point} lies on a blocked cell")
    if goal in obstacles:
        raise ValueError(f"goal {goal} lies on a blocked cell")
    costs = _normalize_costs(costs, width, height)
    if not isinstance(trace, bool):
        raise TypeError(
            f"trace must be a bool, got {type(trace).__name__}"
        )
    frames = _normalize_dynamic_blocked(dynamic_blocked, width, height)
    if frames is not None:
        for point in start_points:
            if point in frames[0]:
                raise ValueError(f"start {point} is blocked at frame 0")
    # The waiting flag, budget, snapshot flag and cost limit keep
    # ``plan``'s positions: after every grid and starts check and before
    # any search starts. The time-varying cost frames follow every
    # pre-existing check, still before the search; providing both
    # ``costs`` and ``dynamic_costs`` is a ``ValueError``.
    _validate_allow_wait(allow_wait)
    _validate_budget(max_expanded)
    _validate_snapshot_flag(snapshot)
    _validate_max_cost(max_cost)
    cost_frames = _normalize_dynamic_costs(dynamic_costs, width, height)
    if cost_frames is not None and costs is not None:
        raise ValueError(
            "costs and dynamic_costs cannot both be provided"
        )
    if frames is not None or cost_frames is not None:
        return _search_multi_dynamic(
            width, height, obstacles, frames, start_points, goal, costs,
            trace, max_expanded, max_cost, snapshot, allow_wait,
            cost_frames=cost_frames
        )
    return _search_multi_static(
        width, height, obstacles, start_points, goal, costs, trace,
        max_expanded, max_cost, snapshot, allow_wait
    )


def plan_k(width, height, blocked, start, goal, k, costs=None,
           dynamic_blocked=None, max_expanded=None, max_cost=None,
           allow_wait=False, dynamic_costs=None):
    """Return up to ``k`` distinct routes ordered by priority.

    ``plan_k(width, height, blocked, start, goal, k, costs=None,
    dynamic_blocked=None, max_expanded=None, max_cost=None,
    allow_wait=False, dynamic_costs=None)`` runs the
    same grid/obstacle/costs/frame validation as ``plan`` first and then
    checks ``k``: it must be a positive, non-bool integer (wrong types
    raise ``TypeError`` and non-positive values raise ``ValueError``).
    The optional ``max_expanded`` and ``max_cost`` limits are validated
    after all of those checks (and after the ``k`` check), still before
    the search starts: each must be a non-negative, non-bool integer
    (other types raise ``TypeError`` and negative values raise
    ``ValueError``), reusing ``plan``'s budget and cost-limit rules.
    ``allow_wait`` must be a bool and is checked right before the limits
    (a non-bool raises ``TypeError``). The optional ``dynamic_costs``
    frames are validated after all of those checks, still before the
    search starts, with the shared time-varying cost rules (providing
    both ``costs`` and ``dynamic_costs`` raises ``ValueError``).
    Every error is decided before the search starts. The search keeps
    four-neighborhood moves, zero start cost, entering-cell cost
    accumulation, the persistent last frame, no waiting (unless
    ``allow_wait`` is set) and no repeated
    coordinates; routes reaching the same cell through different
    coordinate histories are distinct candidates and are never merged.

    Returns ``{"paths": [...], "costs": [...], "expanded": int}`` with the
    at most ``k`` unique complete routes ordered by ascending total cost
    and, on ties, by the lexicographic order of the complete coordinate
    sequence (``costs`` corresponds entry by entry). When no route is
    feasible both lists are empty; ``expanded`` still reports the number
    of route candidates actually closed to determine the result -- once
    per complete history, never counting filtered or still-pending
    candidates. The ``start == goal`` result is the single-point zero-cost
    route alone, with one expansion. Every returned route verifies in
    ``replay`` with the same cost and step count, in both static and
    dynamic mode.

    When either ``max_expanded`` or ``max_cost`` is provided (not
    omitted/``None``) the result additionally carries ``status``:
    ``"found"`` when all ``k`` routes were closed or the search exhausted
    naturally with at least one route; ``"budget_exhausted"`` when fewer
    than ``k`` routes were found and the close budget stopped the search
    while candidates remained pending; ``"cost_exhausted"`` when fewer
    than ``k`` routes were found, no budget stop occurred and at least one
    candidate was discarded for exceeding the cost limit; and
    ``"unreachable"`` when no route exists with neither stop nor discard.
    A budget stop wins over a cost-limit stop. The already closed routes
    (and their costs, in rank order) are always returned even when a
    limit cuts the search short, and empty lists are used when none were
    found; ``expanded`` is always the actual closed-candidate count.
    Omitting both limits keeps the exact legacy result keys and values.
    """
    # --- Validation: exactly ``plan``'s shared checks first, then the  ---
    # --- ``k`` check, then the optional limits, all before the search  ---
    # --- begins.                                                       ---
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
    if not _is_int(k):
        raise TypeError(f"k must be an int, got {type(k).__name__}")
    if k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}")
    # The waiting flag and the optional limits follow every pre-existing
    # check (k included), reusing the shared bool and
    # non-negative-integer rules, before the search. The time-varying
    # cost frames come last of all; providing both ``costs`` and
    # ``dynamic_costs`` is a ``ValueError``.
    _validate_allow_wait(allow_wait)
    _validate_budget(max_expanded)
    _validate_max_cost(max_cost)
    cost_frames = _normalize_dynamic_costs(dynamic_costs, width, height)
    if cost_frames is not None and costs is not None:
        raise ValueError(
            "costs and dynamic_costs cannot both be provided"
        )
    return _search_k(width, height, obstacles, frames, start, goal,
                     costs, k, max_expanded, max_cost, allow_wait,
                     cost_frames=cost_frames)


def distance_field(width, height, blocked, goal, costs=None, trace=False):
    """Compute minimum entering-cell distances to ``goal`` for every cell.

    ``distance_field(width, height, blocked, goal, costs=None,
    trace=False)`` is an offline analysis entry point for the static
    grid; it accepts no start, dynamic obstacles, budget, cost limit or
    snapshot option. The dimensions, goal coordinate, static obstacles,
    cost matrix and trace flag are validated with ``plan``'s public
    rules for those shared arguments, all before the search starts.

    Returns ``{"distances": ..., "expanded": int}``: ``distances[y][x]``
    is ``None`` for obstacles and goal-unreachable cells, ``0`` for the
    goal, and otherwise the minimum sum of the entering-cell costs along
    a four-neighborhood route from that cell to the goal (unit steps
    without ``costs``), matching the ``cost`` ``plan`` reports from that
    cell. ``expanded`` counts the reachable cells closed once each, the
    goal included; the closing order is smallest current distance first,
    then the ``(x, y)`` coordinate lexicographically. With
    ``trace=True`` the result additionally carries ``expanded_nodes``,
    the coordinate pairs in that closing order, and
    ``expanded == len(expanded_nodes)``.
    """
    # --- Validation: ``plan``'s shared checks for the dimensions, the  ---
    # --- goal coordinate, the static obstacles, the cost matrix and    ---
    # --- the trace flag, all decided before the search begins.         ---
    width = _validate_dimension(width, "width")
    height = _validate_dimension(height, "height")
    goal = _normalize_point(goal, "goal")
    _check_bounds(goal, width, height, "goal")
    obstacles = _normalize_blocked(blocked, width, height)
    if goal in obstacles:
        raise ValueError(f"goal {goal} lies on a blocked cell")
    costs = _normalize_costs(costs, width, height)
    if not isinstance(trace, bool):
        raise TypeError(
            f"trace must be a bool, got {type(trace).__name__}"
        )
    return _search_distance_field(
        width, height, obstacles, goal, costs, trace
    )


def distance_field_any(width, height, blocked, goals, costs=None,
                       trace=False):
    """Compute minimum entering-cell distances to any of several goals.

    ``distance_field_any(width, height, blocked, goals, costs=None,
    trace=False)`` is the multi-goal counterpart of ``distance_field``
    and the offline-analysis sibling of ``plan_any``; it accepts no
    start, dynamic obstacles, budget, cost limit or snapshot option and
    never produces a ``status`` field. The dimensions, static obstacles,
    cost matrix and trace flag are validated with ``distance_field``'s
    public rules, and ``goals`` uses ``plan_any``'s rules in ``goal``'s
    position in that sequence: a non-empty sequence of grid coordinates,
    ``None``/strings/bytes/other non-sequences raise ``TypeError``, an
    empty sequence raises ``ValueError``, elements that are not exactly
    two non-bool integers raise ``TypeError``, and out-of-bounds or
    statically blocked endpoints raise ``ValueError``. Duplicates merge
    and every check runs before the search starts.

    Returns ``{"distances": ..., "targets": ..., "expanded": int}``:
    ``distances[y][x]`` is ``None`` for obstacles and cells that cannot
    reach any endpoint, ``0`` for every endpoint, and otherwise the
    minimum sum of entering-cell costs along a four-neighborhood route
    to any endpoint (unit steps without ``costs``), matching the
    ``cost`` ``plan_any`` reports from that cell. ``targets[y][x]``
    records the winning endpoint coordinate for every reachable cell and
    is ``None`` elsewhere; on an exact distance tie the endpoint with the
    lexicographically smallest ``(x, y)`` wins. ``expanded`` counts the
    reachable cells closed once each, all endpoints included, in the
    order smallest distance first then ``(x, y)`` lexicographically.
    With ``trace=True`` the result additionally carries
    ``expanded_nodes``, the coordinate pairs in that closing order, and
    ``expanded == len(expanded_nodes)``. The whole result is
    JSON-serializable.
    """
    # --- Validation: ``distance_field``'s shared checks for the         ---
    # --- dimensions, with ``goals`` taking ``goal``'s position, then    ---
    # --- the static obstacles, the cost matrix and the trace flag; all  ---
    # --- decided before the search begins.                              ---
    width = _validate_dimension(width, "width")
    height = _validate_dimension(height, "height")
    goal_points = _normalize_goals(goals)
    for point in goal_points:
        _check_bounds(point, width, height, "goal")
    obstacles = _normalize_blocked(blocked, width, height)
    for point in goal_points:
        if point in obstacles:
            raise ValueError(f"goal {point} lies on a blocked cell")
    costs = _normalize_costs(costs, width, height)
    if not isinstance(trace, bool):
        raise TypeError(
            f"trace must be a bool, got {type(trace).__name__}"
        )
    return _search_distance_field_any(
        width, height, obstacles, goal_points, costs, trace
    )


def replay(width, height, blocked, start, goal, path, costs=None,
           dynamic_blocked=None, diagnose=False, allow_wait=False,
           dynamic_costs=None):
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
    _validate_allow_wait(allow_wait)
    if not isinstance(diagnose, bool):
        raise TypeError(
            f"diagnose must be a bool, got {type(diagnose).__name__}"
        )
    # The time-varying cost frames follow every pre-existing check, still
    # before the path structure is inspected; providing both ``costs``
    # and ``dynamic_costs`` is a ``ValueError``.
    cost_frames = _normalize_dynamic_costs(dynamic_costs, width, height)
    if cost_frames is not None and costs is not None:
        raise ValueError(
            "costs and dynamic_costs cannot both be provided"
        )
    points = _normalize_path(path, width, height)

    last_cost_frame = (len(cost_frames) - 1
                       if cost_frames is not None else None)

    def step_cost(point, t):
        # Cost of entering ``point`` at path index ``t``; the start cell
        # is never entered. Time-varying frames take the value of frame
        # ``t`` (the last frame persists); otherwise the static ``costs``
        # matrix or unit steps apply.
        if cost_frames is not None:
            frame = (cost_frames[t] if t <= last_cost_frame
                     else cost_frames[last_cost_frame])
            return frame[point[1]][point[0]]
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
            # A consecutive stay is a legal wait only with ``allow_wait``
            # and dynamic frames, and only while the stay still lies
            # within the provided frames; it adds no entering-cell cost.
            if (allow_wait and frames is not None
                    and previous is not None and point == previous):
                if t > last_frame:
                    return invalid("wait_after_final_frame", t)
                if point in frames[t]:
                    return invalid("dynamic_blocked", t)
                previous = point
                continue
            return invalid("repeated_coordinate", t)  # second occurrence
        if previous is not None:
            if (abs(point[0] - previous[0])
                    + abs(point[1] - previous[1])) != 1:
                return invalid("non_adjacent", t)  # later point of the pair
            total += step_cost(point, t)
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
    # search once more. The checkpoint's cost limit (``None`` when the
    # field is absent), its waiting flag (``False`` when absent) and its
    # time-varying cost frames (``None`` when absent) keep governing the
    # search.
    max_cost = restored["max_cost"]
    allow_wait = restored["allow_wait"]
    cost_frames = restored["cost_frames"]
    if restored["planner"] == "plan_multi_start":
        if restored["frames"] is None and cost_frames is None:
            return _search_multi_static(
                restored["width"], restored["height"],
                restored["obstacles"], restored["starts"],
                restored["goal"], restored["costs"],
                True, max_expanded, max_cost, snapshot=True,
                allow_wait=allow_wait, state=restored["state"]
            )
        return _search_multi_dynamic(
            restored["width"], restored["height"], restored["obstacles"],
            restored["frames"], restored["starts"], restored["goal"],
            restored["costs"], True, max_expanded, max_cost, snapshot=True,
            allow_wait=allow_wait, state=restored["state"],
            cost_frames=cost_frames
        )
    if (restored["planner"] == "plan" and restored["frames"] is None
            and cost_frames is None):
        return _search_static(
            restored["width"], restored["height"], restored["obstacles"],
            restored["start"], restored["goal"], restored["costs"],
            True, max_expanded, max_cost, snapshot=True,
            allow_wait=allow_wait, state=restored["state"]
        )
    if restored["planner"] == "plan":
        return _search_dynamic(
            restored["width"], restored["height"], restored["obstacles"],
            restored["frames"], restored["start"], restored["goal"],
            restored["costs"], True, max_expanded, max_cost, snapshot=True,
            allow_wait=allow_wait, state=restored["state"],
            cost_frames=cost_frames
        )
    return _search_any(
        restored["width"], restored["height"], restored["obstacles"],
        restored["frames"], restored["start"], restored["goals"],
        restored["costs"], True, max_expanded, max_cost, snapshot=True,
        allow_wait=allow_wait, state=restored["state"],
        cost_frames=cost_frames
    )
