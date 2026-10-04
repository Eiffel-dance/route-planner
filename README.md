# Route Planner

A dependency-free Python reference implementation for robotics, path-planning, a-star.

Run with: python3 demo.py
Tests: python3 -m unittest discover -s tests -v

## Scope

实现一个确定性的二维栅格 A* 路径规划器。规划器支持障碍物、四邻域移动、固定的 tie-break 规则和不可达结果；返回路径、总代价及扩展节点统计，不能越过障碍或产生重复节点。公开数据结构和边界行为应当适合离线回放，后续可在同一基线上扩展代价地图和动态障碍处理。

## API 约定

`plan(width, height, blocked, start, goal, costs=None, trace=False, dynamic_blocked=None, max_expanded=None, snapshot=False, max_cost=None)` 默认返回 `{"path", "cost", "expanded"}`；`trace=True` 时额外返回 `expanded_nodes`；提供 `max_expanded` 或 `max_cost` 时额外返回 `status`；`snapshot=True` 且因预算耗尽返回 `budget_exhausted` 时额外返回 `checkpoint`。

**搜索预算（`max_expanded`）**
- `max_expanded` 省略或为 `None` 时，返回键、值、异常类型与校验顺序与既有行为完全一致。提供时必须是非负且非布尔的整数：其他类型抛 `TypeError`，负数抛 `ValueError`；这些校验在全部既有网格、`costs`、`trace` 与 `dynamic_blocked` 校验（含第 0 帧检查）完成之后、搜索开始前进行。
- 预算按关闭节点数计（即 `expanded` 的计数口径），达到上限后不再关闭任何节点。提供预算时结果始终额外携带 `status`：目标在限额内关闭为 `"found"`（返回既有的 `path`/`cost`）；候选耗尽仍未到达目标为 `"unreachable"`；尚有候选但预算耗尽为 `"budget_exhausted"`（`path`/`cost` 为 `None`，`expanded` 为实际关闭数，不超过 `max_expanded`）。
- 预算对静态与动态模式同样生效，且不改变扩展顺序、路径选择、轨迹条目或确定性：带预算的轨迹是无预算轨迹的前缀。`trace=True` 时只记录实际关闭的节点，预算为零时即使 `start == goal` 也返回 `budget_exhausted`（`expanded` 为 0，`expanded_nodes` 为空）；单点成功结果要求至少关闭一个节点（`max_expanded >= 1`）。
- `replay` 不受预算影响：不接受预算参数，仍只校验候选路径。

**代价上限（`max_cost`）**
- `max_cost` 省略或为 `None` 时，返回键、值、异常类型与校验顺序与既有行为完全一致。提供时必须是非负且非布尔的整数：其他类型抛 `TypeError`，负数抛 `ValueError`；该校验在全部既有网格、`costs`、`trace`、`dynamic_blocked`、`max_expanded` 与 `snapshot` 校验完成之后、搜索开始前进行。
- 上限约束路线从起点出发的累计进入格子代价：起点代价仍为 0，进入格子的代价沿用 `costs`（未提供时每步为 1）。累计值超过上限的候选被丢弃，恰好等于上限的候选仍参与竞争。上限对静态、动态与 `plan_any` 同样生效，且不改变扩展顺序、固定 tie-break、最小总代价选择、多终点的终点坐标与完整路径字典序比较或轨迹条目。
- 提供 `max_cost` 时结果始终携带 `status`：成功为 `"found"`；没有可行终点时 `path`/`cost` 为 `None`，`expanded` 只统计实际关闭的节点，若有候选因超限被丢弃则为 `"cost_exhausted"`，否则为 `"unreachable"`；若同时给出 `max_expanded`，预算耗尽优先为 `"budget_exhausted"`。`trace` 仍只记录实际关闭的节点；`start == goal` 的单点零代价结果与零扩展预算的既有规则不变。
- `snapshot=True` 生成的 checkpoint 在提供 `max_cost` 时记录该值（未提供时不新增字段）；`resume` 对缺少该字段的旧 checkpoint 按 `None` 兼容，字段存在时按同一规则校验（类型错误抛 `TypeError`，负值或与内部状态不一致抛 `ValueError`），所有检查先于恢复搜索，且恢复结果与一次未暂停的调用完全一致。`replay` 不接受 `max_cost` 参数。

`plan_any(width, height, blocked, start, goals, costs=None, trace=False, dynamic_blocked=None, max_expanded=None, snapshot=False, max_cost=None)` 接受与 `plan` 相同的网格、`blocked`、`start` 及可选代价、轨迹、动态障碍、预算、快照和代价上限参数，另收一个非空 `goals` 候选终点序列，返回一条确定的最低代价路线（返回结构与 `plan` 相同，`path` 末点为选中的候选终点）；该路径可直接以其末点作为 `goal` 交给 `replay` 离线核验。

**多终点（`goals`）**
- 外层必须是非空序列：字符串、字节串、`None` 或其他非序列抛 `TypeError`，空序列抛 `ValueError`。每个终点沿用坐标的结构和整数规则：元素形状或坐标类型不合格抛 `TypeError`，越界或落在静态障碍上抛 `ValueError`。`goals` 在 `plan` 校验序列中占据 `goal` 的位置，网格、`costs`、`trace`、`dynamic_blocked` 与预算的校验顺序及异常类型沿用 `plan`；`start` 仍须通过静态障碍与动态第 0 帧检查。
- 重复候选点合并，输入排列（以及 `blocked`、帧内坐标的迭代顺序）不影响任何结果。
- 候选终点在某条路线到达的时间帧被动态障碍阻挡时不在输入阶段报错：该路线按其时间帧不可达，其他到达时刻仍可参与竞争；末帧之后持续使用最后一帧。
- 搜索保留四邻域移动、禁止等待和回访、进入格子的代价累计。总 `cost` 最小者优先；同价时先比较终点坐标 `(x, y)` 字典序，再比较完整路径字典序。与动态 `plan` 同理，经不同历史到达同一格子的路线是不同状态、互不剪枝；`trace` 在静态模式记录坐标二元组、动态模式记录 `(x, y, t)` 三元组，且只记录实际关闭的节点。
- 没有任何可行终点时 `path`/`cost` 为 `None`；提供预算时按既有规则返回 `status` 为 `found`、`unreachable` 或 `budget_exhausted`，提供 `max_cost` 时同样始终携带 `status`（规则与 `plan` 一致，含 `cost_exhausted`），预算或上限不会把已确定的较优路线替换成严格次优（更贵）的路线。`goals` 含 `start` 时按既有 `start == goal` 单点零代价规则处理（零预算同样为 `budget_exhausted`）。

`plan_k(width, height, blocked, start, goal, k, costs=None, dynamic_blocked=None)` 在一次请求中返回同一栅格上按优先级排列的前 `k` 条候选路线，供离线比较多条可行方案。

**前 k 条路线（`k`）**
- 先沿用 `plan` 对尺寸、坐标、静态障碍、正整数代价矩阵与动态帧的类型及取值校验（含第 0 帧 `start` 检查），这些共享检查全部完成后再检查 `k`：`k` 必须是非布尔正整数，类型错误抛 `TypeError`，非正值（0 或负数）抛 `ValueError`；所有错误在搜索开始前确定。`plan_k` 不接受 `trace`、`max_expanded`、`snapshot`、`max_cost` 参数。
- 搜索仍只允许四邻域移动，起点代价为零，进入格子的代价按 `costs` 累加（未提供时每步为 1），路径不得越过静态障碍或路径下标对应时间帧的动态障碍（超过末帧持续使用末帧），也不得重复坐标或原地等待。经不同完整坐标历史到达同一位置的候选是不同状态、不提前合并，因此不同完整坐标序列始终是不同候选，静态与动态模式一致。
- 返回对象固定包含 `paths`、`costs`、`expanded`：`paths` 为最多 `k` 条唯一完整路线，按总代价升序、同价按完整坐标序列字典序排列；`costs` 与 `paths` 逐项对应，且首条路线即 `plan` 返回的路线。没有可行路线时两个数组均为空，`expanded` 仍报告为确定结果而实际关闭的候选状态数：每个完整历史只计一次，被过滤（越界、静态/动态障碍、回访）或仍未关闭的候选不计入；第 `k` 条目标路线一关闭即停止，因此较小的 `k` 比较大的 `k` 关闭更少候选。`start == goal` 时只返回单点零代价路线并统计一次扩展。
- 无论静态还是动态结果，每条路线都可逐条交给 `replay`（以路线末点作为 `goal`）并得到相同代价和步数（`steps == len(path) - 1`）。结果不依赖障碍集合、目标参数、代价矩阵行序或动态帧/帧内坐标的输入顺序。

`replay(width, height, blocked, start, goal, path, costs=None, dynamic_blocked=None, diagnose=False)` 用于离线核验一条已保存的候选路径并重新计算代价；它不执行搜索，返回值不含 `expanded` 或 `expanded_nodes` 字段，且不接受 `trace` 参数。

**快照与恢复（`snapshot` 与 `resume`）**
- `snapshot` 只能是布尔值，默认 `False`；省略或为 `False` 时 `plan`/`plan_any` 的返回键、值、异常类型、校验顺序、tie-break、不可达结果与轨迹完全不变。非布尔值在全部既有校验（含 `max_expanded` 校验）之后、搜索开始前抛 `TypeError`。
- `snapshot=True` 且本次搜索因达到 `max_expanded` 上限返回 `budget_exhausted` 时，结果额外携带 `checkpoint`；已找到终点或确认不可达的结果不生成该字段。`checkpoint` 只含可 JSON 序列化的值：快照版本、规划器类型（`"plan"`/`"plan_any"`）、规范化网格约束（尺寸、障碍、起终点或候选终点、`costs`、`dynamic_blocked`，以及提供 `max_cost` 时的代价上限）、已关闭节点计数、至此的轨迹以及待处理候选的完整状态；所有由集合派生的列表均排序存储，键与值的顺序不受集合迭代、目标列表或帧内坐标顺序影响，调用方可原样保存后传回 `resume`。
- `resume(checkpoint, max_expanded=None)` 按快照继续同一确定性搜索。`max_expanded` 仍表示从起点累计允许关闭的节点总数：上限不大于已关闭数时不再关闭任何节点。恢复结果的 `path`、`cost`、`expanded`、`status`（仅在提供 `max_expanded` 时返回）与 `expanded_nodes`（恢复结果始终返回；静态模式记录坐标二元组，动态模式记录 `(x, y, t)` 三元组）等价于一次未暂停的调用，重复恢复不会重复计入已关闭节点。再次达到上限时返回新的 `checkpoint`，可继续恢复；最终 `found` 或 `unreachable` 时省略它。恢复成功的路径可直接交给 `replay` 并得到相同代价与步数。
- 校验：`checkpoint` 不是对象、缺少必要字段或字段类型不对时抛 `TypeError`；快照版本、规划器类型、网格约束、障碍、起终点、候选终点、`costs`、`dynamic_blocked` 或内部状态不一致时统一抛 `ValueError`；所有错误在搜索前确定。`resume` 的 `max_expanded` 复用 `plan` 的非负整数规则（其他类型抛 `TypeError`，负数抛 `ValueError`）。

**诊断模式（`diagnose=True`）**
- `diagnose` 只能是布尔值；省略或为 `False` 时，`replay` 的返回键、值、异常类型与校验顺序与默认行为完全一致。非布尔值在全部网格校验（含动态第零帧）完成后、`path` 结构检查之前抛出 `TypeError`。
- 传 `True` 且结构合法时，有效路径仍返回 `valid`、`cost`、`steps`，并额外返回 `"error": None, "error_index": None`；`cost` 仍按进入格子规则重算，`steps == len(path) - 1`。
- 无效路径仍不抛异常，`cost`、`steps` 为 `None`，并返回唯一的 `error` 及首个违规元素的零基 `path` 下标。按固定优先级（同一点同时满足多项时取最前者，且不返回任何部分累计代价）：
  - `start_mismatch`：首点不是 `start`，下标 0；
  - `goal_mismatch`：末点不是 `goal`，下标为末点（`len(path) - 1`）；
  - `start_goal_extra`：`start == goal` 却含多余点，下标 1；
  - `repeated_coordinate`：重复坐标，指向第二次出现的位置；
  - `non_adjacent`：相邻点不是四邻域移动，指向后一个点；
  - `static_blocked`：经过静态障碍，指向该点；
  - `dynamic_blocked`：在对应时间帧（或持续使用的末帧）中被禁行，指向该点。

**replay 返回结构**
- 路径合规时返回 `{"valid": True, "cost": int, "steps": int}`：`steps` 为移动边数，恒等于 `len(path) - 1`（`start == goal` 的单点路径为 0）；`cost` 按 `plan` 既有的进入格子规则重算，即路径所进入格子的代价之和，未提供 `costs` 时每步为 1，提供时起点代价不计入，因此与 `plan` 对同一路径给出的代价一致。
- 路径语义不合规时不抛异常，固定返回 `{"valid": False, "cost": None, "steps": None}`，不返回任何部分累计值。不合规情形包括：未从 `start` 开始、未在 `goal` 结束、经过静态障碍、经过路径下标 `t` 对应的动态障碍帧、相邻点不是四邻域移动、出现重复坐标，以及 `start == goal` 却包含多余点。
- 合规判定与动态时间索引：`dynamic_blocked` 省略、为 `None` 或为空序列时按静态模式检查；否则路径下标 `t` 处的坐标必须避开第 `t` 帧，超过最后一帧后持续使用最后一帧（与 `plan` 的索引规则一致）。

**replay 异常边界（校验先于路径判定）**
- 网格参数（`width`、`height`、`blocked`、`start`、`goal`、`costs`、`dynamic_blocked`）沿用 `plan` 的公开校验与既有 `TypeError`/`ValueError` 边界及顺序，包括 `start` 落在静态障碍或第 0 帧时抛出 `ValueError`；这些校验全部在检查 `path` 之前完成。
- `path` 必须是非空坐标序列：字符串、字节串、`None` 或其他非序列抛出 `TypeError`；空序列抛出 `TypeError`；元素不是恰好两个整数（含长度不为二、非整数、布尔坐标）抛出 `TypeError`；坐标越界抛出 `ValueError`。
- 仅语义上无效（结构合法但不满足上述合规条件）的路径不抛异常，返回固定的无效结构。

**距离场（`distance_field`）**
- `distance_field(width, height, blocked, goal, costs=None, trace=False)` 是面向静态栅格的离线分析入口：从 `goal` 向外做 Dijkstra 遍历，计算每个格子沿四邻域到达 `goal` 的最小进入格子代价总和。它不接受 `start`、`dynamic_blocked`、`max_expanded`、`snapshot` 或 `max_cost` 参数；尺寸、`goal`、`blocked`、`costs` 与 `trace` 沿用 `plan` 的公开校验（`goal` 越界或落在障碍上抛 `ValueError`，`trace` 非布尔抛 `TypeError`），全部校验在搜索开始前完成。
- 返回 `{"distances": [...], "expanded": int}`。`distances` 按 `distances[y][x]` 索引，尺寸恒等于网格：障碍物与不可达格子为 `None`，`goal` 恒为 0，其余可达格子为从该格出发到达 `goal` 的最小代价（未提供 `costs` 时每步为 1；提供时沿用 `costs[y][x]` 的进入格子语义，因此可达的 `plan` 起点处的值与 `plan` 返回的 `cost` 相同）。任何可达格子的距离都不为负，结果可直接 JSON 序列化。
- 关闭顺序先按当前最小距离、再按 `(x, y)` 字典序；`expanded` 只统计被确定并关闭一次的可达格子（`goal` 计入）。`trace=True` 时额外返回 `expanded_nodes`（按关闭顺序的坐标二元组，恒有 `expanded == len(expanded_nodes)`）；`trace=False` 或省略时不含该字段。障碍集合与 `costs` 的输入顺序不影响矩阵、统计或轨迹。

**plan 输入校验（搜索开始前完成）** —— replay 的网格参数沿用同一套校验：
- `width`、`height` 必须为正整数。
- `start`、`goal` 及 `blocked` 中每个元素必须是两个整数构成的坐标（元组或长度为二的列表等序列，统一规范化为元组）。
- `costs` 可省略或为 `None`（单位代价）；否则必须是 `height` 行、`width` 列的正整数矩阵（按行排列，`costs[y][x]` 为进入格子 `(x, y)` 的代价）。字符串/字节串、非整数单元格、布尔值抛出 `TypeError`；行数或列数不符、代价非正抛出 `ValueError`。
- `trace` 必须为布尔值，其他类型在搜索开始前抛出 `TypeError`。
- 类型或结构不符抛出 `TypeError`；尺寸非正、坐标越界、端点落在障碍物上抛出 `ValueError`。
- `blocked` 中的重复坐标被合并，不影响结果。

**搜索语义**
- 四邻域移动，曼哈顿启发式。未提供 `costs` 时为单位步长代价；提供时进入格子的代价为该格的值，起点代价不计入，`cost` 等于路径所进入格子的代价之和。
- 平局次序固定且公开：先比较 f，再比较 h，再按 (x, y) 字典序；结果不依赖 `blocked` 的迭代顺序。
- `expanded` 只统计从优先队列取出并首次关闭的节点；成功时计入 goal，`start == goal` 时返回单节点路径、代价 0、`expanded` 为 1；不可达时 `path`/`cost` 为 `None`，`expanded` 等于实际关闭的可通行节点数。

**搜索轨迹（`trace=True`）**
- 成功与不可达结果都额外返回 `expanded_nodes`：坐标元组构成的序列，按节点首次计入 `expanded` 的顺序记录，包含 `start`；成功时包含 `goal`，不可达时记录搜索结束前关闭的全部可通行节点。
- `expanded_nodes` 不含未扩展项、重复坐标或障碍物，且恒有 `expanded == len(expanded_nodes)`；`start == goal` 时序列只有一个坐标。
- `trace=False` 或省略时返回对象不含该字段，其余键、值与校验顺序与默认行为完全一致。

**动态障碍（`dynamic_blocked`）**
- `dynamic_blocked` 省略、为 `None` 或为空序列时与未启用完全等价；否则必须是按时间帧排列的序列，每一帧是该帧禁行坐标的可迭代集合。第 0 帧约束 `start`，路径下标 `t` 处的坐标必须同时避开静态障碍和第 `t` 帧，超过最后一帧后持续使用最后一帧。
- 每一步仍只能向四邻域移动，不允许原地等待或重复坐标；`cost` 仍按进入格子的代价累计，起点不计。由于路线的可延续性取决于它已访问的坐标序列，到达同一 `(x, y, t)` 但历史不同的候选是不同状态，互不剪枝：所有满足约束的路线都能被搜索到。
- 返回总代价最小的可行路径；总代价相同时，固定优先级依次为 f、h、x、y、t，仍相同时按完整坐标路径的字典序决定取舍，因此结果不依赖障碍集合、帧内坐标顺序或遍历顺序。
- 动态模式下 `expanded_nodes` 按实际关闭顺序记录 `(x, y, t)` 三元组，`expanded` 按条目计数；被丢弃的候选不记录。同一 `(x, y, t)` 若由不同历史分别关闭，其三元组按关闭顺序分别出现，`expanded` 逐条计数。不可达时 `path`/`cost` 为 `None`。
- 校验：外层必须是序列，每一帧必须是可迭代坐标集合；字符串/字节串、非二整数坐标抛出 `TypeError`，越界坐标抛出 `ValueError`，帧内重复坐标合并；`start` 在第 0 帧被禁止时抛出 `ValueError`。所有校验在搜索开始前完成，既有校验顺序不变。
