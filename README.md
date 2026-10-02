# Route Planner

A dependency-free Python reference implementation for robotics, path-planning, a-star.

Run with: python3 demo.py
Tests: python3 -m unittest discover -s tests -v

## Scope

实现一个确定性的二维栅格 A* 路径规划器。规划器支持障碍物、四邻域移动、固定的 tie-break 规则和不可达结果；返回路径、总代价及扩展节点统计，不能越过障碍或产生重复节点。公开数据结构和边界行为应当适合离线回放，后续可在同一基线上扩展代价地图和动态障碍处理。

## API 约定

`plan(width, height, blocked, start, goal, costs=None, trace=False, dynamic_blocked=None)` 默认返回 `{"path", "cost", "expanded"}`；`trace=True` 时额外返回 `expanded_nodes`。

**输入校验（搜索开始前完成）**
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
- 每一步仍只能向四邻域移动，不允许原地等待或重复坐标；`cost` 仍按进入格子的代价累计，起点不计。
- 动态模式下 `expanded_nodes` 按关闭顺序记录 `(x, y, t)` 三元组，`expanded` 统计关闭的时空状态数；不可达时 `path`/`cost` 为 `None`。优先队列平局依次比较 f、h、x、y、t，结果不依赖任何障碍帧或坐标集合的迭代顺序。
- 校验：外层必须是序列，每一帧必须是可迭代坐标集合；字符串/字节串、非二整数坐标抛出 `TypeError`，越界坐标抛出 `ValueError`，帧内重复坐标合并；`start` 在第 0 帧被禁止时抛出 `ValueError`。所有校验在搜索开始前完成，既有校验顺序不变。
