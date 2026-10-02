# Route Planner

A dependency-free Python reference implementation for robotics, path-planning, a-star.

Run with: python3 demo.py
Tests: python3 -m unittest discover -s tests -v

## Scope

实现一个确定性的二维栅格 A* 路径规划器。规划器支持障碍物、四邻域移动、固定的 tie-break 规则和不可达结果；返回路径、总代价及扩展节点统计，不能越过障碍或产生重复节点。公开数据结构和边界行为应当适合离线回放，后续可在同一基线上扩展代价地图和动态障碍处理。

## API 约定

`plan(width, height, blocked, start, goal, costs=None)` 返回 `{"path", "cost", "expanded"}`。

**输入校验（搜索开始前完成）**
- `width`、`height` 必须为正整数。
- `start`、`goal` 及 `blocked` 中每个元素必须是两个整数构成的坐标（元组或长度为二的列表等序列，统一规范化为元组）。
- `costs` 可选；不提供或为 `None` 时按单位代价工作。提供时必须是 `height` 行 × `width` 列的二维序列（字符串/字节串除外），每个单元格为正整数，表示进入该格子的代价。
- 类型或结构不符抛出 `TypeError`；尺寸非正、坐标越界、端点落在障碍物上、代价矩阵形状不符或单元格代价非正抛出 `ValueError`。
- `blocked` 中的重复坐标被合并，不影响结果。

**搜索语义**
- 四邻域移动，曼哈顿启发式。每步代价为进入格子的代价（默认 1，提供 `costs` 时为 `costs[y][x]`）；起点格子的代价不计入总代价。
- 平局次序固定且公开：先比较 f，再比较 h，再按 (x, y) 字典序；结果不依赖 `blocked` 的迭代顺序。
- `expanded` 只统计从优先队列取出并首次关闭的节点；成功时计入 goal，`start == goal` 时返回单节点路径、代价 0、`expanded` 为 1；不可达时 `path`/`cost` 为 `None`，`expanded` 等于实际关闭的可通行节点数。
