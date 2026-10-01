# Route Planner

A dependency-free Python reference implementation for robotics, path-planning, a-star.

Run with: python3 demo.py
Tests: python3 -m unittest discover -s tests -v

## Scope

实现一个确定性的二维栅格 A* 路径规划器。规划器支持障碍物、四邻域移动、固定的 tie-break 规则和不可达结果；返回路径、总代价及扩展节点统计，不能越过障碍或产生重复节点。公开数据结构和边界行为应当适合离线回放，后续可在同一基线上扩展代价地图和动态障碍处理。
