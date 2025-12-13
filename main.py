#!/usr/bin/env python3
"""
CSE 565 Project: Flow Decomposition into s-t paths + cycles.

I/O format (per spec):
- Input .graph:
  First line: |V| |E|
  Next E lines: u v f(e)   (integral flow on edge)
  Vertices labeled 1..|V|, s=1, t=|V|.  :contentReference[oaicite:1]{index=1}

- Output .out:
  First line: |P| |C|
  Next |P| lines: w(p) v1 v2 ... vk   (s-t path)
  Next |C| lines: w(c) v1 v2 ... v1   (cycle)  :contentReference[oaicite:2]{index=2}

Run:
  python3 main.py student_test_cases/NAME.graph
Outputs:
  outputs/NAME.out
"""

from __future__ import annotations
import os
import sys
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional

Edge = Tuple[int, int]
Flow = Dict[Edge, int]
WeightedSeq = Tuple[int, List[int]]


# ----------------------------
# Read graph
# ----------------------------
def read_flow_graph(path: str) -> Tuple[int, int, Flow]:
    flow: Flow = {}
    with open(path, "r") as f:
        header = f.readline().strip()
        if not header:
            return 0, 0, flow
        V, E = map(int, header.split())
        for _ in range(E):
            line = f.readline()
            if not line:
                break
            u, v, w = map(int, line.split())
            if w > 0:
                flow[(u, v)] = w
    return V, E, flow


# ----------------------------
# Build adjacency from positive-flow edges
# Sort neighbors by decreasing flow to bias search toward "heavier" routes.
# ----------------------------
def build_adj(flow: Flow) -> Dict[int, List[int]]:
    tmp: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for (u, v), w in flow.items():
        tmp[u].append((v, w))

    adj: Dict[int, List[int]] = {}
    for u, lst in tmp.items():
        lst.sort(key=lambda x: (-x[1], x[0]))
        adj[u] = [v for v, _ in lst]
    return adj


# ----------------------------
# Find ONE s->t path via BFS (on the current positive-flow graph)
# ----------------------------
def find_st_path(adj: Dict[int, List[int]], s: int, t: int) -> Optional[List[int]]:
    parent = {s: 0}
    q = deque([s])
    while q:
        u = q.popleft()
        if u == t:
            break
        for v in adj.get(u, []):
            if v not in parent:
                parent[v] = u
                q.append(v)
    if t not in parent:
        return None

    path: List[int] = []
    cur = t
    while cur != 0:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


# ----------------------------
# Find ONE directed cycle using iterative DFS with colors.
# Returns [v0, v1, ..., v0] or None.
# ----------------------------
def find_cycle(adj: Dict[int, List[int]], V: int) -> Optional[List[int]]:
    # 0=unvisited, 1=visiting, 2=done
    color = [0] * (V + 1)
    parent = [0] * (V + 1)

    for start in range(1, V + 1):
        if color[start] != 0:
            continue

        stack: List[Tuple[int, int]] = [(start, 0)]
        parent[start] = 0

        while stack:
            u, idx = stack[-1]
            if color[u] == 0:
                color[u] = 1  # enter

            nbrs = adj.get(u, [])
            if idx >= len(nbrs):
                color[u] = 2  # exit
                stack.pop()
                continue

            v = nbrs[idx]
            stack[-1] = (u, idx + 1)

            if color[v] == 0:
                parent[v] = u
                stack.append((v, 0))
            elif color[v] == 1:
                # Found a back edge u -> v
                cyc = [v]
                cur = u
                while cur != v and cur != 0:
                    cyc.append(cur)
                    cur = parent[cur]
                cyc.append(v)
                cyc.reverse()
                return cyc

    return None


# ----------------------------
# Extract path/cycle by bottleneck subtraction
# ----------------------------
def extract_path(path: List[int], flow: Flow, out_paths: List[WeightedSeq]) -> None:
    w = min(flow[(path[i], path[i + 1])] for i in range(len(path) - 1))
    out_paths.append((w, path))

    for i in range(len(path) - 1):
        e = (path[i], path[i + 1])
        nw = flow[e] - w
        if nw == 0:
            del flow[e]
        else:
            flow[e] = nw


def extract_cycle(cycle: List[int], flow: Flow, out_cycles: List[WeightedSeq]) -> None:
    w = min(flow[(cycle[i], cycle[i + 1])] for i in range(len(cycle) - 1))
    out_cycles.append((w, cycle))

    for i in range(len(cycle) - 1):
        e = (cycle[i], cycle[i + 1])
        nw = flow[e] - w
        if nw == 0:
            del flow[e]
        else:
            flow[e] = nw


# ----------------------------
# Cleaning / dedup:
# - merge identical paths (same vertex list)
# - merge cycles up to rotation (same cycle, different starting point)
# - drop malformed entries
# ----------------------------
def _normalize_cycle_rotation(cyc: List[int]) -> Tuple[int, ...]:
    # cyc: [a,b,c,a]
    core = cyc[:-1]
    m = min(core)
    best = None
    for i, x in enumerate(core):
        if x == m:
            rot = core[i:] + core[:i] + [m]
            t = tuple(rot)
            if best is None or t < best:
                best = t
    return best  # type: ignore


def clean_paths_and_cycles(paths: List[WeightedSeq], cycles: List[WeightedSeq], s: int, t: int) -> Tuple[List[WeightedSeq], List[WeightedSeq]]:
    P = defaultdict(int)  # tuple(path) -> total weight
    C = defaultdict(int)  # tuple(cycle canonical) -> total weight

    # Paths: must start at s, end at t, be simple
    for w, nodes in paths:
        if w <= 0 or not nodes:
            continue
        if nodes[0] != s or nodes[-1] != t:
            continue
        if len(nodes) != len(set(nodes)):  # repeated vertex in "path" => malformed
            continue
        P[tuple(nodes)] += w

    # Cycles: must be closed and simple (except repeated start/end)
    for w, nodes in cycles:
        if w <= 0 or not nodes or len(nodes) < 2:
            continue
        if nodes[0] != nodes[-1]:
            continue
        core = nodes[:-1]
        if len(core) != len(set(core)):
            continue
        canon = _normalize_cycle_rotation(nodes)
        C[canon] += w

    final_paths = [(w, list(k)) for k, w in P.items() if w > 0]
    final_cycles = [(w, list(k)) for k, w in C.items() if w > 0]

    final_paths.sort(key=lambda x: (-x[0], x[1]))
    final_cycles.sort(key=lambda x: (-x[0], x[1]))
    return final_paths, final_cycles


# ----------------------------
# Main decomposition routine
# ----------------------------
def decompose_flow(V: int, flow: Flow, s: int, t: int) -> Tuple[List[WeightedSeq], List[WeightedSeq]]:
    paths: List[WeightedSeq] = []
    cycles: List[WeightedSeq] = []

    # Phase 1: remove cycles early
    while True:
        adj = build_adj(flow)
        cyc = find_cycle(adj, V)
        if cyc is None:
            break
        extract_cycle(cyc, flow, cycles)

    # Phase 2: extract s->t paths
    while True:
        adj = build_adj(flow)
        p = find_st_path(adj, s, t)
        if p is None:
            break
        extract_path(p, flow, paths)

    # Phase 3: remove remaining cycles
    while True:
        adj = build_adj(flow)
        cyc = find_cycle(adj, V)
        if cyc is None:
            break
        extract_cycle(cyc, flow, cycles)

    # Clean and merge duplicates
    return clean_paths_and_cycles(paths, cycles, s, t)


# ----------------------------
# Output writer
# ----------------------------
def write_output(out_path: str, paths: List[WeightedSeq], cycles: List[WeightedSeq]) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(f"{len(paths)} {len(cycles)}\n")
        for w, nodes in paths:
            f.write(f"{w} " + " ".join(map(str, nodes)) + "\n")
        for w, nodes in cycles:
            f.write(f"{w} " + " ".join(map(str, nodes)) + "\n")


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python3 main.py student_test_cases/NAME.graph", file=sys.stderr)
        return 2

    in_file = sys.argv[1]
    V, E, flow = read_flow_graph(in_file)
    if V <= 0:
        print("Error: empty or malformed input file.", file=sys.stderr)
        return 1

    s, t = 1, V
    paths, cycles = decompose_flow(V, flow, s, t)

    base = os.path.splitext(os.path.basename(in_file))[0]
    out_file = os.path.join("outputs", base + ".out")
    write_output(out_file, paths, cycles)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
