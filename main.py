#!/usr/bin/env python3
from __future__ import annotations

import heapq
import os
import random
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

Edge = Tuple[int, int]
Residual = Dict[Edge, int]

# ---------------- IO ----------------

def read_graph(path: str) -> Tuple[int, List[Edge], Dict[Edge, int], Dict[int, List[int]]]:
    with open(path, "r", encoding="utf-8") as f:
        n, m = map(int, f.readline().split())
        edges: List[Edge] = []
        flow: Dict[Edge, int] = {}
        adj: Dict[int, List[int]] = defaultdict(list)
        for _ in range(m):
            u, v, fe = map(int, f.readline().split())
            edges.append((u, v))
            flow[(u, v)] = fe
            adj[u].append(v)
    return n, edges, flow, adj

def write_output(out_path: str, paths: List[Tuple[int, List[int]]], cycles: List[Tuple[int, List[int]]]) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"{len(paths)} {len(cycles)}\n")
        for w, p in paths:
            f.write(f"{w} " + " ".join(map(str, p)) + "\n")
        for w, c in cycles:
            f.write(f"{w} " + " ".join(map(str, c)) + "\n")

# ---------------- Core helpers ----------------

def subtract_along(vertices: List[int], w: int, res: Residual) -> None:
    for i in range(len(vertices) - 1):
        e = (vertices[i], vertices[i + 1])
        res[e] -= w
        if res[e] == 0:
            res.pop(e, None)
        elif res[e] < 0:
            raise ValueError(f"Negative residual on {e}: {res[e]}")

def widest_path_pref_short(
    n: int, adj: Dict[int, List[int]], res: Residual, s: int, t: int
) -> Tuple[Optional[List[int]], int]:
    """
    Widest path (max bottleneck). Tie-breaker: shorter path length.
    Returns (path, bottleneck) or (None, 0).
    """
    INF = 10**18
    best_b = [0] * (n + 1)
    best_len = [10**9] * (n + 1)
    parent = [-1] * (n + 1)

    best_b[s] = INF
    best_len[s] = 0

    # max-heap: (-bottleneck, length, node)
    heap = [(-best_b[s], 0, s)]

    while heap:
        neg_b, clen, u = heapq.heappop(heap)
        b = -neg_b

        if b < best_b[u] or (b == best_b[u] and clen > best_len[u]):
            continue
        if u == t:
            break

        for v in adj.get(u, []):
            cap = res.get((u, v), 0)
            if cap <= 0:
                continue
            nb = min(b, cap)
            nlen = clen + 1
            if (nb > best_b[v]) or (nb == best_b[v] and nlen < best_len[v]):
                best_b[v] = nb
                best_len[v] = nlen
                parent[v] = u
                heapq.heappush(heap, (-nb, nlen, v))

    if best_b[t] <= 0:
        return None, 0

    # Reconstruct
    path: List[int] = []
    cur = t
    while cur != -1:
        path.append(cur)
        if cur == s:
            break
        cur = parent[cur]
    if path[-1] != s:
        return None, 0
    path.reverse()
    return path, int(best_b[t])

def find_any_cycle(n: int, adj: Dict[int, List[int]], res: Residual) -> Tuple[Optional[List[int]], int]:
    """
    DFS in the positive-residual subgraph; return (cycle with repeated start, bottleneck) or (None,0).
    """
    state = [0] * (n + 1)  # 0=unvisited, 1=stack, 2=done
    stack: List[int] = []
    idx: Dict[int, int] = {}

    def dfs(u: int) -> Optional[List[int]]:
        state[u] = 1
        idx[u] = len(stack)
        stack.append(u)

        for v in adj.get(u, []):
            if res.get((u, v), 0) <= 0:
                continue
            if state[v] == 0:
                cyc = dfs(v)
                if cyc is not None:
                    return cyc
            elif state[v] == 1:
                j = idx[v]
                return stack[j:] + [v]  # repeats start at end

        stack.pop()
        idx.pop(u, None)
        state[u] = 2
        return None

    for u in range(1, n + 1):
        if state[u] != 0:
            continue
        if not any(res.get((u, v), 0) > 0 for v in adj.get(u, [])):
            continue
        cyc = dfs(u)
        if cyc is not None:
            b = min(res[(cyc[i], cyc[i + 1])] for i in range(len(cyc) - 1))
            return cyc, int(b)

    return None, 0

def canonicalize_cycle(cyc: List[int]) -> List[int]:
    # cyc: [v0,...,vk,v0]
    core = cyc[:-1]
    k = len(core)
    best = None
    for sh in range(k):
        rot = core[sh:] + core[:sh]
        tup = tuple(rot)
        if best is None or tup < best:
            best = tup
    out = list(best)
    out.append(out[0])
    return out

def merge_identical(items: List[Tuple[int, List[int]]]) -> List[Tuple[int, List[int]]]:
    acc: Dict[Tuple[int, ...], int] = defaultdict(int)
    for w, seq in items:
        if w > 0:
            acc[tuple(seq)] += w
    merged = [(w, list(seq)) for seq, w in acc.items() if w > 0]
    merged.sort(key=lambda x: (-x[0], tuple(x[1])))
    return merged

def validate(edges: List[Edge], original: Dict[Edge, int],
             paths: List[Tuple[int, List[int]]], cycles: List[Tuple[int, List[int]]]) -> bool:
    got: Dict[Edge, int] = defaultdict(int)
    for w, p in paths:
        for i in range(len(p) - 1):
            got[(p[i], p[i + 1])] += w
    for w, c in cycles:
        for i in range(len(c) - 1):
            got[(c[i], c[i + 1])] += w
    return all(got.get(e, 0) == original.get(e, 0) for e in edges)

# ---------------- Decomposition ----------------

def decompose_once(n: int, edges: List[Edge], flow: Dict[Edge, int], adj: Dict[int, List[int]]) \
        -> Tuple[List[Tuple[int, List[int]]], List[Tuple[int, List[int]]]]:
    s, t = 1, n
    res: Residual = dict(flow)

    paths: List[Tuple[int, List[int]]] = []
    cycles: List[Tuple[int, List[int]]] = []

    # 1) widest s->t paths
    while True:
        p, w = widest_path_pref_short(n, adj, res, s, t)
        if p is None or w <= 0:
            break
        subtract_along(p, w, res)
        paths.append((w, p))

    # 2) leftover circulation -> cycles
    while True:
        cyc, w = find_any_cycle(n, adj, res)
        if cyc is None or w <= 0:
            break
        subtract_along(cyc, w, res)
        cycles.append((w, canonicalize_cycle(cyc)))

    paths = merge_identical(paths)
    cycles = merge_identical(cycles)
    return paths, cycles

def best_of_trials(n: int, edges: List[Edge], flow: Dict[Edge, int], base_adj: Dict[int, List[int]],
                   trials: int = 40, time_budget_s: float = 0.25) \
        -> Tuple[List[Tuple[int, List[int]]], List[Tuple[int, List[int]]]]:
    start = time.time()
    rng = random.Random(1234567)

    best = None  # (k, paths, cycles)
    for _ in range(trials):
        if time.time() - start > time_budget_s:
            break

        # shuffle adjacency to change tie-breaks
        adj = {u: vs[:] for u, vs in base_adj.items()}
        for u in adj:
            rng.shuffle(adj[u])

        paths, cycles = decompose_once(n, edges, flow, adj)

        if not validate(edges, flow, paths, cycles):
            continue

        k = len(paths) + len(cycles)
        if best is None or k < best[0]:
            best = (k, paths, cycles)

        # can't do better than 1 object in sane cases; early exit for very small
        if best[0] <= 2:
            break

    if best is None:
        # fallback: single attempt without shuffle
        paths, cycles = decompose_once(n, edges, flow, base_adj)
        return paths, cycles

    return best[1], best[2]

# ---------------- main ----------------

def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python3 main.py path/to/NAME.graph", file=sys.stderr)
        return 2

    in_path = sys.argv[1]
    n, edges, flow, adj = read_graph(in_path)

    paths, cycles = best_of_trials(n, edges, flow, adj)

    # ensure outputs dir
    os.makedirs("outputs", exist_ok=True)
    base = os.path.splitext(os.path.basename(in_path))[0]
    out_path = os.path.join("outputs", f"{base}.out")
    write_output(out_path, paths, cycles)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
