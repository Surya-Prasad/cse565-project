#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import time
import random
import heapq
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# ---------- Types ----------
Edge = Tuple[int, int]
Flow = Dict[Edge, int]
Adj = Dict[int, List[int]]

@dataclass
class Obj:
    kind: str           # "P" or "C"
    verts: List[int]    # path: v1..vk ; cycle: v1..vk..v1
    M: int              # big-M upper bound on weight
    incid: List[int]    # indices of edges used in edge_list


# ---------- IO ----------
def read_graph(path: str) -> Tuple[int, List[Edge], Flow, Adj]:
    with open(path, "r", encoding="utf-8") as f:
        n, m = map(int, f.readline().split())
        edges: List[Edge] = []
        flow: Flow = {}
        adj: Adj = defaultdict(list)
        for _ in range(m):
            u, v, fe = map(int, f.readline().split())
            e = (u, v)
            edges.append(e)
            flow[e] = fe
            adj[u].append(v)
    return n, edges, flow, adj


def write_out(path: str, paths: List[Tuple[int, List[int]]], cycles: List[Tuple[int, List[int]]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{len(paths)} {len(cycles)}\n")
        for w, p in paths:
            f.write(str(w) + " " + " ".join(map(str, p)) + "\n")
        for w, c in cycles:
            f.write(str(w) + " " + " ".join(map(str, c)) + "\n")


# ---------- Basic decomposition primitives (used for candidate generation + repair) ----------
def subtract_along(verts: List[int], w: int, res: Flow) -> None:
    for i in range(len(verts) - 1):
        e = (verts[i], verts[i + 1])
        res[e] -= w
        if res[e] == 0:
            res.pop(e, None)
        elif res[e] < 0:
            raise ValueError(f"Residual negative on {e}: {res[e]}")


def widest_path(n: int, adj: Adj, res: Flow, s: int, t: int) -> Tuple[Optional[List[int]], int]:
    """
    Widest path: maximize bottleneck = min residual along path.
    Tie-break implicitly via heap; we randomize adjacency upstream.
    """
    INF = 10**18
    best = [0] * (n + 1)
    parent = [-1] * (n + 1)
    best[s] = INF
    heap = [(-INF, s)]
    while heap:
        negb, u = heapq.heappop(heap)
        b = -negb
        if b < best[u]:
            continue
        if u == t:
            break
        for v in adj.get(u, []):
            cap = res.get((u, v), 0)
            if cap <= 0:
                continue
            nb = min(b, cap)
            if nb > best[v]:
                best[v] = nb
                parent[v] = u
                heapq.heappush(heap, (-nb, v))
    if best[t] <= 0:
        return None, 0
    # reconstruct
    path = []
    cur = t
    while cur != -1:
        path.append(cur)
        if cur == s:
            break
        cur = parent[cur]
    if path[-1] != s:
        return None, 0
    path.reverse()
    return path, int(best[t])


def find_any_cycle(n: int, adj: Adj, res: Flow) -> Tuple[Optional[List[int]], int]:
    """
    DFS cycle-finder in residual>0 subgraph.
    Returns cycle with repeated start: [v0,...,vk,v0]
    """
    state = [0] * (n + 1)  # 0 unvisited, 1 in stack, 2 done
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
                return stack[j:] + [v]
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


def validate(edges: List[Edge], flow: Flow,
             paths: List[Tuple[int, List[int]]], cycles: List[Tuple[int, List[int]]]) -> bool:
    got = defaultdict(int)
    for w, p in paths:
        for i in range(len(p) - 1):
            got[(p[i], p[i + 1])] += w
    for w, c in cycles:
        for i in range(len(c) - 1):
            got[(c[i], c[i + 1])] += w
    return all(got.get(e, 0) == flow.get(e, 0) for e in edges)


# ---------- Candidate generation ----------
def verts_to_edge_indices(verts: List[int], edge_index: Dict[Edge, int]) -> List[int]:
    idxs = []
    for i in range(len(verts) - 1):
        idxs.append(edge_index[(verts[i], verts[i + 1])])
    return idxs


def generate_candidates(n: int, edges: List[Edge], flow: Flow, base_adj: Adj,
                        max_paths: int = 250, max_cycles: int = 250,
                        trials: int = 15, time_budget_s: float = 0.5) -> List[Obj]:
    """
    Build a pool of candidate paths/cycles by repeatedly decomposing with randomized tie-breaks.
    """
    start = time.time()
    edge_index = {e: i for i, e in enumerate(edges)}
    cand_map: Dict[Tuple[str, Tuple[int, ...]], Obj] = {}

    rng = random.Random(12345)

    for _ in range(trials):
        if time.time() - start > time_budget_s:
            break

        # shuffled adjacency
        adj = {u: vs[:] for u, vs in base_adj.items()}
        for u in adj:
            rng.shuffle(adj[u])

        res = dict(flow)
        s, t = 1, n

        # paths
        pcount = 0
        while pcount < max_paths:
            p, w = widest_path(n, adj, res, s, t)
            if p is None or w <= 0:
                break
            # candidate path, with big-M = bottleneck in ORIGINAL flow (safe upper bound)
            M = min(flow.get((p[i], p[i+1]), 0) for i in range(len(p)-1))
            key = ("P", tuple(p))
            if key not in cand_map:
                cand_map[key] = Obj("P", p, M, verts_to_edge_indices(p, edge_index))
            subtract_along(p, w, res)
            pcount += 1

        # cycles from leftover
        ccount = 0
        while ccount < max_cycles:
            cyc, w = find_any_cycle(n, adj, res)
            if cyc is None or w <= 0:
                break
            cyc = canonicalize_cycle(cyc)
            M = min(flow.get((cyc[i], cyc[i+1]), 0) for i in range(len(cyc)-1))
            key = ("C", tuple(cyc))
            if key not in cand_map:
                cand_map[key] = Obj("C", cyc, M, verts_to_edge_indices(cyc, edge_index))
            subtract_along(cyc, w, res)
            ccount += 1

    return list(cand_map.values())


# ---------- LP / ILP solve ----------
def solve_milp_or_lp(edges: List[Edge], flow: Flow, cands: List[Obj], use_milp: bool) -> Tuple[List[float], List[float]]:
    """
    Returns (y, w) for each candidate.
    If use_milp=True: y binary, w continuous >=0.
    Else: LP relaxation y in [0,1].
    Requires PuLP.
    """
    try:
        import pulp  # type: ignore
    except Exception as e:
        raise RuntimeError("PuLP not installed. Install with: pip install pulp") from e

    m = len(cands)
    E = len(edges)

    prob = pulp.LpProblem("flow_rep_min_objects", pulp.LpMinimize)

    # variables
    y = [pulp.LpVariable(f"y_{j}", lowBound=0, upBound=1,
                         cat=("Binary" if use_milp else "Continuous")) for j in range(m)]
    w = [pulp.LpVariable(f"w_{j}", lowBound=0, cat="Continuous") for j in range(m)]

    # objective: minimize number of used objects
    prob += pulp.lpSum(y)

    # edge constraints: sum_j A_ej * w_j == f(e)
    # (A_ej is 1 if candidate j uses edge e)
    use_list = [[] for _ in range(E)]
    for j, obj in enumerate(cands):
        for ei in obj.incid:
            use_list[ei].append(j)

    for ei, e in enumerate(edges):
        prob += pulp.lpSum(w[j] for j in use_list[ei]) == flow[e], f"edge_{ei}"

    # linking: w_j <= M_j * y_j
    for j, obj in enumerate(cands):
        # If M accidentally becomes 0 (shouldn't for meaningful candidates), guard it
        Mj = max(0, int(obj.M))
        prob += w[j] <= Mj * y[j], f"link_{j}"

    # solve
    # CBC is the default MILP solver for pulp; for LP it also works.
    status = prob.solve(pulp.HiGHS_CMD(msg=False))
    if pulp.LpStatus[status] not in ("Optimal", "Feasible"):
        raise RuntimeError(f"Solver status: {pulp.LpStatus[status]}")

    y_val = [float(v.value()) for v in y]
    w_val = [float(v.value()) for v in w]
    return y_val, w_val


# ---------- Rounding + Repair ----------
def apply_solution_to_flow(edges: List[Edge], flow: Flow, cands: List[Obj], w_int: List[int]) -> Flow:
    """Compute leftover residual after subtracting integer-weight candidates."""
    res = dict(flow)
    for j, obj in enumerate(cands):
        wj = w_int[j]
        if wj <= 0:
            continue
        for i in range(len(obj.verts) - 1):
            e = (obj.verts[i], obj.verts[i + 1])
            res[e] -= wj
            if res[e] == 0:
                res.pop(e, None)
            elif res[e] < 0:
                raise ValueError("Overshot an edge flow during rounding/repair.")
    return res


def repair_with_decomposition(n: int, base_adj: Adj, residual: Flow) -> Tuple[List[Tuple[int, List[int]]], List[Tuple[int, List[int]]]]:
    """
    Guaranteed valid repair:
    - decompose residual into widest s->t paths then cycles (all integral).
    """
    adj = {u: vs[:] for u, vs in base_adj.items()}
    paths: List[Tuple[int, List[int]]] = []
    cycles: List[Tuple[int, List[int]]] = []
    s, t = 1, n
    res = dict(residual)

    while True:
        p, w = widest_path(n, adj, res, s, t)
        if p is None or w <= 0:
            break
        subtract_along(p, w, res)
        paths.append((w, p))

    while True:
        cyc, w = find_any_cycle(n, adj, res)
        if cyc is None or w <= 0:
            break
        cyc = canonicalize_cycle(cyc)
        subtract_along(cyc, w, res)
        cycles.append((w, cyc))

    if any(v > 0 for v in res.values()):
        raise RuntimeError("Repair failed: leftover positive residual remains.")
    return paths, cycles


def lp_round_and_repair(n: int, edges: List[Edge], flow: Flow, adj: Adj,
                        cands: List[Obj], y: List[float], w: List[float],
                        tau: float = 0.5) -> Tuple[List[Tuple[int, List[int]]], List[Tuple[int, List[int]]]]:
    """
    One simple rounding scheme:
      1) keep candidates with y >= tau
      2) set integer weight = floor(w)
      3) repair leftover exactly using decomposition
    """
    m = len(cands)

    w_int = [0] * m
    for j in range(m):
        if y[j] >= tau and w[j] > 1e-9:
            w_int[j] = int(w[j])  # floor

    residual = apply_solution_to_flow(edges, flow, cands, w_int)

    # Convert selected candidates into output lists
    out_paths: List[Tuple[int, List[int]]] = []
    out_cycles: List[Tuple[int, List[int]]] = []
    for j, obj in enumerate(cands):
        if w_int[j] <= 0:
            continue
        if obj.kind == "P":
            out_paths.append((w_int[j], obj.verts))
        else:
            out_cycles.append((w_int[j], obj.verts))

    # Repair leftover
    rep_paths, rep_cycles = repair_with_decomposition(n, adj, residual)
    out_paths.extend(rep_paths)
    out_cycles.extend(rep_cycles)

    # (Optional) merge identical sequences
    out_paths = merge_same(out_paths)
    out_cycles = merge_same(out_cycles)

    return out_paths, out_cycles


def merge_same(items: List[Tuple[int, List[int]]]) -> List[Tuple[int, List[int]]]:
    acc = defaultdict(int)
    for w, v in items:
        acc[tuple(v)] += w
    merged = [(w, list(v)) for v, w in acc.items() if w > 0]
    merged.sort(key=lambda x: (-x[0], tuple(x[1])))
    return merged


# ---------- Main driver ----------
def solve_instance(graph_path: str, out_dir: str = "outputs",
                   use_milp: bool = False,
                   tau: float = 0.5) -> str:
    n, edges, flow, adj = read_graph(graph_path)

    # 1) candidates
    cands = generate_candidates(n, edges, flow, adj,
                               max_paths=300, max_cycles=300,
                               trials=25, time_budget_s=0.8)

    if not cands:
        raise RuntimeError("No candidates generated (unexpected).")

    # 2) LP or MILP
    y, w = solve_milp_or_lp(edges, flow, cands, use_milp=use_milp)

    # 3) rounding + repair
    paths, cycles = lp_round_and_repair(n, edges, flow, adj, cands, y, w, tau=tau)

    if not validate(edges, flow, paths, cycles):
        raise RuntimeError("Final output failed validation.")

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(graph_path))[0]
    out_path = os.path.join(out_dir, base + ".out")
    write_out(out_path, paths, cycles)
    return out_path


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python3 ilp_lp_rounding.py path/to/NAME.graph [--milp] [--tau=0.6]", file=sys.stderr)
        return 2

    graph_path = sys.argv[1]
    use_milp = any(a == "--milp" for a in sys.argv[2:])
    tau = 0.5
    for a in sys.argv[2:]:
        if a.startswith("--tau="):
            tau = float(a.split("=", 1)[1])

    out_path = solve_instance(graph_path, use_milp=use_milp, tau=tau)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
