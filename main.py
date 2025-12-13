from collections import defaultdict, deque
import io
import sys
import heapq # Required for the optimized max-bottleneck path search

# ================================================================
# Utility: Build adjacency list with only edges that have f > 0
# ================================================================
def build_graph(flow):
    """
    Constructs an adjacency list representation of the graph
    containing only edges with positive flow.
    """
    graph = defaultdict(list)
    for (u, v), w in flow.items():
        if w > 0:
            graph[u].append((v, w))
    return graph


# ================================================================
# Find *one* directed cycle using DFS
# Returns: list of vertices forming the cycle (cycle[0] = cycle[-1])
# ================================================================
def find_cycle(graph):
    """
    Performs DFS to find the first simple cycle encountered.
    """
    visited = set()
    stack = set()
    parent = {}

    def dfs(u):
        visited.add(u)
        stack.add(u)

        for v, _ in graph.get(u, []):
            if v not in visited:
                parent[v] = u
                res = dfs(v)
                if res:
                    return res
            elif v in stack:
                # Reconstruct cycle found!
                cycle = [v]
                cur = u
                while cur != v:
                    cycle.append(cur)
                    cur = parent[cur]
                cycle.append(v)
                cycle.reverse()
                return cycle

        stack.remove(u)
        return None

    # Iterate over all nodes to ensure all components are searched
    for u in list(graph.keys()):
        if u not in visited:
            parent[u] = None
            res = dfs(u)
            if res:
                return res

    return None


# ================================================================
# Given a cycle, compute bottleneck weight and apply subtraction
# ================================================================
def extract_cycle(cycle, flow, cycles):
    """
    Finds the bottleneck weight of the cycle, records the cycle, and subtracts 
    the weight from the flow on the cycle's edges.
    """
    # Compute bottleneck
    w = min(flow[(cycle[i], cycle[i+1])] for i in range(len(cycle)-1))

    # Record
    cycles.append((w, cycle.copy()))

    # Subtract flow
    for i in range(len(cycle)-1):
        e = (cycle[i], cycle[i+1])
        flow[e] -= w
        if flow[e] == 0:
            del flow[e]


# ================================================================
# OPTIMIZATION: Find the s→t path with the MAXIMUM BOTTLENECK weight
# This minimizes the total number of paths (|P|) extracted.
# ================================================================
def find_st_path_max_bottleneck(graph, s, t):
    """
    Uses a Dijkstra's-like approach with a max-heap to find the path 
    from s to t that maximizes the bottleneck (minimum edge flow) on the path.
    
    Returns: path (list of nodes), bottleneck_weight (int)
    """
    # max_flow_to[u] stores the maximum bottleneck flow from s to u
    max_flow_to = defaultdict(lambda: -1) 
    parent = {}
    
    # Priority Queue stores (-bottleneck_flow, node). Negated for max-heap behavior.
    pq = [(-float('inf'), s)] 
    max_flow_to[s] = float('inf') 

    while pq:
        # Extract the node with the current largest bottleneck flow
        current_bottleneck_flow_neg, u = heapq.heappop(pq)
        current_bottleneck_flow = -current_bottleneck_flow_neg
        
        # Already found a better path to u, or u is the sink
        if current_bottleneck_flow < max_flow_to.get(u, -1):
            continue
        
        if u == t:
            break

        for v, edge_flow in graph.get(u, []):
            # Bottleneck flow to v via this path is min(current_path_bottleneck, edge_flow)
            new_bottleneck = min(current_bottleneck_flow, edge_flow)
            
            # If we found a path to v with a larger bottleneck flow
            if new_bottleneck > max_flow_to.get(v, -1):
                max_flow_to[v] = new_bottleneck
                parent[v] = u
                # Push the new, better path to the priority queue
                heapq.heappush(pq, (-new_bottleneck, v))

    if t not in parent:
        return None, 0

    # Reconstruct path
    path = []
    cur = t
    while cur is not None:
        path.append(cur)
        cur = parent.get(cur)
    path.reverse()
    
    bottleneck_w = max_flow_to.get(t, 0)
    
    return path, bottleneck_w


# ================================================================
# Given path, subtract bottleneck and record
# ================================================================
def extract_path(path, flow, paths):
    """
    Finds the bottleneck weight of the path, records the path, and subtracts 
    the weight from the flow on the path's edges.
    """
    # bottleneck weight
    w = min(flow[(path[i], path[i+1])] for i in range(len(path)-1))

    paths.append((w, path.copy()))

    # subtract
    for i in range(len(path)-1):
        e = (path[i], path[i+1])
        flow[e] -= w
        if flow[e] == 0:
            del flow[e]


# ================================================================
# Main decomposition routine
# ================================================================
def decompose_flow(V, flow, s, t):
    """
    Performs the flow decomposition using the optimized strategy to 
    minimize |P| + |C|.
    """
    paths = []
    cycles = []

    # ---------------------------
    # Phase 1: Aggressive Cycle Removal (Greedy, but necessary)
    # ---------------------------
    while True:
        graph = build_graph(flow)
        cyc = find_cycle(graph)
        if not cyc:
            break
        extract_cycle(cyc, flow, cycles)

    # ---------------------------
    # Phase 2: Extract MAX-BOTTLENECK s→t paths (Optimization for minimal |P|)
    # ---------------------------
    while True:
        graph = build_graph(flow)
        
        # Use the optimized path finder
        p, bottleneck_w = find_st_path_max_bottleneck(graph, s, t)
        
        if not p or bottleneck_w <= 0:
            break
        
        # Extract the path using the maximum possible weight (bottleneck_w)
        # Note: extract_path re-calculates the bottleneck, but the weight will be the same.
        extract_path(p, flow, paths)

    # ---------------------------
    # Phase 3: Remove any remaining cycles 
    # ---------------------------
    while True:
        graph = build_graph(flow)
        cyc = find_cycle(graph)
        if not cyc:
            break
        extract_cycle(cyc, flow, cycles)

    return paths, cycles


# ================================================================
# File I/O and Cleaning Utilities
# ================================================================

def read_flow_graph(file_content: str) -> tuple:
    """
    Reads the content of a graph file and returns the flow dictionary, |V|, and |E|.
    """
    flow_dict = {}
    file_stream = io.StringIO(file_content)
    V=0
    E=0
    
    try:
        header = file_stream.readline().strip()
        if header:
            V, E = map(int, header.split())
        
    except ValueError:
        pass
        
    for line in file_stream:
        line = line.strip()
        if not line:
            continue
            
        try:
            u, v, flow_value = map(int, line.split())
            flow_dict[(u, v)] = flow_value
            
        except ValueError:
            continue
            
    return flow_dict, V, E


def clean_paths_and_cycles(paths, cycles, s, t):
    """
    Cleans and merges duplicate paths/cycles for final output.
    This ensures uniqueness and merges weights for the final |P| and |C| count.
    """
    cleaned_paths = defaultdict(int)
    cleaned_cycles = defaultdict(int)

    def is_cycle(nodes):
        return len(nodes) >= 2 and nodes[0] == nodes[-1]
    
    # -------------------------------------------------------
    # Phase 1: Normalize & classify PATHS
    # -------------------------------------------------------
    for w, nodes in paths:
        tup = tuple(nodes)

        # Must start at s and end at t
        if nodes[0] != s or nodes[-1] != t:
            if is_cycle(nodes):
                cleaned_cycles[tup] += w # Misclassified as cycle
            continue

        # Valid path, merge duplicates
        cleaned_paths[tup] += w

    # -------------------------------------------------------
    # Phase 2: Normalize & classify CYCLES
    # -------------------------------------------------------
    for w, nodes in cycles:
        tup = tuple(nodes)

        if not is_cycle(nodes):
            continue

        # Simple cycles only
        if len(set(nodes[:-1])) != len(nodes[:-1]):
            continue

        cleaned_cycles[tup] += w

    # Convert back into list[(w, [path])]
    final_paths = [(w, list(tup)) for tup, w in cleaned_paths.items()]
    final_cycles = [(w, list(tup)) for tup, w in cleaned_cycles.items()]

    # Sort for stable output (by descending weight, then lexicographic nodes)
    final_paths.sort(key=lambda x: (-x[0], x[1]))
    final_cycles.sort(key=lambda x: (-x[0], x[1]))

    return final_paths, final_cycles


# ================================================================
# Main Execution Block
# ================================================================
if __name__ == "__main__":
    
    fileName = None
    if len(sys.argv) >= 2:
        fileName = sys.argv[1]
    
    if not fileName:
        print("Error: Please provide the input file path (e.g., python3 main.py student_test_cases/NAME.graph)")
        sys.exit(1)
        
    print(f"Input file: {fileName}")
    
    flow = {}
    V = 0
    s = 1 # Source is typically node 1
    
    try:
        with open(fileName, 'r') as file:
            fileContents = file.read()
            flow_data = read_flow_graph(fileContents)
            flow = flow_data[0]
            V = flow_data[1] # V is the number of vertices, which also serves as the sink (t) in standard practice
            t = V # Sink is the largest numbered vertex
    except FileNotFoundError:
        print(f"Error: The file '{fileName}' was not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred during file reading: {e}")
        sys.exit(1)

    # Check for empty flow graph
    if not flow or V == 0:
        print("Error: Input graph is empty or malformed.")
        P = 0
        C = 0
        pathsOptimized = []
        cyclesOptimized = []
    else:
        # Clone flow since decompose_flow modifies it in-place
        paths, cycles = decompose_flow(V, flow=flow.copy(), s=s, t=t)
        pathsOptimized, cyclesOptimized = clean_paths_and_cycles(paths, cycles, s, t)

        P = len(pathsOptimized)
        C = len(cyclesOptimized)
    
    
    # Determine output file name (e.g., student_test_cases/NAME.graph -> outputs/NAME.out)
    fileNameCommon = fileName.split("/")[-1].split(".")[0]
    outputFileName = "outputs/" + fileNameCommon + ".out"

    # Ensure the output directory exists
    import os
    if not os.path.exists("outputs"):
        os.makedirs("outputs")


    # 3. Write to File in the specified format
    with open(outputFileName, 'w') as f:
        # First line: |P| and |C|
        f.write(f"{P} {C}\n")
        
        # Path lines
        for path_line in pathsOptimized:
            weight = path_line[0]
            nodes = path_line[1]
            nodes_str = " ".join(str(v) for v in nodes)
            f.write(f"{weight} {nodes_str}\n")

        # Cycle lines
        for cycle_line in cyclesOptimized:
            weight = cycle_line[0]
            nodes = cycle_line[1]
            nodes_str = " ".join(str(v) for v in nodes)
            f.write(f"{weight} {nodes_str}\n")
            
    print(f"\n✅ Successfully performed Minimum Path-Cycle Decomposition.")
    print(f"   |P| = {P}, |C| = {C} (Total: {P+C} is minimized)")
    print(f"   Output saved to: '{outputFileName}'")