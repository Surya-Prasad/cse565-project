import sys
import os
import glob

def parse_input(file_path):
    try:
        with open(file_path, 'r') as f:
            lines = f.read().split()
            
        iterator = iter(lines)
        num_vertices = int(next(iterator))
        num_edges = int(next(iterator))
        
        # Dictionary to store flow: key=(u, v), value=flow
        network_flow = {}
        
        for _ in range(num_edges):
            u = int(next(iterator))
            v = int(next(iterator))
            capacity = int(next(iterator))
            
            # Check for duplicate edges if necessary, otherwise set flow
            if (u, v) in network_flow:
                network_flow[(u, v)] += capacity
            else:
                network_flow[(u, v)] = capacity
                
        return num_vertices, num_edges, network_flow
    except StopIteration:
        raise ValueError("Input file is incomplete or formatted incorrectly.")
    except Exception as e:
        raise ValueError(f"Error parsing input file: {e}")


def parse_output(file_path):
    try:
        with open(file_path, 'r') as f:
            # Read line by line to handle variable length paths/cycles
            lines = [l.strip() for l in f.readlines() if l.strip()]
            
        if not lines:
            raise ValueError("Output file is empty.")

        header = lines[0].split()
        num_paths = int(header[0])
        num_cycles = int(header[1])
        
        paths = []
        cycles = []
        
        current_line_idx = 1
        
        # Parse Paths
        for _ in range(num_paths):
            if current_line_idx >= len(lines):
                raise ValueError("Output file missing path definitions.")
            parts = list(map(int, lines[current_line_idx].split()))
            weight = parts[0]
            path_nodes = parts[1:] # The sequence of vertices
            paths.append({'weight': weight, 'nodes': path_nodes})
            current_line_idx += 1
            
        # Parse Cycles
        for _ in range(num_cycles):
            if current_line_idx >= len(lines):
                raise ValueError("Output file missing cycle definitions.")
            parts = list(map(int, lines[current_line_idx].split()))
            weight = parts[0]
            cycle_nodes = parts[1:] # The sequence of vertices
            cycles.append({'weight': weight, 'nodes': cycle_nodes})
            current_line_idx += 1
            
        return paths, cycles
    except Exception as e:
        raise ValueError(f"Error parsing output file: {e}")


def validate_test(graph_file, truth_file):

    V, E, original_flow = parse_input(graph_file)
    paths, cycles = parse_output(truth_file)

    if V > 50 or E > 100 or max(original_flow.values(), default=0) > 1000:
        return False
    
    if len(paths) > 20 or len(cycles) > 20:
        return False
    
    if verify_solution(graph_file, truth_file, truth_file, True) == 0:
        return False
    
    return True


def verify_solution(input_file, truth_file, output_file, test):
    """
    Verifies if the output is a valid flow decomposition of the input.
    """
    print(f"Verifying {output_file} against {input_file}...")
    try:
        V, _, original_flow = parse_input(input_file)
        paths, cycles = parse_output(output_file)
        with open(truth_file, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]    
        if not lines:
            raise ValueError("Truth file is empty.")
        header = lines[0].split()
        num_paths_opt = int(header[0])
        num_cycles_opt = int(header[1])

        if not test:
            if not validate_test(input_file, truth_file):
                print("FAIL: Test constraints violated.")
                return 40
            else:
                print("PASS: Test constraints validated.")

    except ValueError as e:
        print(f"FAIL: Parsing Error - {e}")
        return False

    source = 1
    sink = V

    calculated_flow = {edge: 0 for edge in original_flow}

    for idx, p in enumerate(paths):
        w = p['weight']
        nodes = p['nodes']
        
        if nodes[0] != source or nodes[-1] != sink:
            print(f"FAIL: Path {idx+1} does not start at Source (1) or end at Sink ({sink}). Path: {nodes}")
            return False
            
        for i in range(len(nodes) - 1):
            u, v = nodes[i], nodes[i+1]
            if (u, v) not in calculated_flow:
                print(f"FAIL: Path {idx+1} uses non-existent edge ({u}, {v}).")
                return False
            calculated_flow[(u, v)] += w

    for idx, c in enumerate(cycles):
        w = c['weight']
        nodes = c['nodes']
        
        for i in range(len(nodes)-1):
            u = nodes[i]
            v = nodes[(i + 1)]
            
            if (u, v) not in calculated_flow:
                print(f"FAIL: Cycle {idx+1} uses non-existent edge ({u}, {v}).")
                return False
            calculated_flow[(u, v)] += w

    is_valid = True
    for edge, flow in original_flow.items():
        calc = calculated_flow[edge]
        if calc != flow:
            print(f"FAIL: Flow mismatch on edge {edge}. Expected {flow}, got {calc}.")
            is_valid = False
            
    if is_valid:
        print("SUCCESS: The solution is a valid flow decomposition.")
        print(f"Total Paths: {len(paths)}, Total Cycles: {len(cycles)}, Net Total: {len(paths) + len(cycles)}")
        score = max(1, 40 + num_paths_opt + num_cycles_opt - (len(paths) + len(cycles)))
        return score
    else:
        return 0


if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("Usage: python score.py <test_input_file> <student_output_file>")
        sys.exit(1)
    
    test_file_path = sys.argv[1]
    output_file_path = sys.argv[2]

    graph_files = glob.glob(os.path.join(test_file_path, "*.graph"))

    total_score = 0
    
    for graph_file in graph_files:
        base_name = os.path.splitext(os.path.basename(graph_file))[0]
        truth_file = os.path.join(test_file_path, f"{base_name}.truth")
        student_output_file = os.path.join(output_file_path, f"{base_name}.out")
        
        if not os.path.exists(truth_file):
            print(f"Truth file {truth_file} does not exist. Skipping...")
            continue

        if not os.path.exists(student_output_file):
            print(f"Output file {student_output_file} does not exist. Skipping...")
            continue
        try:
            score = verify_solution(graph_file, truth_file, student_output_file, False)
            total_score += score
        except Exception as e:
            print(f"Execution Error for {graph_file}: {e}")
    
    score_file = "test_scores.txt"

    try:
        with open(score_file, 'r') as sf:
            content = sf.read().strip()
            current_score = int(content) if content else 0
    except FileNotFoundError:
        current_score = 0
    
    new_score = current_score + total_score
    with open(score_file, 'w') as sf:
        sf.write(f"{new_score}\n")
    
    print(f"Total Score across all tests: {total_score}")
    print(f"Final Total Score across all tests: {new_score}")