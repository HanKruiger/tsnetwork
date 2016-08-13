import os.path
from scipy.io import mmread
import graph_tool.all as gt
import csv


# Read a Matrix Market file, and construct an undirected weighted graph from it.
def load_mm(mm_file):
    adj = mmread(mm_file)

    assert adj.shape[0] == adj.shape[1]

    # Initialize graph
    g = gt.Graph(directed=False)

    edge_weight = g.edge_properties["weight"] = g.new_edge_property("double")

    # Create vertex for every row/column
    g.add_vertex(adj.shape[0])

    print('[graph_io] Reading matrix market file with {0} explicit elements...'.format(len(adj.data)))

    # Loop over all explicit elements in the sparse matrix
    for iteration, (i, j, w) in enumerate(zip(adj.row, adj.col, adj.data)):
        # Skip self-edges.
        if i == j:
            continue

        # Add edge to the graph, if its 'symmetric partner' is not already there.
        # (Undirected graph, so g.edge(i, j) == g.edge(j, i))
        if g.edge(i, j) is None:
            g.add_edge(i, j)

        edge_weight[i, j] = w

        # Print progress every 5%
        if iteration % (int(0.05 * len(adj.data))) == 0:
            perc = 100 * iteration / len(adj.data)
            print('[graph_io] {0:.1f}%'.format(perc), end='\r')
    print('\n[graph_io] Done!')
    return g


# Read a csv file, and construct an undirected weighted graph from it.
def load_csv(csv_file_name):
    g = gt.Graph(directed=False)

    num_lines = sum(1 for _ in open(csv_file_name))

    # Property map for label that was used in the file (different from internal graph-tool index!)
    v_label = g.vertex_properties['label'] = g.new_vertex_property("string")
    # Dictionary from label to graph-tool index
    v_dict = {}

    print('[graph_io] Reading csv-file with ' + str(num_lines) + ' lines...')
    with open(csv_file_name) as csv_file:
        reader = csv.DictReader(csv_file, delimiter='\t')
        for iteration, row in enumerate(reader):
            from_node = row['FromNodeId']
            to_node = row['ToNodeId']

            if from_node not in v_dict.keys():
                v = g.add_vertex()
                v_label[v] = from_node
                v_dict[from_node] = v
            if to_node not in v_dict.keys():
                v = g.add_vertex()
                v_label[v] = to_node
                v_dict[to_node] = v

            i = v_dict[from_node]
            j = v_dict[to_node]

            # Skip self-edges.
            if i == j:
                continue

            # Add edge to the graph, if its 'symmetric partner' is not already there.
            # (Undirected graph, so g.edge(i, j) == g.edge(j, i))
            if g.edge(i, j) is None:
                g.add_edge(i, j)

            # Print progress every 5%
            if iteration % (int(0.05 * num_lines - 1)) == 0:
                perc = 100 * iteration / num_lines - 1
                print('[graph_io] {0:.1f}%'.format(perc), end='\r')

    print('\n[graph_io] Done!')
    return g


# Read a chaco file, and construct an undirected weighted graph from it.
def load_chaco(file_name):
    g = gt.Graph(directed=False)

    num_lines = sum(1 for _ in open(file_name))

    print('[graph_io] Reading chaco-file with ' + str(num_lines) + ' lines...')
    with open(file_name, 'r') as f:
        meta = [int(num) for num in f.readline().split()]
        if len(meta) > 3:
            raise Exception('Chaco parser cannot read this file.')
        if len(meta) == 3:
            if meta[2] != 0:
                raise Exception('Chaco parser cannot read this file.')

        # Add this many vertices.
        g.add_vertex(meta[0])

        for i, line in enumerate(f):
            # Minus 1 to convert to 0-indexing, j != i to skip self-edges
            to_nodes = [int(j) - 1 for j in line.split() if int(j) != i]

            # Add edge to the graph, if its 'symmetric partner' is not already there.
            # (Undirected graph, so g.edge(i, j) == g.edge(j, i))
            for j in to_nodes:
                if g.edge(i, j) is None:
                    g.add_edge(i, j)

            # Print progress every 5%
            if i % (int(0.05 * num_lines - 1)) == 0:
                perc = 100 * i / num_lines - 1
                print('[graph_io] {0:.1f}%'.format(perc), end='\r')

        assert g.num_vertices() == meta[0]
        assert g.num_edges() == meta[1]

    print('\n[graph_io] Done!')
    return g


def load_graph(file):
    if os.path.splitext(file)[1] == '.mtx':
        g = load_mm(file)
    elif os.path.splitext(file)[1] == '.csv':
        g = load_csv(file)
    elif os.path.splitext(file)[1] == '.graph':
        g = load_chaco(file)
    else:
        # Give the file to graph_tool and hope for the best.
        g = gt.load_graph(file)

        g.set_directed(False)

    gt.remove_parallel_edges(g)

    return g
