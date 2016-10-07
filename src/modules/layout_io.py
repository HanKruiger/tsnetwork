import pickle
import numpy as np
import graph_tool.all as gt
import matplotlib.cm


# Save a drawing of a layout
def save_drawing(output_folder, g, pos, description, color_property_map=None, color_array=None, formats=None, verbose=True, opacity=0.2, edge_colors=None, draw_vertices=True):
    if formats is None:
        formats = ['jpg', 'pdf']

    # GraphViz needs the positions to be between 0 and 1. So normalize first.
    pos_normalized = g.new_vertex_property('vector<double>')
    vertices = list(g.vertices())
    if type(pos) is not np.ndarray:
        Y = pos.get_2d_array(range(2))
    else:
        Y = pos
    # Translate s.t. smallest values for both x and y are 0.
    Y[0, :] += -Y[0, :].min()
    Y[1, :] += -Y[1, :].min()
    # Scale s.t. max(max(x, y)) = 1 (while keeping the same aspect ratio!)
    scaling = 1 / (np.absolute(Y).max())
    Y *= scaling
    pos_normalized.set_2d_array(Y)

    # Output size in cm (matches UF images)
    out_size = [14.3] * 2

    # Crop for aspect ratio
    if max(Y[0, :]) < max(Y[1, :]):
        out_size[0] *= max(Y[0, :])
    else:
        out_size[1] *= max(Y[1, :])

    # Use the supplied color array for the vertex colors, if given.
    if color_array is not None:
        color_property_map = g.new_vertex_property('double')
        assert len(color_array) == g.num_vertices()
        for v in vertices:
            color_property_map[int(v)] = color_array[int(v)]

    if verbose:
        print('[layout_io] Saving layout drawing... ({0})'.format(description))

    
    if edge_colors == "rgb":
        edge_color = g.new_edge_property('string')
        edge_length = g.new_edge_property('float')
        edges = list(g.edges())
        for e in edges:
            v1 = e.source()
            v2 = e.target()
            length = ((Y[:, int(v1)] - Y[:, int(v2)]) ** 2).sum() ** 0.5
            edge_length[e] = length
        lengths = edge_length.get_array()
        for e in edges:
            # Colour coding the edges based on edge length
            x = (edge_length[e] - np.min(lengths)) / (np.max(lengths) - np.min(lengths))
            red = min(max(0, 1 - 2 * x), 1)
            green = max(0, 1 - abs(2 * x - 1))
            blue = min(max(0, -1 + 2 * x), 1)
            edge_color[e] = "#"
            edge_color[e] += "{0:0{1}x}".format(int(red * 255), 2)
            edge_color[e] += "{0:0{1}x}".format(int(green * 255), 2)
            edge_color[e] += "{0:0{1}x}".format(int(blue * 255), 2)
            edge_color[e] += "{0:0{1}x}".format(int(opacity * 255), 2)
    else:
        edge_color = "#000000{0:0{1}x}".format(int(opacity * 255), 2)

    for extension in formats:
        # Use the graphviz interface that graph_tool supplies to save drawings.
        if not draw_vertices:
            # For this to work correctly, the gt.graphviz_draw implementation needs some tweaking:
            #   * Edge attribute headclip: set to "false"
            #   * Edge attribute tailclip: set to "false"
            #   * Node attribute shape: set to "none"
            gt.graphviz_draw(g, fork=True, pos=pos_normalized, pin=True, penwidth=1, ecolor=edge_color, vsize=0, vcolor='#00ff0000', output=output_folder + '/' + description + '.' + extension, size=tuple(out_size))
        elif color_property_map is None:
            gt.graphviz_draw(g, fork=True, pos=pos_normalized, pin=True, penwidth=1, ecolor=edge_color, vsize=0.1, vcolor='#009900', output=output_folder + '/' + description + '.' + extension, size=tuple(out_size))
        else:
            gt.graphviz_draw(g, fork=True, pos=pos_normalized, pin=True, penwidth=1, ecolor=edge_color, vsize=0.1, vcmap=matplotlib.cm.hot, vcolor=color_property_map, output=output_folder + '/' + description + '.' + extension, size=tuple(out_size))


# Save a pickle file with the serialized graph, distance matrix, and layout.
def save_layout(out_file, g, X, Y):
    with open(out_file, 'wb') as f:
        pickle.dump(g, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(X, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(Y, f, pickle.HIGHEST_PROTOCOL)
        f.close()


# Read a pickle file with the serialized graph, distance matrix, and layout.
def load_layout(in_file):
    with open(in_file, 'rb') as f:
        g = pickle.load(f)
        X = pickle.load(f)
        Y = pickle.load(f)
        f.close()
    return g, X, Y

# Read from a file exported by Tulip. Coordinates are in the 'graphics' vertex property.
def load_tulip_layout(in_file):
    g = gt.load_graph(in_file)
    g.set_directed(False)
    gt.remove_parallel_edges(g)

    graphics = g.vertex_properties['graphics']
    Y = np.zeros((g.num_vertices(), 2))
    for i in range(g.num_vertices()):
        Y[i, :] = [graphics[i]['x'], graphics[i]['y']]
    return g, Y


def load_vna_layout(in_file):
    with open(in_file) as f:
        all_lines = f.read().splitlines()

        it = iter(all_lines)

        # Ignore preamble
        line = next(it)
        while not line.startswith('*Node properties'):
            line = next(it)
        
        node_properties = next(it).split(' ')
        assert('ID' in node_properties and 'x' in node_properties and 'y' in node_properties)

        vertices = dict()
        line = next(it)
        gt_idx = 0 # Index for gt
        while not line.startswith('*Tie data'):
            entries = line.split(' ')
            vna_id = entries[0]
            vertex = dict()
            for i, prop in enumerate(node_properties):
                vertex[prop] = entries[i]
            vertex['ID'] = gt_idx # Replace VNA ID by numerical gt index
            vertices[vna_id] = vertex # Retain VNA ID as key of the vertices dict
            
            gt_idx += 1
            line = next(it)
        print(vertices)

        edge_properties = next(it).split(' ')
        assert(edge_properties[0] == 'from' and edge_properties[1] == 'to')

        edges = []
        try:
            while True:
                line = next(it)
                entries = line.split(' ')
                v_i = vertices[entries[0]]['ID']
                v_j = vertices[entries[1]]['ID']
                edges.append((v_i, v_j))
        except StopIteration:
            print('Finished reading file. Making gt.Graph...')

        g = gt.Graph(directed=False)
        g.add_vertex(len(vertices))
        for v_i, v_j in edges:
            g.add_edge(v_i, v_j)

        gt.remove_parallel_edges(g)

        Y = np.zeros((g.num_vertices(), 2))
        for v in vertices.keys():
            Y[vertices[v]['ID'], 0] = float(vertices[v]['x'])
            Y[vertices[v]['ID'], 1] = float(vertices[v]['y'])
        print(Y.shape)
        pos = g.new_vertex_property('vector<double>')
        pos.set_2d_array(Y.T)

        return g, Y
    return None

def save_vna_layout(out_file, g, Y):
    with open(out_file, 'w') as f:
        f.write('*Node properties\n')
        f.write('ID x y\n')
        for v in g.vertices():
            x = Y[int(v), 0]
            y = Y[int(v), 1]
            f.write('{0} {1} {2}\n'.format(int(v), x, y))
        f.write('*Tie data\n')
        f.write('from to\n')
        for v1, v2 in g.edges():
            f.write('{0} {1}\n'.format(int(v1), int(v2)))
        f.close()



# Save a text file with the (edge-based) layout.
def save_layout_txt(out_file, g, Y):
    edges = list(g.edges())
    vertices = list(g.vertices())
    with open(out_file, 'w') as f:
        for i, e in enumerate(edges):
            v1 = int(e.source())
            v2 = int(e.target())
            f.write('{0}: {1} {2} {3} {4}\n'.format(i, Y[v1, 0], Y[v1, 1], Y[v2, 0], Y[v2, 1]))
        f.close()
