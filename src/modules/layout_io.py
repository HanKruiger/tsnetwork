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

def normalize_layout(Y):
    Y_cpy = Y.copy()
    # Translate s.t. smallest values for both x and y are 0.
    Y_cpy[:, 0] += -Y_cpy[:, 0].min()
    Y_cpy[:, 1] += -Y_cpy[:, 1].min()
    # Scale s.t. max(max(x, y)) = 1 (while keeping the same aspect ratio!)
    scaling = 1 / (np.absolute(Y_cpy).max())
    Y_cpy *= scaling
    return Y_cpy

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
            pass

        g = gt.Graph(directed=False)
        g.add_vertex(len(vertices))
        for v_i, v_j in edges:
            g.add_edge(v_i, v_j)

        gt.remove_parallel_edges(g)

        Y = np.zeros((g.num_vertices(), 2))
        for v in vertices.keys():
            Y[vertices[v]['ID'], 0] = float(vertices[v]['x'])
            Y[vertices[v]['ID'], 1] = float(vertices[v]['y'])
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

def load_ply_layout(file):
    g = gt.Graph(directed=False)

    with open(file) as f:
        all_lines = f.read().splitlines()
        it = iter(all_lines)
        
        line = next(it)
        assert(line == 'ply')

        line = next(it)
        assert(line.startswith('format ascii'))

        line = next(it)
        while not line.startswith('element'):
            line = next(it)

        words = line.split(' ')
        assert(words[0] == 'element')
        assert(words[1] == 'vertex')
        assert(words[2].isdigit())
        n_vertices = int(words[2])
        g.add_vertex(n_vertices)
        assert(g.num_vertices() == n_vertices)

        line = next(it)
        v_props = OrderedDict()
        while line.startswith('property'):
            words = line.split(' ')
            the_type = words[1]
            if the_type == 'list':
                name = words[4]
                v_props[name] = dict()
                count_type = words[2]
                entry_type = words[3]
                v_props[name]['count_type'] = count_type
                v_props[name]['entry_type'] = entry_type
            else:
                name = words[2]
                v_props[name] = dict()
            v_props[name]['type'] = the_type
            line = next(it)
        print(v_props)

        vps = dict()
        for i, v_prop in enumerate(v_props):
            name = list(v_props.keys())[i]
            the_type = v_props[name]['type']
            if the_type == 'float':
                vp = g.new_vp(the_type)
                vps[name] = vp
            else:
                raise NotImplementedError()

        print(vps)
        assert('x' in vps.keys())
        assert('y' in vps.keys())
        assert('z' in vps.keys())
        
        # Scan to next element
        while not line.startswith('element'):
            line = next(it)

        words = line.split(' ')
        assert(words[0] == 'element')
        assert(words[1] == 'face')
        assert(words[2].isdigit())
        n_faces = int(words[2])
        print(n_faces)

        line = next(it)
        f_props = OrderedDict()
        while line.startswith('property'):
            words = line.split(' ')
            the_type = words[1]
            if the_type == 'list':
                name = words[4]
                f_props[name] = dict()
                count_type = words[2]
                entry_type = words[3]
                f_props[name]['count_type'] = count_type
                f_props[name]['entry_type'] = entry_type
            else:
                name = words[2]
                f_props[name] = dict()
            f_props[name]['type'] = the_type
            line = next(it)
        print(f_props)

        while not line.startswith('end_header'):
            line = next(it)

        for i in range(n_vertices):
            line = next(it)
            words = line.split(' ')
            words = [word for word in words if word != '']
            assert(len(words) == len(v_props.keys()))
            for j, word in enumerate(words):
                name = list(v_props.keys())[j]
                the_type = v_props[name]['type']
                if the_type == 'float':
                    vps[name][i] = float(word)
                else:
                    raise NotImplementedError
        
        for _ in range (n_faces):
            line = next(it)
            words = line.split(' ')
            words = [word for word in words if word != '']
            i = 0
            for name in f_props.keys():
                the_type = f_props[name]['type']
                if the_type == 'list':
                    if f_props[name]['count_type'] == 'uchar':
                        n_items = int(words[i])
                    else:
                        raise NotImplementedError
                    the_list = [int(word) for word in words[i + 1:i + 1 + n_items]]
                    i += 1 + n_items

                    if name == 'vertex_indices':
                        for j, idx1 in enumerate(the_list):
                            idx2 = the_list[(j + 1) % len(the_list)]
                            g.add_edge(idx1, idx2)
            assert(i == len(words))


    gt.remove_parallel_edges(g)

    largest_connected_component = gt.label_largest_component(g)
    unreferenced = sum([1 for i in largest_connected_component.a if i == 0])
    if unreferenced > 0:
        g.set_vertex_filter(largest_connected_component)
        g.purge_vertices()
        print('Filtered {0} unreferenced vertices.'.format(unreferenced))
    
    if 'x' in vps.keys() and 'y' in vps.keys():
        if 'z' in vps.keys():
            Y = np.zeros((n_vertices, 3))
            for v in g.vertices():
                print(type(v))
                Y[v, 0] = vps['x'][v]
                Y[v, 1] = vps['y'][v]
                Y[v, 2] = vps['z'][v]
        else:
            Y = np.zeros((n_vertices, 2))
            for v in g.vertices():
                Y[v, 0] = vps['x'][v]
                Y[v, 1] = vps['y'][v]


    return g, Y