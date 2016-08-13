import pickle
import numpy as np
import graph_tool.all as gt
import matplotlib.cm


# Save a drawing of a layout
def save_drawing(output_folder, g, pos, description, color_property_map=None, color_array=None, formats=None, verbose=True, opacity=0.2):
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

    # Compute output size based on #vertices (Heuristic based on uniform
    # density of vertices that seems to work most of the time)
    out_size = [(g.num_vertices()) ** 0.5] * 2
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

    # Hexadecimal representation of the opacity of the edges
    opacity_string = hex(int(opacity * 255)).split('x')[1]

    for extension in formats:
        # Use the graphviz interface that graph_tool supplies to save drawings.
        if color_property_map is None:
            gt.graphviz_draw(g, fork=True, pos=pos_normalized, pin=True, penwidth=0.5, ecolor='#000000' + opacity_string, vsize=0.1, vcolor='#009900', output=output_folder + '/' + description + '.' + extension, size=(out_size[0], out_size[1]))
        else:
            gt.graphviz_draw(g, fork=True, pos=pos_normalized, pin=True, penwidth=0.5, ecolor='#000000' + opacity_string, vsize=0.1, vcmap=matplotlib.cm.hot, vcolor=color_property_map, output=output_folder + '/' + description + '.' + extension, size=(out_size[0], out_size[1]))


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


# Save a text file with the layout.
def save_layout_txt(out_file, g, Y):
    edges = list(g.edges())
    vertices = list(g.vertices())
    with open(out_file, 'w') as f:
        for i, e in enumerate(edges):
            v1 = int(e.source())
            v2 = int(e.target())
            f.write('{0}: {1} {2} {3} {4}\n'.format(i, Y[v1, 0], Y[v1, 1], Y[v2, 0], Y[v2, 1]))
        f.close()
