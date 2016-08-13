import numpy as np
import graph_tool.all as gt

import modules.layout_io as layout_io
import modules.user_input as usr_input


# Perform an SFDP placement on the graph, and save a drawing of the layout.
def sfdp_placement(g, output_folder, color_property_map=None, ask_for_acceptance=True):
    pos_sfdp = None
    while True:
        print('[tsnetwork] Performing SFDP')
        pos = gt.sfdp_layout(g, multilevel=True, C=1.2, p=1)

        pos_sfdp = g.new_vertex_property('vector<double>')
        for v in g.vertices():
            pos_sfdp[v] = (float(pos[v][0]), float(pos[v][1]))

        print('[tsnetwork] Saving SFDP layout...')
        layout_io.save_drawing(output_folder, g, pos=pos_sfdp, description='sfdp', color_property_map=color_property_map)

        if ask_for_acceptance:
            if usr_input.confirm('[tsnetwork] Is the generated sfdp layout ({0}) acceptable? [y/n]: '.format(output_folder + '/sfdp.pdf')):
                break
        else:
            break

    # Copy SFDP vertex coordinates to Y_init
    Y_init = np.zeros((g.num_vertices(), 2))
    for v in g.vertices():
        Y_init[int(v), :] = pos_sfdp[v]

    layout_io.save_layout(output_folder + '/sfdp.pickle', g, None, Y_init)

    return Y_init, pos_sfdp
