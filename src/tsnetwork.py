#!/bin/python3
import shutil

import graph_tool.all as gt
import numpy as np

import modules.distance_matrix as distance_matrix
import modules.graph_io as graph_io
import modules.layout_io as layout_io
import modules.thesne as thesne
import modules.user_input as usr_input
import modules.animation as animations
from modules.tsn_config import TsnConfig
from modules.sfdp_layout import sfdp_placement


def main():
    import sys
    import os.path
    import glob
    import itertools
    from argparse import ArgumentParser

    parser = ArgumentParser(description='Read a graph, and produce a layout with t-SNE.')

    # Input
    parser.add_argument('graphs', nargs='+', help='(List of) input graph(s). Or a folder with graphs.')

    # Output
    parser.add_argument('-o', default='./output', help='Folder to write output to. Default: ./output')
    parser.add_argument('--save_every', type=int, help='Save a jpg snapshot ever x epochs.')
    parser.add_argument('--render_video', action='store_true', help='Render a video of the layout evolution. Needs ImageMagick and ffmpeg.')
    parser.add_argument('--retain_snaps', action='store_true', help='Retain the snapshots. This argument is ignored if no video is rendered.')
    parser.add_argument('--save_layout_data', action='store_true', help='Save all layout coordinates in a .pickle file and a .txt file.')
    parser.add_argument('--opacity', type=float, default=0.3, help='Edge opacity.')

    # Manipulations to graph
    parser.add_argument('--strip_graph', action='store_true', help='Retain only the largest connected component in the graph.')
    parser.add_argument('--rnd_seed', '-r', type=int, nargs='+', default=[None], help='Seed for random state. (Default: Random seed)')
    parser.add_argument('--pre_sfdp', action='store_true', help='If this flag is given, the vertices will be pre-initialized with SFDP.')
    parser.add_argument('--only_sfdp', action='store_true', help='If this flag is given, only SFDP will be done.')
    parser.add_argument('--accept_all_sfdp', action='store_true', help='If this flag is given, no confirmation is asked for the SFDP layouts.')
    parser.add_argument('--remove_rnd_edges', nargs='+', type=float, default=[0], help='Mutate the graph by removing random edges. If this is used without a random seed, a random random seed will be generated. The value given to this argument is the fraction of edges that will be removed.')

    # Hyperparameters
    parser.add_argument('--n_epochs', '-e', nargs='+', type=int, default=[1000], help='One or more numbers of t-SNE epochs.')
    parser.add_argument('--lr_init', nargs='+', type=float, default=[80], help='One or more initial learning rates.')
    parser.add_argument('--lr_final', nargs='+', type=float, default=[None], help='One or more final learning rates. Default: Same as lr_init.')
    parser.add_argument('--lr_switch', nargs='+', type=int, default=[None], help='One or more learning rate switch-points.')
    parser.add_argument('--momentum_init', nargs='+', type=float, default=[0.5], help='One or more initial momenta.')
    parser.add_argument('--momentum_final', nargs='+', type=float, default=[0.5], help='One or more initial momenta.')
    parser.add_argument('--momentum_switch', nargs='+', type=int, default=[None], help='One or more momentum switch-points.')

    # Distance metric parameters
    parser.add_argument('--distance_metric', '-d', choices=['shortest_path', 'spdm', 'modified_adjacency', 'mam'], default='spdm', help='The distance metric that is used for the pairwise distances.')
    parser.add_argument('-k', nargs='+', type=float, default=[1], help='Exponent for transfer function.')

    # Cost function parameters
    #   Kullback-Leibler
    parser.add_argument('--perplexity', '-p', nargs='+', type=float, default=[80], help='One or more perplexities.')
    parser.add_argument('--l_kl_init', nargs='+', type=float, default=[1], help='One or more KL factors.')
    parser.add_argument('--l_kl_final', nargs='+', type=float, default=[1], help='One or more KL factors.')
    parser.add_argument('--l_kl_switch', nargs='+', type=int, default=[None], help='One or more KL switch-points')
    #   Edge contraction
    parser.add_argument('--l_e_init', nargs='+', type=float, default=[0], help='One or more edge contraction factors.')
    parser.add_argument('--l_e_final', nargs='+', type=float, default=[0], help='One or more edge contraction factors.')
    parser.add_argument('--l_e_switch', nargs='+', type=int, default=[None], help='One or more edge contraction switch-points')
    #   Compression
    parser.add_argument('--l_c_init', nargs='+', type=float, default=[1.2], help='One or more compression factors.')
    parser.add_argument('--l_c_final', nargs='+', type=float, default=[0], help='One or more compression factors.')
    parser.add_argument('--l_c_switch', nargs='+', type=int, default=[None], help='One or more compression switch-points')
    #   Repulsion
    parser.add_argument('--l_r_init', nargs='+', type=float, default=[0], help='One or more repulsion factors.')
    parser.add_argument('--l_r_final', nargs='+', type=float, default=[0.5], help='One or more repulsion factors.')
    parser.add_argument('--l_r_switch', nargs='+', type=int, default=[None], help='One or more repulsion switch-points')
    parser.add_argument('--r_eps', nargs='+', type=float, default=[0.2], help='Additional term in denominator to prevent near-singularities.')

    args = parser.parse_args()

    # Retrieve a list of all files in the directory, if args.graphs[0] is a directory.
    if len(args.graphs) == 1 and os.path.isdir(args.graphs[0]):
        args.graphs = glob.glob(args.graphs[0] + '/*')

    # Check graph input
    for g_file in args.graphs:
        if not os.path.isfile(g_file):
            raise FileNotFoundError(g_file + ' is not a file.')

    # Generate random random seed if none is given.
    if args.rnd_seed == [None]:
        args.rnd_seed = [np.random.randint(1e8)]

    # Ignore retain_snaps argument if no video is rendered.
    if not args.render_video:
        args.retain_snaps = True

    # Get names of the graphs (by splitting of path and extension)
    names = [os.path.split(os.path.splitext(file)[0])[1] for file in args.graphs]

    # Determine output folders. One is created in the specified output folder
    # for every graph that is supplied.
    output_folders = [args.o + '/' + name for name in names]

    # Check (and possibly create) output folders
    for folder in [args.o] + output_folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # At least everything is fine for now.
    there_were_exceptions = False

    # Loop over all graphs (and their respective output folders)
    for g_file, g_name, output_folder in zip(args.graphs, names, output_folders):
        # Load the graph
        g = graph_io.load_graph(g_file)
        print('[tsnetwork] Loaded graph {0} (|V| = {1}, |E| = {2}) into memory.'.format(g_name, g.num_vertices(), g.num_edges()))

        # Add graph name as propery in the internal representation
        g.graph_properties['name'] = g.new_graph_property('string', g_name)

        # Usually this loop has just one iteration, with only 0 as the value
        # for rmv_edge_frac (that is, no edges are removed).
        for rmv_edge_frac in args.remove_rnd_edges:
            print('[tsnetwork] Original graph: (|V|, |E|) = ({0}, {1}).'.format(g.num_vertices(), g.num_edges()))

            # Create a temporary copy of the graph that will be manipulated.
            gv = gt.GraphView(g)

            # Remove rmv_edge_frac of the graphs edges from gv.
            gv.clear_filters()
            gv.reindex_edges()
            edge_list = list(gv.edges())
            not_here_ep = gv.new_edge_property('bool', val=True)
            n_remove_edges = int(rmv_edge_frac * gv.num_edges())
            for e in np.random.randint(0, gv.num_edges(), n_remove_edges):
                not_here_ep[edge_list[e]] = False
            gv.set_edge_filter(not_here_ep)

            if n_remove_edges > 0:
                print('[tsnetwork] Removed {2} random edges: (|V|, |E|) = ({0}, {1}).'.format(gv.num_vertices(), gv.num_edges(), n_remove_edges))

            # Filter the graph s.t. only the largest connected component
            # remains.
            if args.strip_graph:
                largest_connected_component = gt.label_largest_component(gv)
                gv.set_vertex_filter(largest_connected_component)
                gv.purge_vertices()
                print('[tsnetwork] Filtered largest component: (|V|, |E|) = ({0}, {1}).'.format(gv.num_vertices(), gv.num_edges()))

            if args.pre_sfdp or args.only_sfdp:
                # Perform a SFDP layout (either as the only layout or as a
                # starting point for t-SNE.)
                Y_init, _ = sfdp_placement(gv, output_folder, ask_for_acceptance=not args.accept_all_sfdp, opacity=args.opacity)
                if args.only_sfdp:
                    continue
            else:
                # Random positions will be generated
                Y_init = None

            # Compute distance matrix of this graph with the specified metric
            X = distance_matrix.get_distance_matrix(gv, args.distance_metric)

            # Retrieve the adjacency matrix of the graph
            Adj_sparse = gt.adjacency(gv)
            Adj = np.zeros(Adj_sparse.shape, dtype='float32')
            for i, j in zip(*Adj_sparse.nonzero()):
                Adj[i, j] = Adj_sparse[i, j]

            # Make list of tsnetwork configuration objects. These are objects
            # that represent a configuration for a t-SNE layout.
            tsn_configs = []
            for perplexity, n_epochs, initial_lr, final_lr, lr_switch, initial_momentum,\
                final_momentum, momentum_switch,\
                initial_l_kl, final_l_kl, l_kl_switch,\
                initial_l_e, final_l_e, l_e_switch,\
                initial_l_c, final_l_c, l_c_switch,\
                initial_l_r, final_l_r, l_r_switch,\
                r_eps, k, rnd_seed in itertools.product(
                    args.perplexity, args.n_epochs, args.lr_init, args.lr_final,
                    args.lr_switch, args.momentum_init, args.momentum_final,
                    args.momentum_switch,
                    args.l_kl_init, args.l_kl_final, args.l_kl_switch,
                    args.l_e_init, args.l_e_final, args.l_e_switch,
                    args.l_c_init, args.l_c_final, args.l_c_switch,
                    args.l_r_init, args.l_r_final, args.l_r_switch,
                    args.r_eps, args.k, args.rnd_seed):

                # Use 50% for the switching points if no argument is given
                if lr_switch is None:
                    lr_switch = int(n_epochs * 0.5)
                if momentum_switch is None:
                    momentum_switch = int(n_epochs * 0.5)
                if l_kl_switch is None:
                    l_kl_switch = int(n_epochs * 0.5)
                if l_e_switch is None:
                    l_e_switch = int(n_epochs * 0.5)
                if l_c_switch is None:
                    l_c_switch = int(n_epochs * 0.5)
                if l_r_switch is None:
                    l_r_switch = int(n_epochs * 0.5)

                if final_lr is None:
                    final_lr = initial_lr

                cfg = TsnConfig(
                    perplexity=perplexity, n_epochs=n_epochs,
                    initial_lr=initial_lr, final_lr=final_lr, lr_switch=lr_switch,
                    initial_momentum=initial_momentum,
                    final_momentum=final_momentum, momentum_switch=momentum_switch,
                    initial_l_kl=initial_l_kl, final_l_kl=final_l_kl, l_kl_switch=l_kl_switch,
                    initial_l_e=initial_l_e, final_l_e=final_l_e, l_e_switch=l_e_switch,
                    initial_l_c=initial_l_c, final_l_c=final_l_c, l_c_switch=l_c_switch,
                    initial_l_r=initial_l_r, final_l_r=final_l_r, l_r_switch=l_r_switch,
                    r_eps=r_eps, k=k, pre_sfdp=args.pre_sfdp, rmv_edge_frac=rmv_edge_frac,
                    rnd_seed=rnd_seed, distance_matrix=args.distance_metric
                )

                # Do no add the configurations that already have files matching
                # the description, unless the user confirms to overwrite.
                if any([file.startswith(cfg.get_description() + '.') for file in os.listdir(output_folder)]):
                    if not usr_input.confirm('[tsnetwork] ' + cfg.get_description() + ' files exists! Overwrite?'):
                        continue
                tsn_configs.append(cfg)

            # Loop over the t-SNE configurations for a single graph
            for cfg in tsn_configs:
                print('[tsnetwork] Processing: ' + cfg.get_description())

                # String that has the path to the directory where the snapshots
                # will come. (If --save_every is given)
                snaps_dir = output_folder + '/snaps_' + cfg.get_description()

                # Clean out existing snaps directory if it exists.
                if args.save_every is not None and os.path.exists(snaps_dir):
                    if usr_input.confirm('[tsnetwork] ' + snaps_dir + ' exists. Delete contents?'):
                        for file in os.listdir(snaps_dir):
                            file_path = os.path.join(snaps_dir, file)
                            try:
                                if os.path.isfile(file_path):
                                    os.unlink(file_path)
                                elif os.path.isdir(file_path):
                                    shutil.rmtree(file_path)
                            except Exception as e:
                                print(e)
                elif args.save_every is not None and not os.path.exists(snaps_dir):
                    # Make folder for snaps, if it is necessary and it doesn't
                    # exist yet.
                    os.makedirs(snaps_dir)

                # Apply the transfer function
                X_transfered = X ** cfg.k

                # Try to do the tsne layout.
                try:
                    Y, costs = thesne.tsne(X_transfered, random_state=cfg.rnd_seed, perplexity=cfg.perplexity, n_epochs=cfg.n_epochs,
                                           Y=Y_init,
                                           initial_lr=cfg.initial_lr, final_lr=cfg.final_lr, lr_switch=cfg.lr_switch,
                                           initial_momentum=cfg.initial_momentum, final_momentum=cfg.final_momentum, momentum_switch=cfg.momentum_switch,
                                           initial_l_kl=cfg.initial_l_kl, final_l_kl=cfg.final_l_kl, l_kl_switch=cfg.l_kl_switch,
                                           initial_l_e=cfg.initial_l_e, final_l_e=cfg.final_l_e, l_e_switch=cfg.l_e_switch,
                                           initial_l_c=cfg.initial_l_c, final_l_c=cfg.final_l_c, l_c_switch=cfg.l_c_switch,
                                           initial_l_r=cfg.initial_l_r, final_l_r=cfg.final_l_r, l_r_switch=cfg.l_r_switch,
                                           r_eps=cfg.r_eps, Adj=Adj, g=gv, snaps_output_folder=snaps_dir, save_every=args.save_every)
                except (thesne.NaNException, thesne.SigmaTooLowException) as e:
                    there_were_exceptions = True
                    print('[exception] {0}'.format(e))

                    # Also write exception to a file.
                    with open(output_folder + '/exception_' + cfg.get_description() + '.out', 'w') as f:
                        print('{0}'.format(e), file=f)
                        f.close()
                    print('[tsnetwork] Continuing with next TsnConfig.')
                    continue

                # Render an animation of the snapshots
                if args.render_video:
                    animations.save_animation(snaps_dir, cfg.get_description())

                # Remove the directory with snapshots.
                if args.save_every is not None and not args.retain_snaps and os.path.exists(snaps_dir):
                    print('[tsnetwork] Cleaning up snaps directory.')
                    shutil.rmtree(snaps_dir)

                # Save the data (graph, vertex coordinates)
                if args.save_layout_data:
                    layout_io.save_vna_layout(output_folder + '/layout_' + cfg.get_description() + '.vna', gv, Y)
                    layout_io.save_layout_txt(output_folder + '/layout_edges_' + cfg.get_description() + '.txt', gv, Y)

                # Save final drawing of the layout
                layout_io.save_drawing(output_folder, gv, Y.T, cfg.get_description(), formats=['jpg', 'pdf'], edge_colors="rgb", draw_vertices=False, opacity=args.opacity)

    if there_were_exceptions:
        print('[tsnetwork] Done! However, be wary. There were exceptions.')
    else:
        print('[tsnetwork] Done!')


if __name__ == '__main__':
    main()
