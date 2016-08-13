# Class that represents the configuration of a tsnetwork placement.
class TsnConfig:

    def __init__(self, perplexity, n_epochs, initial_lr, final_lr, lr_switch,
                 initial_momentum, final_momentum, momentum_switch,
                 initial_l_kl, final_l_kl, l_kl_switch,
                 initial_l_e, final_l_e, l_e_switch,
                 initial_l_c, final_l_c, l_c_switch,
                 initial_l_r, final_l_r, l_r_switch,
                 r_eps, k, pre_sfdp, rmv_edge_frac, rnd_seed, distance_matrix):
        self.perplexity = perplexity
        self.n_epochs = n_epochs

        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.lr_switch = lr_switch

        self.initial_momentum = initial_momentum
        self.final_momentum = final_momentum
        self.momentum_switch = momentum_switch

        self.initial_l_kl = initial_l_kl
        self.final_l_kl = final_l_kl
        self.l_kl_switch = l_kl_switch

        self.initial_l_e = initial_l_e
        self.final_l_e = final_l_e
        self.l_e_switch = l_e_switch

        self.initial_l_c = initial_l_c
        self.final_l_c = final_l_c
        self.l_c_switch = l_c_switch

        self.initial_l_r = initial_l_r
        self.final_l_r = final_l_r
        self.l_r_switch = l_r_switch

        self.r_eps = r_eps

        self.k = k
        self.pre_sfdp = pre_sfdp
        self.rmv_edge_frac = rmv_edge_frac
        self.rnd_seed = rnd_seed
        self.distance_matrix = distance_matrix

    # Return a description string of the configuration.
    # Parts are commented out to prevent too long filenames.
    def get_description(self):
        description = 'e_{:0>5d}'.format(self.n_epochs)
        description += '_p_{:0>.1f}'.format(self.perplexity)
        description += '_lrni_{:.1f}'.format(self.initial_lr)
        # description += '_lrnf_{:.1f}'.format(self.final_lr)
        # description += '_mi_{:.1f}'.format(self.initial_momentum)
        # description += '_mf_{:.1f}'.format(self.final_momentum)
        # description += '_lkli_{:.2f}'.format(self.initial_l_kl)
        # description += '_lklf_{:.2f}'.format(self.final_l_kl)
        description += '_lei_{:.2f}'.format(self.initial_l_e)
        # description += '_lef_{:.2f}'.format(self.final_l_e)
        description += '_lci_{:.2f}'.format(self.initial_l_c)
        # description += '_lcf_{:.2f}'.format(self.final_l_c)
        # description += '_lri_{:.2f}'.format(self.initial_l_r)
        description += '_lrf_{:.2f}'.format(self.final_l_r)
        description += '_reps_{:.2f}'.format(self.r_eps)
        description += '_k_{:0>.2f}'.format(self.k)
        # description += '_rer_{:.2f}'.format(self.rmv_edge_frac)
        description += '_rs{:d}'.format(self.rnd_seed)
        description += '_' + self.distance_matrix

        if self.pre_sfdp:
            description = 'pre_sfdp_' + description

        # S.t. LaTeX doesn't complain about the extension.
        description = description.replace('.', 'p')
        return description
