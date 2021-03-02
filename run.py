from mdshark.main import *

if __name__ == "__main__":
    mds = MDSharkOptimizer('glc_bm_test')

    # mds.initialize_structures()

    n_iteration = 0
    mds.g_submit(n_iteration)

    pass