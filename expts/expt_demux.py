from kernels import GaussianKernel, LaplaceKernel, ReluNTK

def expt_demux(expt_id):
    # Defaults
    N_SAMPLES = 80_000
    N_KERNEL = 25_000
    N_TRAIN_MAX = 15_000
    N_TEST = 5_000
    P_MODES = 30_000
    DATASET = "gaussian"
    DATA_DIM = 200
    DATA_EIGVAL_EXP = 2.0
    ZCA_STRENGTH = 0
    NORMALIZE = False
    TARGET = "powerlaws"
    NUM_MARKERS = 0
    KERNEL_TYPE = GaussianKernel
    KERNEL_WIDTH = 4
    RIDGE = 1e-3
    FEWER_TRIALS = False
    
    if expt_id == 1:
        DATASET = "cifar5m"
        TARGET = "vehicle"
    if expt_id == 2:
        DATASET = "cifar5m"
        TARGET = "domesticated"
    if expt_id == 3:
        DATASET = "cifar5m"
        TARGET = "plane-frog"
    if expt_id == 4:
        DATASET = "cifar5m"
        TARGET = "car-ship"
    if expt_id == 5:
        DATASET = "cifar5m"
        TARGET = "bird-cat"
    if expt_id == 6:
        DATASET = "cifar5m"
        TARGET = "dog-else"
    if expt_id == 7:
        DATASET = "cifar5m"
        TARGET = "deer-horse"
    if expt_id == 8:
        DATASET = "imagenet32"
        N_SAMPLES = 50_000
        TARGET = "monomials"
        KERNEL_TYPE = LaplaceKernel
        NUM_MARKERS = 250
        ZCA_STRENGTH = 1e-3
        NORMALIZE = True
        FEWER_TRIALS = True
    if expt_id == 9:
        DATASET = "cifar5m"
        TARGET = "monomials"
        KERNEL_WIDTH = 8
    if expt_id == 10:
        DATASET = "cifar5m"
        TARGET = "monomials"
        KERNEL_TYPE = LaplaceKernel
        NUM_MARKERS = 250
        ZCA_STRENGTH = 3e-3
        NORMALIZE = True
        FEWER_TRIALS = True
    if expt_id == 11:
        DATASET = "svhn"
        TARGET = "evenodd"
        KERNEL_WIDTH = 10
        NORMALIZE = True
    if expt_id == 12:
        DATASET = "svhn"
        TARGET = "loops"
        KERNEL_WIDTH = 10
        NORMALIZE = True
    if expt_id == 13:
        DATASET = "svhn"
        TARGET = "primes"
        KERNEL_WIDTH = 10
        NORMALIZE = True
    if expt_id == 14:
        DATASET = "svhn"
        TARGET = "4-2"
        KERNEL_WIDTH = 10
        NORMALIZE = True
    if expt_id == 15:
        DATASET = "svhn"
        TARGET = "4-9"
        KERNEL_WIDTH = 10
        NORMALIZE = True
    if expt_id == 16:
        DATASET = "svhn"
        TARGET = "0-else"
        KERNEL_WIDTH = 10
        NORMALIZE = True
    if expt_id == 17:
        DATASET = "cifar5m"
        TARGET = "vehicle"
        KERNEL_TYPE = LaplaceKernel
        ZCA_STRENGTH = 5e-3
        NORMALIZE = True
    if expt_id == 18:
        DATASET = "cifar5m"
        TARGET = "cat-else"
        KERNEL_TYPE = LaplaceKernel
        ZCA_STRENGTH = 5e-3
        NORMALIZE = True
    if expt_id == 19:
        DATASET = "cifar5m"
        TARGET = "dog-frog"
        KERNEL_TYPE = LaplaceKernel
        ZCA_STRENGTH = 5e-3
        NORMALIZE = True
    if expt_id == 20:
        DATASET = "cifar5m"
        TARGET = "car-truck"
        KERNEL_TYPE = LaplaceKernel
        ZCA_STRENGTH = 5e-3
        NORMALIZE = True
    if expt_id == 21:
        DATASET = "imagenet32"
        N_SAMPLES = 50_000
        TARGET = "powerlaws"
        KERNEL_TYPE = ReluNTK
        ZCA_STRENGTH = 1e-2
        NORMALIZE = True
    if expt_id == 22:
        DATASET = "svhn"
        TARGET = "evenodd"
        ZCA_STRENGTH = 1e-3
    if expt_id == 23:
        DATASET = "svhn"
        TARGET = "loops"
        ZCA_STRENGTH = 1e-3
    if expt_id == 24:
        DATASET = "svhn"
        TARGET = "4-2"
        ZCA_STRENGTH = 1e-3
    if expt_id == 25:
        DATASET = "svhn"
        TARGET = "3-7"
        ZCA_STRENGTH = 1e-3
    if expt_id == 26:
        DATASET = "cifar5m"
        TARGET = "monomials"
    if expt_id == 27:
        DATASET = "imagenet32"
        N_SAMPLES = 50_000
        TARGET = "powerlaws"
        KERNEL_TYPE = LaplaceKernel
        ZCA_STRENGTH = 1e-2
        NORMALIZE = True
    
    hypers = dict(
        n_samples = N_SAMPLES,
        n_kernel = N_KERNEL,
        n_train_max = N_TRAIN_MAX,
        n_test = N_TEST,
        p_modes = P_MODES,
        dataset = DATASET,
        data_dim = DATA_DIM,
        data_eigval_exp = DATA_EIGVAL_EXP,
        zca_strength = ZCA_STRENGTH,
        normalize = NORMALIZE,
        target = TARGET,
        num_markers = NUM_MARKERS,
        kernel_name = KERNEL_TYPE.__name__,
        kernel_width = KERNEL_WIDTH,
        ridge = RIDGE,
        fewer_trials = FEWER_TRIALS
    )
    return hypers