import matplotlib.pyplot as plt

def plot_beta(ns, test_mses, intercept=None, beta=None, errorbars=False):
    if errorbars:
        plt.errorbar(ns, test_mses.mean(axis=0), yerr=test_mses.var(axis=0))
    else:
        plt.scatter(ns, test_mses.mean(axis=0))
    plt.xscale('log')
    plt.xlabel("N")
    plt.ylabel("Test MSE")
    plt.yscale('log')
    if beta is not None:
        plt.plot(ns, 10**intercept.cpu()*ns**(-beta.cpu().numpy()+1))