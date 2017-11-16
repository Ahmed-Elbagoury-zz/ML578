import numpy as np
import matplotlib.pyplot as plt

def plot_cross_validation_SVM(k, n_vals, labels, mean_vals, error_vals, output_file,
                              columns_to_plot, set_yticks):
    max_y = 0
    plt.clf()
    for i in range(mean_vals.shape[1]):
        plt.clf()
        plt.errorbar(n_vals, mean_vals[:, i], yerr = error_vals[:, i])
        plt.xlabel('C')
        # plt.legend([labels[i] for i in range(mean_vals.shape[1])])
        plt.ylabel(labels[i])
        plt.title(labels[i])
        # plt.legend(labels[i])
        # plt.xlim(n_vals[0], n_vals[-1])
        plt.savefig('%s_%s.png' %(output_file, labels[i]))
    if set_yticks:
            plt.yticks(list(np.linspace(0, 5, 21)))
            plt.ylim(0, 5)