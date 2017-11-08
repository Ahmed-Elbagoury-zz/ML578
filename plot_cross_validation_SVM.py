import numpy as np
import matplotlib.pyplot as plt

def plot_cross_validation_SVM(n_vals, labels, vals, output_file,
                              columns_to_plot, set_yticks):
    max_y = 0
    for column_index in range(len(vals[0])):
        plt.plot(n_vals, list(np.array(vals)[:, column_index]),
                 label=labels[column_index])
	plt.xlabel('C')
	plt.legend([labels[i] for i in columns_to_plot], loc = "center right")
	plt.xlim(0, n_vals[-1])
	if set_yticks:
            plt.yticks(list(np.linspace(0, 5, 21)))
            plt.ylim(0, 5)
	# plt.title('Trained using %s dataset, with batch size = 10, %d nodes'%(dataset_name, num_nodes)) 
	plt.savefig(output_file)	
	plt.clf()
