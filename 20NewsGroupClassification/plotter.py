import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, NullFormatter

def plot2d_hist(xstr, ystr, x_data, y_data, data):
    heatmap = plt.pcolor(data)
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            plt.text(x + 0.5, y + 0.5, '%.4f\n%s = %s\n%s = %s' % (data[y, x], xstr, x_data[x], ystr, y_data[y]),horizontalalignment='center', verticalalignment='center', clip_on=True)
    plt.colorbar(heatmap)
    plt.show()

def plot_matrix(matrix):
    fig, ax = plt.subplots()
    min_val, max_val = 0, matrix.shape[0]
    ax.matshow(matrix)
    ax.set_xticks(np.arange(max_val), minor=True)
    ax.set_yticks(np.arange(max_val), minor=True)
    ax.yaxis.set_minor_formatter(FormatStrFormatter("%s"))
    ax.xaxis.set_minor_formatter(FormatStrFormatter("%s"))
    ax.yaxis.set_major_formatter(NullFormatter())
    ax.xaxis.set_major_formatter(NullFormatter())

    for i in np.arange(max_val):
        for j in np.arange(max_val):
            ax.text(i,j,str(matrix[j,i]), va='center', ha='center',fontsize = 6)
    plt.show()

def plot3d_hist(xstr, ystr, x_data, y_data, data, zstr, z_data):
    print(data.shape[0])
    for i in np.arange(data.shape[0]):
        datum = data[i]
        plt.subplot(1, data.shape[0], i+1)
        plt.title("{} = {}".format(zstr, z_data[i]))
        heatmap = plt.pcolor(datum)
        for y in range(datum.shape[0]):
            for x in range(datum.shape[1]):
                plt.text(x + 0.5, y + 0.5, '%.4f\n%s = %s\n%s = %s' % (datum[y, x], xstr, x_data[x], ystr, y_data[y]),horizontalalignment='center', verticalalignment='center', clip_on=True)
        plt.colorbar(heatmap)
    plt.tight_layout()
    plt.show()

def test_plotter():
    gamma = np.array([0.1, 1, 10])
    c = np.array([0.1, 100])
    c_lasso = np.array([1000, 2000])
    data = np.random.randn(c_lasso.shape[0],c.shape[0], gamma.shape[0])
    plot3d_hist("gamma", "c", gamma, c, data, "c_lasso", c_lasso)  

if __name__ == '__main__':
    test_plotter()  
