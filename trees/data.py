from nn import nn, scale
import matplotlib.pyplot as plt
import numpy as np, json

def normal(centerx, centery, disp, N):
    x = disp * np.random.randn(N) + centerx
    y = disp * np.random.randn(N) + centery
    return np.column_stack((x, y))

def gen(n, m, N):
    data = np.zeros((1, 3))
    np.random.seed(3)
    for i in range(n):
        for j in range(m):
            cls = np.ones(N) * (1 if ((i + j) % 2 == 0) else 0)
            tmp = normal(i, j, 1/5*1.5, N)
            tmp = np.column_stack((tmp, cls))
            data = np.row_stack((data, tmp))
    return data[1:,:]

def getData(rescale=True, for_neural_network=False):
    N = 100
    square = 2
    ret = gen(square, square, N)
    x, y = ret[:,:2], ret[:,2]
    if rescale:
        scaling_data = scale.normal_scaling_data(x)
        x = scale.rescale(x, scaling_data)
    if for_neural_network:
        y = y.reshape(y.size, 1)
    return x, y


if __name__ == "__main__":
    inp, out = getData()
    l = inp.shape[0]
    colors = []
    for i in range(l):
        if out[i]==-1:
            colors.append('red')
        else:
            colors.append('green')
    plt.scatter(inp[:,0], inp[:,1], color=colors, s=50)
    plt.show()
