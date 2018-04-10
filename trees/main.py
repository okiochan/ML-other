from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn import tree
import numpy as np
from data import getData

iris = load_iris()

def plot_tree_layout():
    # берем 2й и 3й признаки, 3 класса
    # pair = [2,3]
    # n_classes = 3
    # plot_colors = "ryb"
    # plot_step = 0.02
    # X = iris.data[:, pair]
    # y = iris.target
    
    n_classes = 3
    plot_colors = "rb"
    plot_step = 0.02

    # >>> clf = tree.DecisionTreeRegressor()
    # >>> clf = clf.fit(X, y)
    # >>> clf.predict([[1, 1]])
    
    # передали признаки и ответы
    # X, y = getData()
    # clf = DecisionTreeRegressor().fit(X, y)
    # clf.predict([to_predict])
    
    # передали признаки и ответы
    X, y = getData()
    clf = DecisionTreeClassifier().fit(X, y)

    # подготовили карту
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step))

    # классифицируем карту
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)

    # отрисовываем все
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=color, label=iris.target_names[i],
                    cmap=plt.cm.RdYlBu, edgecolor='black', s=80)
    plt.show()

# можно граф вывести
def save_graph():
    # берем 2й и 3й признаки, 3 класса
    pair = [2,3]
    n_classes = 3
    plot_colors = "ryb"
    plot_step = 0.02
    X = iris.data[:, pair]
    y = iris.target
    # передали признаки и ответы
    clf = DecisionTreeClassifier().fit(X, y)

    import graphviz
    dot_data = tree.export_graphviz(clf, out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render("iris")
    dot_data = tree.export_graphviz(clf, out_file=None,
                        class_names=iris.target_names,
                        filled=True, rounded=True,
                        special_characters=True)
    graph = graphviz.Source(dot_data)
    graph

# X, y = getData()
# plt.scatter(X[:,0], X[:,1], c=y)
# plt.show()
# quit()

save_graph()
plot_tree_layout()