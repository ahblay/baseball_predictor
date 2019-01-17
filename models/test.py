from models.model import Model
import matplotlib.pyplot as plt
from sklearn.feature_selection import chi2, f_classif
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import namedtuple
from setup_logger import logger
from pprint import pprint as pp
from statistics import stdev, mean


class Test:
    def __init__(self, model):
        self.model = model
        self.methods = [chi2, f_classif]
        self.X = model.X
        self.y = model.y

    def bar_graph(self, data):
        pp(data)
        num_groups = len(data)
        accuracy_by_method = {}
        stdev_by_method = {}
        method_names = [key for key in data.pop()]
        # TODO: Ensure colors change independent of number of methods
        colors = ['b', 'r']
        for method in method_names:
            accuracy_by_method[method] = []
            stdev_by_method[method] = []
            for dict in data:
                accuracy_by_method[method].append(dict[method][0])
                stdev_by_method[method].append(dict[method][1])
        fig, ax = plt.subplots(figsize=(15, 5))
        plt.ylim(0.45, 0.65)

        index = np.arange(num_groups - 1)
        bar_width = 0.35

        opacity = 0.4
        error_config = {'ecolor': '0.3'}

        counter = 0

        for method in method_names:
            i = method_names.index(method)
            print(len(index))
            print(len(accuracy_by_method[method]))
            print(bar_width)
            print(method)
            ax.bar(index + counter,
                   accuracy_by_method[method],
                   bar_width,
                   yerr=stdev_by_method[method],
                   alpha=opacity,
                   color=colors[i],
                   error_kw=error_config,
                   label=method)

            counter += bar_width

        ax.set_xlabel('Max Best Features')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy by feature count and function')
        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels(())
        ax.legend()

        fig.tight_layout()
        plt.show()

    def test_k_best(self):
        results = []
        seeds = [i for i in range(0, 25)]
        for k in range(1, 40):
            methods = {}
            for method in self.methods:
                all_seeds = []
                for seed in seeds:
                    logger.info(f"Running model. KBest = {k}, method = {str(method.__name__)}")
                    self.model.X = self.X
                    self.model.y = self.y
                    self.model.k_best(method, k)
                    self.model.split_data(0.2, seed)
                    self.model.standard_scale()
                    self.model.lda(1)
                    self.model.fit_clf()
                    self.model.pred_clf()
                    ac, cm, cr = self.model.eval_clf()
                    logger.info("Completed run.")
                    all_seeds.append(ac)
                methods[str(method.__name__)] = (mean(all_seeds), stdev(all_seeds))
                results.append(methods)
        self.bar_graph(results)
        return results
