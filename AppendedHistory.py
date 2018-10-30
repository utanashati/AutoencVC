import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
from matplotlib.ticker import MaxNLocator
from operator import itemgetter

def get_legend(colormap, vars_, varname, lines=True, alphas=[1, 0.5], linestyles=['--', '-']):
    """Creates a legend with patches and lines for plotting learning curves."""
    
    color_iter = iter(colormap)
    color=next(color_iter)
    legend = []
    if lines:
        legend.append(mlines.Line2D([], [], color=color, alpha=alphas[0], linestyle=linestyles[0], label='Training'))
        legend.append(mlines.Line2D([], [], color=color, alpha=alphas[1], linestyle=linestyles[1], label='Validation'))

    color_iter = iter(colormap)
    for var in vars_:
        legend.append(mpatches.Patch(color=next(color_iter), label=var))

    return legend

class AppendedHistory:
    """A class for saving and plotting histories for multiple iterations
    of a set of models with a varied hyperparameter (variable)."""
    
    def __init__(self, varname='var', keys=['loss', 'acc'], fname=None):
        self.__varname = varname
        self.__keys = keys
        if fname:
            with open(fname, 'r') as file:
                self.__history = json.load(file)
                f_keys = []
                for key in self.__history.keys():
                    if key.endswith('loss'):
                        if 'loss' not in f_keys:
                            f_keys.append('loss')
                    elif key.endswith('acc'):
                        if 'acc' not in f_keys:
                            f_keys.append('acc')
                    else:
                        self.__varname = key
                self.__keys = f_keys
        else:
            self.__history = { self.__varname: [] }
            for key in keys:
                for val in ['', 'val_']:
                    self.__history[val + key] = []

    def get_varname(self):
        return self.__varname

    def get_keys(self):
        return self.__keys

    def get_history(self):
        return self.__history

    def change_varname(self, varname):
        self.__history[varname] = self.__history[self.__varname]
        del self.__history[self.__varname]
        self.__varname = varname
        
    def append_hist(self, var, history):
        """Append fitting history for a new variable."""
        for key in self.__history.keys():
            if key != self.__varname:
                self.__history[key].append(history.history[key])
            else:
                self.__history[self.__varname].append(var)
        
    def add_hist(self, var, history):
        """Add further fitting history to a certain variable."""
        ind = self.__history[self.__varname].index(var)
        for key in self.__history.keys():
            if key != self.__varname:
                self.__history[key][ind] = self.__history[key][ind] + history.history[key]

    def order(self, reverse=False):
    	sorted_inds, sorted_vars = zip(*sorted([(i,e) for i,e in enumerate(self.__history[self.__varname])], key=itemgetter(1), reverse=reverse))
    	for key in self.__history.keys():
    		self.__history[key] = [self.__history[key][i] for i in sorted_inds]
                
    def merge(self, history):
        """Merge 2 histories."""
        for key in self.__history.keys():
            self.__history[key] = self.__history[key] + history.history[key]
        
    def save(self, fname):
        with open(fname, 'w') as file:
            file.write(json.dumps(self.__history))
            
    def get_single_var(self, var):
        """Get the history for a single variable as a new object."""
        new_hist = AppendedHistory()
        ind = self.__history[self.__varname].index(var)
        for key in self.__history.keys():
            new_hist.__history[key] = self.__history[key][ind]
        return new_hist
    
    def get_slice(self, slice_):
        """Get the histories for a given slice of variables as a new object."""
        new_hist = AppendedHistory()
        for key in self.__history.keys():
            if type(slice_) == int:
                new_hist.__history[key] = [self.__history[key][slice_]]
            else:
                new_hist.__history[key] = self.__history[key][slice_]
        return new_hist
    
    def get_sample_slice(self, slice_):
        """Get the sample of a given slice throughout all the variables as a new object."""
        new_hist = AppendedHistory(varname=self.__varname, keys=self.__keys)
        for key in self.__history.keys():
            if key != self.__varname:
                for v in self.__history[key]:
                    new_hist.__history[key].append(v[slice_])
            else:
                new_hist.__history[self.__varname] = self.__history[self.__varname]
        return new_hist
        
    def plot(self, title, mode, marker=False, range_=None, complementary=True):
        """Plots a set of learning curves in one mode (loss OR accuracy) for give histories.
    
        Params:
        -------
        title - string, title of the plot,
        mode - str, {'acc', 'val_acc', 'loss', 'val_loss'},
        complementary - bulean, whether to plot complementary curve (validation is "complementary" to training).
        """

        if range_ != None:
            x = np.arange(range_[0], range_[1])
            slice_ = slice(range_[0], range_[1])

        if marker == False:
            marker = None
        else:
            marker = 'o'

        with plt.style.context('ggplot'):
            fig, ax = plt.subplots(figsize=(9, 6))
            get_ylabel = lambda mode: 'Accuracy' if mode in ['acc', 'val_acc'] else 'Loss'
            ax.set_title(title)
            ax.set_ylabel(get_ylabel(mode))
            ax.set_xlabel('Epochs')
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

            linestyles = ['--', '-']
            alphas = [1, 0.5]
            
            colormap = plt.cm.gist_rainbow(np.linspace(0, 1, len(self.__history[self.__varname])))
            ax.legend(title=self.__varname, handles=get_legend(colormap, self.__history[self.__varname], self.__varname, alphas=alphas, linestyles=linestyles))

            if mode.startswith('val_'): ind = 1
            else: ind = 0
            color_iter = iter(colormap)
            for val in self.__history[mode]:
                color = next(color_iter)
                if range_ == None:
                    ax.plot(val, marker=marker, linestyle=linestyles[ind], color=color, alpha=alphas[ind])
                else:
                    ax.plot(x, val[slice_], marker=marker, linestyle=linestyles[ind], color=color, alpha=alphas[ind])
            if complementary:
                if ind == 1:
                    mode_2 = mode[4:]
                else:
                    mode_2 = 'val_{}'.format(mode)
                color_iter = iter(colormap)
                for val in self.__history[mode_2]:
                    color = next(color_iter)
                    if range_ == None:
                        ax.plot(val, marker=marker, linestyle=linestyles[1-ind], color=color, alpha=alphas[1-ind])
                    else:
                        ax.plot(x, val[slice_], marker=marker, linestyle=linestyles[1-ind], color=color, alpha=alphas[1-ind])

        plt.show()
            
    def plot_acc_loss(self, title, prefix_1, prefix_2=None, slice_=slice(None)):
        """Plots both accuracy and validation traces for given histories.
    
        Params:
        -------
        title - string, title of the plot,
        prefix_1 - {'', 'val_'}, the first prefix,
        prefix_2 - {'', 'val_'}, the second prefix, optional.
        
        Prefixes allow plotting ONLY training or validation or BOTH.
        """

        with plt.style.context('ggplot'):
            fig = plt.figure(figsize=(9, 6))
            ax = []
            ax.append(fig.add_subplot(211))
            ax.append(fig.add_subplot(212))

            ax[0].set_title(title)
            ax[0].set_ylabel('Accuracy')
            ax[1].set_ylabel('Loss')
            ax[0].set_xlabel('Epochs')
            ax[1].set_xlabel('Epochs')
            ax[1].set_yscale('log')

            colormap = cm.gist_rainbow(np.linspace(0, 1, len(self.__history[self.__varname])))[slice_]
            
            for i, mode in enumerate(['acc', 'loss']):
                color_iter = iter(colormap)
                linestyles = ['--', '-']
                alphas = [1, 0.3]
                if prefix_1 == 'val_': ind = 1
                else: ind = 0
                for val in self.__history[prefix_1 + mode][slice_]:
                    color = next(color_iter)
                    ax[i].plot(val, linestyle=linestyles[ind], color=color, alpha=alphas[ind])
                del linestyles[ind]
                del alphas[ind]
                if prefix_2:
                    color_iter = iter(colormap)
                    for val in self.__history[prefix_2 + mode][slice_]:
                        color = next(color_iter)
                        ax[i].plot(val, linestyles[0], color=color, alpha=alphas[0])
               
            for i in [0, 1]:
                box = ax[i].get_position()
                ax[i].set_position([box.x0, box.y0, box.width * 0.8, box.height])

            # Put a legend to the right of the current axis
            if prefix_2:
                legend = get_legend(colormap, self.__history[self.__varname][slice_])
            else:
                legend = get_legend(colormap, self.__history[self.__varname][slice_], lines=False)
            ax[0].legend(handles=legend, loc='lower right')#, bbox_to_anchor=(1.135, 1.0))

        plt.show()
        
    def get_ist(self, i):
        """For each iteration, get the i-st element of the lerning curve."""
        ist = {}
        for key in self.__history.keys():
            if key != self.__varname:
                ist[key] = []
                for hist in self.__history[key]:
                    ist[key].append(hist[i])
            else:
                continue
        return ist
    
    def get_best(self):
        """For each iteration, find the minimal val_loss index
        and return all the metrics for this index."""
        best = {}
        for key in self.__history.keys():
            if key == 'val_loss':
                best[key] = []
                for hist in self.__history[key]:
                    best[key].append(min(hist))
                    best_ind = hist.index(min(hist))
            else:
                continue
        for key in self.__history.keys():
            if (key != 'val_loss') and (key != self.__varname):
                best[key] = []
                for hist in self.__history[key]:
                    best[key].append(hist[best_ind])
        return best