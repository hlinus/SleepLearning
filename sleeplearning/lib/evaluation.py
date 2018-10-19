import itertools
import os
import shutil
import sys
import glob
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, \
    accuracy_score, cohen_kappa_score
from sklearn.metrics import f1_score
from matplotlib import gridspec
import seaborn as sns
CLASSES = ['W', 'N1', 'N2', 'N3', 'REM']


def get_basename_(path):
    name = os.path.basename(os.path.normpath(path))
    # cut of number for ordering
    if len(name)>1 and name[1] == '_':
        name = name.split("_")[-1]
    return name


def cm_figure_(prediction, truth, classes, configuration_name):
    classes = classes.copy()
    cm = confusion_matrix(truth, prediction, labels=range(len(classes)))
    num_classes = cm.shape[0]
    per_class_metrics = np.array(
        precision_recall_fscore_support(truth, prediction, beta=1.0,
                                        labels=range(
                                            len(classes)))).T.round(2)
    cm_norm = cm.astype('float') * 1 / (cm.sum(axis=1)[:, np.newaxis]+1e-7)
    cm_norm = np.nan_to_num(cm_norm, copy=True)

    fig = plt.figure(figsize=(3, 2), dpi=320, facecolor='w',
                     edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(
        np.concatenate((cm_norm, np.zeros((len(classes), 4))), axis=1),
        cmap='Oranges')
    classes += ['PR', 'RE', 'F1', 'S']
    xtick_marks = np.arange(len(classes))
    ytick_marks = np.arange(len(classes) - 4)

    ax.set_xlabel('Predicted', fontsize=5, weight='bold')
    ax.set_xticks(xtick_marks)
    c = ax.set_xticklabels(classes, fontsize=5, ha='center')
    #ax.xaxis.set_label_position('top')
    #ax.xaxis.tick_top()
    ax.set_ylabel('True Label', fontsize=5, weight='bold')
    ax.set_yticks(ytick_marks)
    ax.set_yticklabels(classes[:-4], fontsize=5, va='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()
    ax.set_title(configuration_name, fontsize=5, horizontalalignment='center')

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, '{}\n({:.2f})'.format(cm[i, j], cm_norm[i, j]),
                horizontalalignment="center", fontsize=5,
                verticalalignment='center', color="black")
    for i, j in itertools.product(range(cm.shape[0]),
                                  range(cm.shape[1], cm.shape[1] + 4)):
        val = per_class_metrics[i, j - num_classes]
        ax.text(j, i, val if j != cm.shape[1] + 3 else int(val),
                horizontalalignment="center", fontsize=5,
                verticalalignment='center', color="black")

    return fig


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", xlabel=None, ylabel=None, **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Arguments:
        data       : A 2D numpy array of shape (N,M)
        row_labels : A list or array of length N with the labels
                     for the rows
        col_labels : A list or array of length M with the labels
                     for the columns
    Optional arguments:
        ax         : A matplotlib.axes.Axes instance to which the heatmap
                     is plotted. If not provided, use current axes or
                     create a new one.
        cbar_kw    : A dictionary with arguments to
                     :meth:`matplotlib.Figure.colorbar`.
        cbarlabel  : The label for the colorbar
    All other arguments are directly passed on to the imshow call.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    #cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    #cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Arguments:
        im         : The AxesImage to be labeled.
    Optional arguments:
        data       : Data used to annotate. If None, the image's data is used.
        valfmt     : The format of the annotations inside the heatmap.
                     This should either use the string format method, e.g.
                     "$ {x:.2f}", or be a :class:`matplotlib.ticker.Formatter`.
        textcolors : A list or array of two color specifications. The first is
                     used for values below a threshold, the second for those
                     above.
        threshold  : Value in data units according to which the colors from
                     textcolors are applied. If None (the default) uses the
                     middle of the colormap as separation.

    Further arguments are passed on to the created text labels.
    """
    import matplotlib
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center", fontsize=8)
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i, j] <=1:
                kw.update(color=textcolors[im.norm(data[i, j]) > threshold])
                text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            else:
                text = im.axes.text(j, i, "{:d}".format(int(data[i, j]), None),
                                    **kw)
            texts.append(text)

    return texts


def table_plot_(table, yticks, xticks, agg_table: bool = True):
    num_yticks = len(yticks)

    # m.configs]

    aggs = np.stack([np.mean(table, 0), np.std(table, 0)], axis=0)

    #fig = plt.figure(figsize=(8.27, 11.69), dpi=320, facecolor='w',
    #                 edgecolor='k')
    fig = plt.figure(figsize=(len(xticks), .5*len(yticks)), dpi=120,
                     facecolor='w',
                                      edgecolor='k')
    gs = gridspec.GridSpec(num_yticks + 4, len(xticks))
    ax1 = fig.add_subplot(gs[:num_yticks, :])
    # plt.suptitle(PREFIX, fontsize=12)
    # ax1 =  plt.subplot(211)#fig.add_subplot(2, 1, 1)
    ax1.imshow(table[:num_yticks], cmap='YlGn', aspect="auto")

    for i, j in itertools.product(range(num_yticks),
                                  range(table.shape[1])):
        ax1.text(j, i, '{:.3f}'.format(table[i, j]),
                 horizontalalignment="center", fontsize=10,
                 verticalalignment='center', color="black")
    ytick_marks = np.arange(num_yticks)
    ax1.set_yticks(ytick_marks)
    ax1.set_yticklabels(yticks)
    ax1.set_xticklabels([])

    if agg_table:
        ax2 = fig.add_subplot(gs[num_yticks + 1:, :])
        ax2.imshow(aggs, cmap='YlGn', aspect="auto")
        # ax2.set_aspect('equal', 'box')
        # plt.imshow(table,cmap='Oranges')
        for i, j in itertools.product(range(aggs.shape[0]),
                                      range(aggs.shape[1])):
            ax2.text(j, i, '{:.3f}'.format(aggs[i, j]),
                     horizontalalignment="center", fontsize=10,
                     verticalalignment='center', color="black")


        ytick_marks = np.arange(2)
        ax2.set_yticks(ytick_marks)
        ax2.set_yticklabels(['mean', 'std'])

        ax1 = ax2

    xtick_marks = np.arange(len(xticks))
    ax1.set_xticks(xtick_marks)
    ax1.set_xticklabels(xticks, rotation=60)


    return fig


class Model(object):
    def __init__(self, path):
        self.name = get_basename_(path)
        self.path = path
        print(f"model {self.name}")
        self.configs = [Configurations(p) for p in sorted(glob.glob(path + '/*'))]


class Runs(object):
    def __init__(self, path):
        self.name = get_basename_(path)
        print(f"runs: {self.name}")
        self.path = path
        self.subjects = sorted(glob.glob(path + '/*'))


class Configurations(object):
    def __init__(self, path):
        self.name = get_basename_(path)
        self.path = path
        print(f"config: {self.name}")
        self.runs = [Runs(p) for p in sorted(glob.glob(path + '/*'))]


class Evaluation(object):
    def __init__(self, path):
        self.path = path
        self.models = [Model(p) for p in sorted(glob.glob(path + '/*'))]

    def cm(self):
        for i, model in enumerate(self.models):
            runs = []
            for config in model.configs:
                runs.append(config.name)
                truth = []
                prediction = []
                run = config.runs[0]
                for path in run.subjects:
                    result = self.read_subject_file(path)
                    truth.append(result['y_true'])
                    prediction.append(result['y_pred'])
                truth = list(itertools.chain.from_iterable(truth))
                prediction = list(itertools.chain.from_iterable(prediction))
                cm = confusion_matrix(truth, prediction,
                                      labels=range(5))
                cm_norm = cm.astype('float') * 1 / (
                            cm.sum(axis=1)[:, np.newaxis] + 1e-7)
                cm_norm = np.nan_to_num(cm_norm, copy=True)

                fig, (ax1, ax2) = plt.subplots(2, 1,
                                               figsize=(2.5,5),
                                               dpi=120) #
                plt.subplots_adjust(hspace=.05)

                fig.suptitle(get_basename_(model.name)+"("+config.name+")",
                             fontsize=8, weight="bold",y=0.93)

                per_class_metrics = np.array(
                    precision_recall_fscore_support(truth, prediction, beta=1.0,
                                                    labels=range(
                                                        5))).round(
                    2)

                im = heatmap(per_class_metrics, ['PR', 'RE', 'F1', 'S'],
                             ('W', 'N1', 'N2', 'N3', 'REM'),
                             ax=ax1, cmap="YlGn", vmin=0,vmax=1e10,
                             aspect='auto')
                texts = annotate_heatmap(im, valfmt="{x:.2f} ")

                im = heatmap(cm_norm, ('W', 'N1', 'N2', 'N3', 'REM'),
                             ('W', 'N1', 'N2', 'N3', 'REM'),
                             ax=ax2, cmap="YlGn", aspect='auto',
                             xlabel="Predicted Label", ylabel="True Label")
                texts = annotate_heatmap(im, valfmt="{x:.2f} ")

                ax2.get_shared_x_axes().join(ax1, ax2)
                ax1.tick_params(axis="x", labelbottom=0)

                ax1.tick_params(
                    axis='x',  # changes apply to the x-axis
                    which='both',  # both major and minor ticks are affected
                    bottom=False,  # ticks along the bottom edge are off
                    top=False,  # ticks along the top edge are off
                    labelbottom=False)  # labels along the bottom edge are off

    def boxplot(self, xlabel=None, ymin=.4):
        models = []
        rows = []
        for i, model in enumerate(self.models):
            models.append(model.name)
            configs = []
            for config in model.configs:
                configs.append(config.name)
                if len(config.runs) == 0: continue
                run = config.runs[0]
                for path in run.subjects:
                    result = self.read_subject_file(path)
                    acc = result['acc']/100
                    rows.append([get_basename_(path), model.name, config.name,
                               acc])

        df = pd.DataFrame(rows, columns=['subject', 'model', 'config',
                                         'accuracy'])

        fig, ax = plt.subplots(figsize=(6,4), dpi=120)
        ax.set_title("Subject-wise accuracy", fontsize=14)
        ax = sns.boxplot(x="config", y="accuracy", hue="model", data=df,
                         #palette="Set3",
                         order=[c.name for c in self.models[0].configs])
        ax.tick_params(labelsize=10)
        if xlabel is not None:
            ax.set_xlabel(xlabel, fontsize=10)
        else:
            ax.set_xlabel("")
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                  fancybox=True, shadow=True, ncol=5, fontsize=10)
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='gray', linestyle='dashed')
        ax.set_ylim(ymin=ymin, ymax=1)
        ax.set_ylabel('accuracy', fontsize=10)

    def bar(self, xlabel=None, ymin=0.4):
        models = []
        means = []
        stds = []
        rows = []
        for i, model in enumerate(self.models):
            models.append(model.name)
            runs = []
            model_mean = []
            model_std = []
            for config in model.configs:
                runs.append(config.name)
                accs = np.array([])
                for j, run in enumerate(config.runs):
                    truth = []
                    prediction = []
                    for path in run.subjects:
                        result = self.read_subject_file(path)
                        truth.append(result['y_true'])
                        prediction.append(result['y_pred'])

                    truth = list(itertools.chain.from_iterable(truth))
                    prediction = list(itertools.chain.from_iterable(prediction))
                    acc = accuracy_score(truth, prediction)
                    f1m = f1_score(truth, prediction, average='macro')

                    _, _, f1c, _ = precision_recall_fscore_support(truth,
                                                               prediction,
                                                        beta=1.0,
                                                        labels=range(
                                                            5))

                    kappa = cohen_kappa_score(truth, prediction)
                    rows.append(
                        [model.name, config.name, acc, f1m, kappa] + list(f1c))
                    accs = np.append(accs, acc)
                model_mean.append(np.mean(accs))
                model_std.append(np.std(accs))

            means.append(model_mean)
            stds.append(model_std)
        cols = ['model', 'config',
                                         'accuracy', 'f1m', 'kappa', 'W',
                                         'N1', 'N2', 'N3', 'R']
        df = pd.DataFrame(rows, columns=cols)
        fig, ax = plt.subplots(figsize=(6, 4), dpi=120)

        res = df.groupby(['model', 'config'], as_index=False)[cols].mean()
        print(res.round(3).to_latex())

        ax.set_title("Overall accuracy")
        ax = sns.barplot(x="config", y="accuracy", hue="model", data=df,
                         #palette="Set3",
                         order=[c.name for c in self.models[0].configs])
        ax.tick_params(labelsize=10)
        if xlabel is not None:
            ax.set_xlabel(xlabel, fontsize=10)
        else:
            ax.set_xlabel("")
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                    fancybox=True, shadow=True, ncol=5, fontsize=10)
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='gray', linestyle='dashed')
        ax.set_ylim(ymin=ymin, ymax=1)
        ax.set_ylabel('accuracy', fontsize=10)

    def hypnogram(self, index=0, models=None, config=None, start=None,
                  end=None):
        models = self.models if models is None else [m for m in self.models
                                                     if m.name in models]
        if len(models) == 0: raise ValueError("no matching models found!")
        f, axarr = plt.subplots(len(models), 1, squeeze=False,
                                sharex=True, sharey=True,
                                figsize=(10, 3.5 * len(models)), dpi=320)
        plt.yticks(range(5), ['W', 'N1', 'N2', 'N3', 'REM'], fontsize=10)

        for i, model in enumerate(models):
            cfg = model.configs[0] if config is None else\
                next((item for item in model.configs if item.name == config),
                          None)
            if cfg is None:
                raise ValueError(f"config {config} not found")
            run = cfg.runs[0]
            path = run.subjects[index]
            subject = get_basename_(path)
            f.suptitle(f"{subject}", fontsize=12)

            result = self.read_subject_file(path)
            # only part of record
            if start is None and end is None:
                end = len(result['y_pred'])
                start = 0

            axarr[i, 0].set_xlim(xmin=start, xmax=end)
            axarr[i, 0].plot(range(len(result['y_pred'])), result['y_pred'],
                             label="prediction")
            axarr[i, 0].set_ylim(ymin=0)
            #axarr[i, 0].plot(range(len(result['y_true'])), result[
            #    'y_true'], alpha=0.9, label="truth", linestyle=':')

            wrong = np.argwhere(np.not_equal(result['y_true'], result[
                'y_pred']))
            axarr[i, 0].plot(wrong, result['y_true'][wrong], '.',
                             label="error")
            acc = result['acc']
            axarr[i, 0].set_title(f"{model.name} ({cfg.name}) - "
                                  f"[{acc:.2f}%]", fontsize=10)
            if 'attention' in result.keys():
                ax2 = axarr[i, 0].twinx()
                # same x-axis
                color = 'tab:green'
                ax2.set_ylabel('attention', color=color, fontsize=10)
                attention = result['attention']
                ax2.plot(range(len(attention)), attention, color=color)
                ax2.tick_params(axis='y', labelcolor=color)
                ax2.set_ylim(0, 1)
            if 'drop' in result.keys():
                dropped = np.argwhere(result['drop'])
                for d in dropped:
                    axarr[i, 0].axvspan(d-0.5, d+0.5, alpha=0.2, color='red')

        axarr[i, 0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                           fancybox=True, shadow=True, ncol=5, fontsize=12)
        axarr[i, 0].set_xlabel("epoch", fontsize=10)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    def table(self):
        table = []
        for i, model in enumerate(self.models):
            for config in model.configs:
                column = []
                run = config.runs[0]
                for path in run.subjects:
                    result = self.read_subject_file(path)
                    column.append(result['acc'])
                table.append(column)
        table = np.vstack(table).T
        subjects = [get_basename_(p) for p in run.subjects]
        xticks = [m.name + '-' + r.name for m in self.models for r in m.configs]
        table_plot_(table, subjects, xticks)

    def att_subject_table(self):
        att_models = []
        table = []
        for i, model in enumerate(self.models):
            for config in model.configs:
                column = []
                run = config.runs[0]
                for path in run.subjects:
                    result = self.read_subject_file(path)
                    if not 'attention' in result.keys():
                        continue
                    column.append(np.mean(result['attention']))
                if column != []:
                    table.append(column)
                    att_models.append(model.name + f"({config.name})")
        table = np.vstack(table).T
        subjects = [get_basename_(p) for p in run.subjects]
        #xticks = [m.name + '-' + r.name for m in self.models for r in
        # m.configs]
        table_plot_(table, subjects, att_models)

    def att_table(self):
        att_models = []
        table = []

        for i, model in enumerate(self.models):
            for config in model.configs:
                print(model.name)
                column = [[],[],[],[],[]]
                run = config.runs[0]
                for path in run.subjects:
                    result = self.read_subject_file(path)
                    if not 'attention' in result.keys():
                        continue
                    att_per_label = zip(result['y_pred'], result['attention'])
                    assert(not np.isnan(np.min(result['attention'])))
                    for label, a in att_per_label:
                        column[label].append(a)
                if column != [[],[],[],[],[]]:
                    column = [np.mean(np.array(av)) if av != [] else 0 for av
                              in column]

                    table.append(column)
                    att_models.append(model.name)
        table = np.vstack(table)
        table_plot_(table, att_models, ['W', 'N1', "N2", "N3", "REM"],
                    agg_table=False)

    def extract_experts(self):
        def get_acc(prediction, truth):
            wrong = np.argwhere(np.not_equal(truth, prediction))
            acc = 100 * (1 - (len(wrong) / len(truth)))
            return acc

        for i, model in enumerate(self.models):
            configs = []
            true_label_dict = None
            for config in model.configs:
                experts = None
                soft_votes_dict = defaultdict(lambda : [])
                hard_votes_dict = defaultdict(lambda : [])
                true_label_dict = {}
                configs.append(config.name)
                accs = np.array([])
                if len(config.runs) == 0: continue
                run = config.runs[0]

                # print("run: ", run.name)
                for path in run.subjects:
                    result = self.read_subject_file(path)
                    subject = get_basename_(path)
                    expert_base_path = os.path.join(self.path, os.path.basename(
                        config.path))
                    if experts is None:
                        experts = result['expert_channels']
                        for expert in experts:
                            os.makedirs(
                                os.path.join(self.path, 'Expert-' +
                                             expert, os.path.basename(config.path), 'Expert-' +
                                             expert))

                    voting_models = ['SOFT-V', 'MAJ-V']

                    for new_model in voting_models:
                        path = os.path.join(self.path, new_model, os.path.basename(
                            config.path), os.path.basename(
                            config.path))
                        if os.path.exists(path) and os.path.isdir(path):
                            shutil.rmtree(path)

                    for new_model in voting_models:
                        os.makedirs(os.path.join(self.path, new_model, os.path.basename(
                            config.path), os.path.basename(
                            config.path)))

                    for i in range(result['y_experts'].shape[1]):
                        y_expert_prob = result['y_experts'][:, i, :]
                        y_expert_pred = np.argmax(y_expert_prob, 1)
                        expert = result['expert_channels'][i]
                        y_true = result['y_true']
                        true_label_dict[subject] = y_true
                        a = result['a'][:, i]
                        drop = None
                        if 'drop_channels' in result.keys():
                            drop = result['drop_channels'][:, i]
                        hard_votes_dict[subject].append(y_expert_pred)
                        soft_votes_dict[subject].append(y_expert_prob)
                        wrong = np.argwhere(np.not_equal(y_true, y_expert_pred))
                        acc = 100*(1-wrong.shape[0]/len(y_expert_pred))
                        savepath = os.path.join(self.path, 'Expert-' +
                                             expert, os.path.basename(config.path), 'Expert-' +
                                             expert, subject)
                        savedict = {'y_true': y_true, 'y_pred': y_expert_pred,
                                    'acc': acc, 'attention': a}
                        if drop is not None:
                            savedict['drop'] = drop
                        np.savez(savepath, **savedict)
                for subject, predictions in soft_votes_dict.items():
                    soft_votes = np.array(predictions)
                    soft_vote = np.mean(soft_votes, axis=0)
                    soft_vote = np.argmax(soft_vote, axis=1)
                    y_true = true_label_dict[subject]
                    savepath = os.path.join(self.path, 'SOFT-V', os.path.basename(
                        config.path), os.path.basename(
                            config.path), subject)
                    savedict = {'y_true': y_true, 'y_pred': soft_vote,
                                'acc': get_acc(soft_vote, y_true)}
                    np.savez(savepath, **savedict)

                for subject, predictions in hard_votes_dict.items():
                    hard_votes = np.array(predictions)
                    from scipy.stats import mode
                    maj_vote = mode(hard_votes, axis=0)[0][0]
                    y_true = true_label_dict[subject]
                    savepath = os.path.join(self.path, 'MAJ-V', os.path.basename(
                        config.path), os.path.basename(
                            config.path), subject)
                    savedict = {'y_true': y_true, 'y_pred': maj_vote,
                                'acc': get_acc(maj_vote, y_true)}
                    np.savez(savepath, **savedict)


    def read_subject_file(self, path):
        file = np.load(path)
        truth = file['truth'] if 'truth' in file.keys() else file[
            'y_true']
        pred = file['pred'] if 'pred' in file.keys() else file['y_pred']
        t = file['acc']
        #print(t)
        #print(type(t))
        #print(AverageMeter(t))
        #print("avg: ", t.avg)
        acc = float(t)

        result = {'y_true': truth, 'y_pred': pred, 'acc': acc}
        if 'probs' in file.keys():
            result['probs'] = file['probs']
        if 'y_probs' in file.keys():
            result['probs'] = file['y_probs']
        if 'expert_channels' in file.keys():
            result['expert_channels'] = file['expert_channels']
        if 'y_experts' in file.keys():
            result['y_experts'] = file['y_experts']
        if 'a' in file.keys():
            result['a'] = file['a']
        if 'attention' in file.keys():
            result['attention'] = file['attention']
        if 'drop_channels' in file.keys():
            result['drop_channels'] = file['drop_channels']
        if 'drop' in file.keys():
            result['drop'] = file['drop']
        return result


if __name__ == '__main__':
    path = '/local/home/hlinus/Dev/SleepLearning/reports/results/Physionet18' \
           '/DroppingChannels'
    e = Evaluation(path)
    #e.bar()
    e.hypnogram()
    #e.att_table()
    #e.table()
    #e.extract_experts()
    #e.att_table()
