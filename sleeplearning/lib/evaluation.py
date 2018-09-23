import itertools
import os
import shutil
import sys
import glob
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


def table_plot_(table, subjects, models):
    num_subjects = len(subjects)
    model_names = [m.name + '-' + r.name for m in models for r in m.configs]

    aggs = np.stack([np.mean(table, 0), np.std(table, 0)], axis=0)

    fig = plt.figure(figsize=(8.27, 11.69), dpi=320, facecolor='w',
                     edgecolor='k')
    gs = gridspec.GridSpec(num_subjects + 4, len(model_names))
    ax1 = fig.add_subplot(gs[:num_subjects, :])
    # plt.suptitle(PREFIX, fontsize=12)
    # ax1 =  plt.subplot(211)#fig.add_subplot(2, 1, 1)
    ax1.imshow(table[:num_subjects], cmap='Oranges', aspect="auto")

    for i, j in itertools.product(range(num_subjects),
                                  range(table.shape[1])):
        ax1.text(j, i, '{:.1f}'.format(table[i, j]),
                 horizontalalignment="center", fontsize=8,
                 verticalalignment='center', color="black")
    ytick_marks = np.arange(num_subjects)
    ax1.set_yticks(ytick_marks)
    ax1.set_yticklabels(subjects)
    ax1.set_xticklabels([])

    ax2 = fig.add_subplot(gs[num_subjects + 1:, :])
    ax2.imshow(aggs, cmap='Oranges', aspect="auto")
    # ax2.set_aspect('equal', 'box')
    # plt.imshow(table,cmap='Oranges')
    for i, j in itertools.product(range(aggs.shape[0]),
                                  range(aggs.shape[1])):
        ax2.text(j, i, '{:.2f}'.format(aggs[i, j]),
                 horizontalalignment="center", fontsize=8,
                 verticalalignment='center', color="black")
    xtick_marks = np.arange(len(model_names))
    ax2.set_xticks(xtick_marks)
    ax2.set_xticklabels(model_names, rotation=60)

    ytick_marks = np.arange(2)
    ax2.set_yticks(ytick_marks)
    ax2.set_yticklabels(['mean', 'std'])
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

    def boxplot(self, ymin=.4):
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
        ax.set_title("Subject-wise accuracy", fontsize=14, fontweight='bold')
        ax = sns.boxplot(x="config", y="accuracy", hue="model", data=df,
                         #palette="Set3",
                         order=[c.name for c in self.models[0].configs])
        ax.tick_params(labelsize=12)
        ax.set_xlabel("")
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
                  fancybox=True, shadow=True, ncol=5, fontsize=12)
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='gray', linestyle='dashed')
        ax.set_ylim(ymin=ymin, ymax=1)
        ax.set_ylabel('accuracy', fontsize=12)

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
                    kappa = cohen_kappa_score(truth, prediction)
                    rows.append(
                        [model.name, config.name, acc, f1m, kappa])
                    accs = np.append(accs, acc)
                model_mean.append(np.mean(accs))
                model_std.append(np.std(accs))

            means.append(model_mean)
            stds.append(model_std)

        df = pd.DataFrame(rows, columns=['model', 'config',
                                         'accuracy', 'f1m', 'kappa'])
        fig, ax = plt.subplots(figsize=(6, 4), dpi=120)

        res = df.groupby(['model', 'config'], as_index=False)[
            'accuracy', 'f1m', 'kappa'].mean()
        print(res)

        ax.set_title("Overall accuracy", fontsize=14, fontweight='bold')
        ax = sns.barplot(x="config", y="accuracy", hue="model", data=df,
                         #palette="Set3",
                         order=[c.name for c in self.models[0].configs])
        ax.tick_params(labelsize=12)
        if xlabel is not None:
            ax.set_xlabel(xlabel, fontsize=12)
        else:
            ax.set_xlabel("")
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
                    fancybox=True, shadow=True, ncol=5, fontsize=12)
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='gray', linestyle='dashed')
        ax.set_ylim(ymin=ymin, ymax=1)
        ax.set_ylabel('accuracy', fontsize=12)

    def hypnogram(self, index=0, config=None):
        f, axarr = plt.subplots(len(self.models), 1, squeeze=False,
                                sharex=True, sharey=True,
                                figsize=(12, 4 * len(self.models)), dpi=120)
        plt.yticks(range(5), ['W', 'N1', 'N2', 'N3', 'REM'])

        for i, model in enumerate(self.models):
            cfg = model.configs[0] if config is None else\
                next((item for item in model.configs if item.name == config),
                          None)
            if cfg is None:
                raise ValueError(f"config {config} not found")
            run = cfg.runs[0]
            path = run.subjects[index]
            subject = get_basename_(path)
            f.suptitle(f"{subject}")

            result = self.read_subject_file(path)
            axarr[i, 0].plot(range(len(result['y_pred'])), result['y_pred'],
                             label="prediction")
            axarr[i, 0].plot(range(len(result['y_true'])), result[
                'y_true'], alpha=0.3, label="truth")

            wrong = np.argwhere(np.not_equal(result['y_true'], result[
                'y_pred']))
            #axarr[i, 0].plot(wrong, result['y_pred'][wrong], 'r.')
            acc = result['acc']
            axarr[i, 0].set_title(f"{model.name} ({cfg.name}) - [{acc:.2f}%]")
            if 'attention' in result.keys():
                ax2 = axarr[i, 0].twinx()
                # same x-axis
                color = 'tab:green'
                ax2.set_ylabel('attention', color=color)
                attention = result['attention']
                ax2.plot(range(len(attention)), attention, color=color)
                ax2.tick_params(axis='y', labelcolor=color)
                ax2.set_ylim(0, 1)
        axarr[i, 0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
                           fancybox=True, shadow=True, ncol=5, fontsize=12)

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
        table_plot_(table, subjects, self.models)

    def att_table(self):
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
                    att_models.append(model)
        table = np.vstack(table).T
        subjects = [get_basename_(p) for p in run.subjects]
        table_plot_(table, subjects, att_models)

    def extract_experts(self):
        for i, model in enumerate(self.models):
            configs = []
            model_mean = []
            model_std = []
            for config in model.configs:
                experts = None
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
                    for i in range(result['y_experts'].shape[1]):
                        y_expert_prob = result['y_experts'][:, i, :]
                        y_expert_pred = np.argmax(y_expert_prob, 1)
                        expert = result['expert_channels'][i]
                        y_true = result['y_true']
                        a = result['a'][:, i]
                        wrong = np.argwhere(np.not_equal(y_true, y_expert_pred))
                        acc = 100*(1-wrong.shape[0]/len(y_expert_pred))
                        savepath = os.path.join(self.path, 'Expert-' +
                                             expert, os.path.basename(config.path), 'Expert-' +
                                             expert, subject)
                        savedict = {'y_true': y_true, 'y_pred': y_expert_pred,
                                    'acc': acc, 'attention': a}
                        np.savez(savepath, **savedict)

    def extract_voters(self):

        def get_acc(prediction, truth):
            wrong = np.argwhere(np.not_equal(truth, prediction))
            acc = 100 * (1 - (len(wrong) / len(truth)))
            return acc

        voting_models = ['SOFT-V', 'MAJ-V', 'U-BOUND']

        for new_model in voting_models:
            path = os.path.join(self.path, new_model)
            if os.path.exists(path) and os.path.isdir(path):
                shutil.rmtree(path)
        self.models = sorted(glob.glob(self.path + '/*'))

        for new_model in voting_models:
            os.makedirs(os.path.join(self.path, new_model))

        subjects = sorted(glob.glob(self.models[0] + '/*'))
        subjects = [get_basename_(f) for f in subjects]

        for subject in subjects:
            hard_votes = []
            soft_votes = []
            for i, model in enumerate(self.models):
                path = os.path.join(model, subject)
                result = self.read_subject_file(path)
                hard_votes.append(result['y_pred'])
                soft_votes.append(result['probs'])

            soft_votes = np.array(soft_votes)
            soft_vote = np.mean(soft_votes, axis=0)
            soft_vote = np.argmax(soft_vote, axis=1)
            savepath = os.path.join(self.path, 'SOFT-V', subject)
            savedict = {'y_true': result['y_true'], 'y_pred': soft_vote,
                        'acc': get_acc(soft_vote, result['y_true'])}
            np.savez(savepath, **savedict)


            hard_votes = np.array(hard_votes)
            from scipy.stats import mode
            maj_vote = mode(hard_votes, axis=0)[0][0]
            savepath = os.path.join(self.path, 'MAJ-V', subject)
            savedict = {'y_true': result['y_true'], 'y_pred': maj_vote,
                        'acc': get_acc(maj_vote, result['y_true'])}
            np.savez(savepath, **savedict)

            correct = np.equal(hard_votes, result['y_true'])
            ubound = np.any(correct, axis=0)
            ubound_vote = maj_vote
            ubound_vote[ubound] = result['y_true'][ubound]

            savepath = os.path.join(self.path, 'U-BOUND', subject)
            savedict = {'y_true': result['y_true'], 'y_pred': ubound_vote,
                        'acc': get_acc(ubound_vote, result['y_true'])}
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
        if 'y_experts'  in file.keys():
            result['y_experts'] = file['y_experts']
        if 'a' in file.keys():
            result['a'] = file['a']
        if 'attention' in file.keys():
            result['attention'] = file['attention']
        return result


if __name__ == '__main__':
    path = '/local/home/hlinus/Dev/SleepLearning/reports/results/Physionet18/Moe'
    e = Evaluation(path)
    e.extract_experts()
    #e.table()
    #e.extract_voters()
    e.att_table()
