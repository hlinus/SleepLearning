import tensorflow as tf
import numpy as np
import scipy.misc
import itertools
import tfplot
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sleeplearning.lib.loaders.baseloader import BaseLoader

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO  # Python 3.x


class Logger(object):

    def __init__(self, log_dir, _run):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)
        self._run = _run

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.Summary(
            value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)
        self._run.log_scalar(tag, value, step)

    def cm_summary(self, prediction, truth, tag, step, classes):
        """
        Parameters:
            correct_labels                  : These are your true classification categories.
            predict_labels                  : These are you predicted classification categories
            step                            : Training step (batch/epoch)
        """
        cm = confusion_matrix(truth, prediction, labels=range(len(classes)))
        num_classes = cm.shape[0]
        per_class_metrics = np.array(
            precision_recall_fscore_support(truth, prediction, beta=1.0,
                                            labels=range(
                                                len(classes)))).T.round(2)
        cm_norm = cm.astype('float') * 1 / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm, copy=True)

        fig = plt.figure(figsize=(3, 2), dpi=320, facecolor='w',
                         edgecolor='k')
        ax = fig.add_subplot(1, 1, 1)
        im = ax.imshow(
            np.concatenate((cm, np.zeros((len(classes), 4))), axis=1),
            cmap='Oranges')
        classes += ['PR', 'RE', 'F1', 'S']
        xtick_marks = np.arange(len(classes))
        ytick_marks = np.arange(len(classes) - 4)

        ax.set_xlabel('Predicted', fontsize=6, weight='bold')
        ax.set_xticks(xtick_marks)
        c = ax.set_xticklabels(classes, fontsize=4, ha='center')
        ax.xaxis.set_label_position('top')
        ax.xaxis.tick_top()
        ax.set_ylabel('True Label', fontsize=6, weight='bold')
        ax.set_yticks(ytick_marks)
        ax.set_yticklabels(classes[:-4], fontsize=4, va='center')
        ax.yaxis.set_label_position('left')
        ax.yaxis.tick_left()

        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(j, i, '{}\n({:.2f})'.format(cm[i, j], cm_norm[i, j]),
                    horizontalalignment="center", fontsize=2,
                    verticalalignment='center', color="black")
        for i, j in itertools.product(range(cm.shape[0]),
                                      range(cm.shape[1], cm.shape[1] + 4)):
            val = per_class_metrics[i, j - num_classes]
            ax.text(j, i, val if j != cm.shape[1] + 3 else int(val),
                    horizontalalignment="center", fontsize=2,
                    verticalalignment='center', color="black")
        fig.set_tight_layout(True)
        summary = tfplot.figure.to_summary(fig, tag=tag)
        plt.close()
        self.writer.add_summary(summary, step)

    def log_2D_features(self, train_ds):
        if True:#len(inputdim) == 2:
            # log 50 samples for each label to tensorboard
            for i in np.unique(train_ds.labels):
                for k, j in enumerate(np.where(train_ds.labels == i)[0][:50]):
                    img = train_ds[j][0].data.numpy()[0]
                    self.image_summary('feature/' + BaseLoader.sleep_stages_labels[i], img, k, cmap='jet')

    def image_summary(self, tag, img, step, vmin=None, vmax=None, cmap=None):
        """Log one image."""

        img_summaries = []
        # Write the image to a string
        try:
            s = StringIO()
        except:
            s = BytesIO()
        # normalize
        vmin = img.min() if vmin is None else vmin
        vmax = img.max() if vmax is None else vmax
        if vmin != vmax:
            img = (img - vmin) / (vmax - vmin)  # vmin..vmax
        else:
            # Avoid 0-division
            img = img * 0.
        if cmap is not None:
            cmapper = matplotlib.cm.get_cmap(cmap)
            img = cmapper(img, bytes=True)  # (nxmx4)
        scipy.misc.toimage(img).save(s, format="png")
        # Create an Image object
        img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                   height=img.shape[0],
                                   width=img.shape[1])
        # Create a Summary value
        img_summary = tf.Summary.Value(tag=tag, image=img_sum)
        img_summaries.append(
                tf.Summary.Value(tag=tag, image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)

    def images_summary(self, tag, images, step, vmin=None, vmax=None, cmap=None):
        """Log a list of images."""

        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            try:
                s = StringIO()
            except:
                s = BytesIO()

            # normalize
            vmin = img.min() if vmin is None else vmin
            vmax = img.max() if vmax is None else vmax
            if vmin != vmax:
                img = (img - vmin) / (vmax - vmin)  # vmin..vmax
            else:
                # Avoid 0-division
                img = img * 0.

            if cmap is not None:
                cmapper = matplotlib.cm.get_cmap(cmap)
                img = cmapper(img, bytes=True)  # (nxmx4)

            scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            img_summaries.append(
                tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()