from typing import List
from sleeplearning.lib.base import SleepLearning
import matplotlib.pyplot as plt
import numpy as np
import itertools

class Visualize(object):
    "Class to visualize psg data of sleeplearning objects"
    def __init__(self, sleeplearnings: List[SleepLearning]):
        self.data = sleeplearnings

    def class_distribution(self):
        sleep_stage_dist = []
        subject_labels = []
        total_labels = 0
        for sl in self.data:
            sleep_stage_dist.append(sl.hypnogram)
            total_labels += len(sl.hypnogram)
            subject_labels.append(sl.id_)
        plt.figure(figsize=(10, 5))

        plt.hist(sleep_stage_dist, label=subject_labels, bins=np.arange(8) - 0.5)
        plt.legend()
        plt.ylabel('count')
        plt.title(
            "Sleep Phases Distribution ({0} labels)".format(str(total_labels)))
        _ = plt.xticks(np.arange(7), list(SleepLearning.sleep_stages_labels.values()))

    def transition_distribution(self):
        num_sleep_phases = len(SleepLearning.sleep_stages_labels.keys())
        M = np.zeros((num_sleep_phases, num_sleep_phases))
        for sl in self.data:
            for i, j in zip(sl.hypnogram,sl.hypnogram[1:]):
                M[i, j] += 1
        cmap = plt.cm.Blues
        M /= (0.0001 + np.sum(M, axis=1)[:, np.newaxis])
        plt.figure()
        plt.imshow(M, interpolation='nearest', cmap=cmap)
        tick_marks = np.arange(num_sleep_phases)
        plt.yticks(tick_marks, SleepLearning.sleep_stages_labels.values())
        plt.xticks(tick_marks, SleepLearning.sleep_stages_labels.values())
        plt.title("Transition probabilities")
        for i, j in itertools.product(range(M.shape[0]), range(M.shape[1])):
            plt.text(j, i, '{:.2f}'.format(M[i, j]),
                     horizontalalignment="center",
                     color="black")

    def periodogram(self, psg_key: str, sleep_phases : List[int]):
        plt.figure()
        ax = plt.subplot('111')
        for i in sleep_phases:
            y = np.array([])
            for sl in self.data:
                f, periodograms = sl.get_periodograms(psg_key)
                ind = np.where(sl.hypnogram == i)[0]
                y_add = periodograms[ind]
                y = np.vstack([y, y_add]) if y.size else y_add
            y_mean = np.mean(y, axis=0)
            error = np.std(y, axis=0)

            ax.plot(f, y_mean, label=SleepLearning.sleep_stages_labels[i])
            ax.fill_between(f, y_mean - error / 5, y_mean + error / 5,
                             alpha=0.3)

        ax.set_xlim(xmin=0.5, xmax=24)
        ax.set_ylim(ymin=0.0099)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD [mV**2/Hz]")
        ax.set_yscale("log")
        ax.legend()

    def epoch(self, index : int, psg_key : str):
        for sl in self.data:
            if sl.hypnogram is not None:
                sleep_phase = sl.hypnogram[index]
                sleep_phase = SleepLearning.sleep_stages_labels[sleep_phase]
            else:
                sleep_phase = "UNK"
            f, axarr = plt.subplots(1, 3, sharex=False, figsize=(20, 5))
            f.suptitle('['+sl.id_+'] - epoch '+ str(index) + ' - sleep phase: ['+sleep_phase +']', fontsize=16)
            t = 20 * np.arange(5000, dtype=float) / 5000
            # t+=offset/30
            samples_per_epoch = sl.sampling_rate_*sl.epoch_size
            psg = sl.psgs_[psg_key][index*samples_per_epoch:(index+1)*samples_per_epoch]
            axarr[0].plot(t, psg)
            axarr[0].set_xlabel("time [s]")
            axarr[0].set_ylabel("mV")


            f, t, Sxx_list = sl.get_spectograms(psg_key)
            Sxx = Sxx_list[index]
            # print("Sxx: ", Sxx.shape)
            # print("f shape: ", f.shape)
            Sxx_cut = Sxx#Sxx[(f <= 38)]
            f_cut = f#f[(f <= 38)]
            axarr[1].pcolormesh(t, f_cut, Sxx_cut)
            axarr[1].invert_yaxis()
            axarr[1].set_xlabel("time [s]")
            axarr[1].set_ylabel(psg_key+ ' spectogram [Hz]')
            axarr[1].set_ylim(ymin=30)
            f, Pxx_list = sl.get_periodograms(psg_key)
            pxx = Pxx_list[index]
            axarr[2].plot(f, pxx)
            axarr[2].set_ylabel("mV**2")

    def qualitative_analysis(self, psg_key, from_epoch = 0, to_epoch = None):
        for sl in self.data:
            if to_epoch is None:
                # set to last epoch of sample if no value given
                to_epoch = len(sl.psgs_[psg_key])//sl.epoch_size//sl.sampling_rate_

            fig, axarr = plt.subplots(3, sharex=True, sharey=False, figsize=(20, 20))

            samples_per_epoch = sl.sampling_rate_ * sl.epoch_size
            axarr[0].set_xlim(xmin=from_epoch, xmax=to_epoch)
            axarr[0].plot(np.linspace(from_epoch, to_epoch, (to_epoch-from_epoch)*samples_per_epoch), sl.psgs_['FPZ'][from_epoch*samples_per_epoch:to_epoch*samples_per_epoch])
            axarr[0].set_ylim(ymin=-300, ymax=300)
            axarr[0].set_ylabel('mV')
            f, t, Sxx_list = sl.get_spectograms(psg_key)
            Sxx = Sxx_list[from_epoch:to_epoch]
            Sxx_cut = Sxx  # Sxx[(f >= 40)]
            Sxx_avg = np.mean(Sxx, axis=2)
            Sxx_avg = Sxx_avg.T

            axarr[1].pcolormesh(range(from_epoch, to_epoch), f, Sxx_avg, vmin=0, vmax=20,
                                 cmap='jet')
            axarr[1].set_ylim(ymax=18)
            axarr[1].invert_yaxis()
            axarr[1].set_ylabel("spectogram [Hz]")
            h = range(from_epoch, to_epoch+1)
            axarr[2].plot(h, sl.hypnogram[from_epoch:to_epoch+1])
            axarr[2].set_yticks(range(0,len(SleepLearning.sleep_stages_labels.keys())))
            axarr[2].set_yticklabels(SleepLearning.sleep_stages_labels.values())
            axarr[2].set_xlim(xmin=from_epoch, xmax=to_epoch)
            axarr[2].set_xlabel("epoch")