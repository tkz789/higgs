import logging
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def separation(n, bins):
    value = 0
    dx = bins[1] - bins[0]

    for i, j in zip(n[0], n[1]):
        if i + j == 0:
            continue
        value += ((i - j) ** 2) / (i + j) * dx

    return value / 2


def main():
    # channels = ["lepton0", "lepton2"]
    channel = "lepton0"
    sample_name = {"lepton0": "sample", "lepton2": "Sample"}
    keywords_qq = {"lepton0": "qqZvvH125", "lepton2": "qqZllH125"}
    keywords_gg = {"lepton0": "ggZvvH125", "lepton2": "ggZllH125"}
    file_name = {"lepton0": "lepton0whole_qq_gg.pkl_preprocessed.pkl",
                 "lepton2": "lepton2whole_qq_gg.pkl_preprocessed.pkl"}
    logger.info('Opening' + file_name[channel])
    df = pd.read_pickle(file_name[channel])
    logger.info('Opening finished')

    separations = []
    for key in df.keys():
        n, bins, patches = plt.hist([df[key].loc[df.is_ggZH], df[key].loc[np.logical_not(df.is_ggZH)]],
                                    weights=[df.EventWeight.loc[df.is_ggZH],
                                             df.EventWeight.loc[np.logical_not(df.is_ggZH)]],
                                    bins=100, density=True, label=["ggZH", "qqZH"])
        separations.append([separation(n, bins), key])

        plt.legend()
        plt.xlabel(key)
        plt.show()
        plt.close()
        logger.info(key)
    print(separations)

    for i in np.array([pair[0] for pair in separations]).argsort()[::-1]:
        print(separations[i])


if __name__ == "__main__":
    main()
