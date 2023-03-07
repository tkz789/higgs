from joblib import dump, load
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    logger.info("Opening model")
    model = load("model_with_abs_theta_boosted.joblib")
    # channels = ["lepton0", "lepton2"]
    channel = "lepton2"
    sample_name = {"lepton0": "sample", "lepton2": "Sample"}
    keywords_qq = {"lepton0": "qqZvvH125", "lepton2": "qqZllH125"}
    keywords_gg = {"lepton0": "ggZvvH125", "lepton2": "ggZllH125"}
    file_name = {"lepton0": "lepton0whole_qq_gg.pkl_preprocessed.pkl",
                 "lepton2": "lepton2VOI_preprocessed.pkl"}
    logger.info('Opening' + file_name[channel])
    df = pd.read_pickle(file_name[channel])
    logger.info('Opening finished')

    variables_of_interest = []

    if channel == "lepton2":
        variables_of_interest = ["pTVH", "pTV", "ptL1", "ptL2", 'pTBBJ', 'pTBB', 'nJets', 'HT', "GSCMvh", "etaL1",
                                 "dRBB",
                                 "dPhiLL", "dPhiBB", "dEtaVBB", "absdPhiBB"]
    elif channel == "lepton0":
        variables_of_interest = ["dEtaBB", "dEtaVBB", "dPhiBB", "dPhiVBB", "dRBB", "etaB1", "etaB2", "etaJ3", "mBBJ",
                                 "MEff", "MET", "mJ3", "nForwardJet", "nJ", "pTB1", 'pTB2', "pTJ3", "pTV"]
    else:
        logger.error("Wrong name")

    y = df.is_ggZH
    X = df[variables_of_interest]
    weights = df.EventWeight

    train_X, val_X, train_y, val_y, train_weights, val_weights = train_test_split(X, y, weights, random_state=1)
    y_score = model.decision_function(val_X)

    ggzh = []
    qqzh = []
    threshold_values = np.linspace(0, 1, 100)
    for i in threshold_values:
        threshold = np.ones(y_score.size) * i
        ggzh.append(precision_score(val_y, np.greater(threshold, y_score), sample_weight=val_weights))
        qqzh.append(precision_score(np.logical_not(val_y), np.greater(y_score, threshold), sample_weight=val_weights))

    plt.plot(threshold_values, ggzh)
    plt.plot(threshold_values, qqzh)
    plt.show()


if __name__ == "__main__":
    main()
