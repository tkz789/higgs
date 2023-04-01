from joblib import dump, load
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    logger.info("Opening file")
    model = load("first_model.joblib")
    # channels = ["lepton0", "lepton2"]
    channel = "lepton2"
    sample_name = {"lepton0": "sample", "lepton2": "Sample"}
    keywords_qq = {"lepton0": "qqZvvH125", "lepton2": "qqZllH125"}
    keywords_gg = {"lepton0": "ggZvvH125", "lepton2": "ggZllH125"}
    file_name = {"lepton0": "lepton0whole_qq_gg.pkl_preprocessed.pkl",
                 "lepton2": "lepton2whole_qq_gg.pkl_preprocessed.pkl"}

    # dataframe_path = "/home/daw/Documents/Physics/Particle phys/Higgs Project/"

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

    r = permutation_importance(model, val_X, val_y, sample_weight=val_weights, n_repeats=30, random_state=0)
    for i in r.importances_mean.argsort()[::-1]:
        print(variables_of_interest[i], i, r.importances_mean[i], r.importances_std[i])


if __name__ == "__main__":
    main()
