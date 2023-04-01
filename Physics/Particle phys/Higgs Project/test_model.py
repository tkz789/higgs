from joblib import dump, load
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, roc_curve
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    logger.info("Opening model_boosted")
    model_boosted = load("model_with_abs_theta_boosted.joblib")
    model_tree = load("first_model.joblib")
    # channels = ["lepton0", "lepton2"]
    channel = "lepton2"
    sample_name = {"lepton0": "sample", "lepton2": "Sample"}
    keywords_qq = {"lepton0": "qqZvvH125", "lepton2": "qqZllH125"}
    keywords_gg = {"lepton0": "ggZvvH125", "lepton2": "ggZllH125"}
    file_name = {"lepton0": "lepton0whole_qq_gg.pkl_preprocessed.pkl",
                 "lepton2": "lepton2whole_qq_gg.pkl_preprocessed.pkl"}
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
    mass = df.mBB
    weights = df.EventWeight

    train_X, val_X, train_y, val_y, train_weights, val_weights, train_mass, val_mass = train_test_split(X, y, weights,
                                                                                                        mass,
                                                                                                        random_state=1)
    y_score = model_boosted.predict_proba(val_X)

    # ggzh = []
    # qqzh = []
    # threshold_values = np.linspace(0, 1, 100)
    # for i in threshold_values:
    #     threshold = np.ones(int(y_score.size/2)) * i
    #     ggzh.append(precision_score(val_y, np.greater(y_score[:,1], threshold), sample_weight=val_weights))
    #     qqzh.append(precision_score(np.logical_not(val_y), np.greater(y_score[:,0], threshold), sample_weight=val_weights))
    #
    # plt.plot(threshold_values, ggzh, label='ggZH')
    # plt.plot(threshold_values, qqzh, label='qqZH')
    # plt.legend()
    # plt.xlabel("Threshold")
    # plt.ylabel("Precision")
    # plt.show()
    # plt.close()
    fpr, tpr, thresh = roc_curve(val_y, y_score[:, 1], pos_label=True, sample_weight=val_weights)
    fpr_tree, tpr_tree, thresh2 = roc_curve(val_y, model_tree.predict_proba(val_X)[:, 1], sample_weight=val_weights)

    # roc curve for tpr = fpr
    random_probs = [False for i in range(len(val_y))]
    p_fpr, p_tpr, _ = roc_curve(val_y, random_probs, pos_label=True, )
    plt.plot(fpr, tpr, label="Boosted Decision Tree")
    plt.plot(fpr_tree, tpr_tree, label="Decision tree")
    plt.plot(p_fpr, p_tpr, label="random")
    plt.ylabel("Recall or True Positive Rate")
    plt.xlabel("False positive rate")
    plt.legend()
    plt.show()
    plt.close()

    #   Histograms of higgs mass
    print(y_score)
    mass_ggZH = val_mass.loc[np.greater(y_score[:, 1], 0.5)]
    print(mass_ggZH)
    weights_ggZH = val_weights.loc[np.greater(y_score[:, 1], 0.5)]
    mass_qqZH = val_mass.loc[np.less(y_score[:, 1], 0.5)]
    print(mass_qqZH)
    weights_qqZH = val_weights.loc[np.less(y_score[:, 1], 0.5)]

    plt.hist([mass_qqZH, mass_ggZH], weights=[weights_qqZH, weights_ggZH], label=["qqZH", "ggZH"], bins=100,
             range=[0, 300])
    plt.legend()
    plt.show()
    plt.close()
    plt.hist([mass_qqZH, mass_ggZH], weights=[weights_qqZH, weights_ggZH], label=["qqZH", "ggZH"], bins=100,
             density=True, range=[0, 300])
    plt.legend()
    plt.show()
    plt.close()

    ggzh = []
    qqzh = []
    for i in range(1, 12):
        threshold = np.ones(int(y_score.size / 2)) * i
        ggzh.append(precision_score(val_y, np.greater_equal(val_X.nJets, threshold), sample_weight=val_weights))
        qqzh.append(
            precision_score(np.logical_not(val_y), np.less(val_X.nJets, threshold), sample_weight=val_weights))

    plt.plot(np.linspace(1, 12, 11), ggzh, label='ggZH')
    plt.plot(np.linspace(1, 12, 11), qqzh, label='qqZH')
    plt.legend()
    plt.xlabel("Threshold")
    plt.ylabel("Precision")
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
