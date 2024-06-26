import pandas as pd
import uproot
import logging
import numpy
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, confusion_matrix
from joblib import dump, load
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt


# def probability_of_correct_classification(model, val_X, val_y, val_weights):
#     for true_value in val_y:
#         if model.predict

def test_model():
    return load("first_model.joblib")

def main():
    logger.info("Opening file")

    # channels = ["lepton0", "lepton2"]
    channel = "lepton2"
    sample_name = {"lepton0": "sample", "lepton2": "Sample"}
    keywords_qq = {"lepton0": "qqZvvH125", "lepton2": "qqZllH125"}
    keywords_gg = {"lepton0": "ggZvvH125", "lepton2": "ggZllH125"}
    file_name = {"lepton0": "lepton0whole_qq_gg.pkl_preprocessed.pkl",
                 "lepton2": "lepton2VOI_preprocessed.pkl"}

    # dataframe_path = "/home/daw/Documents/Physics/Particle phys/Higgs Project/"

    logger.info('Opening' + file_name[channel])
    df = pd.read_pickle(file_name[channel])
    logger.info('Opening finished')

    variables_of_interest = ["pTVH", "pTV", "ptL1", "ptL2", 'pTBBJ', 'pTBB', 'nJets', 'HT', "GSCMvh", "etaL1", "dRBB",
                             "dPhiLL", "dPhiBB", "dEtaVBB",
                             "absdPhiBB"]

    y = df.is_ggZH
    X = df[variables_of_interest]
    weights = df.EventWeight

    train_X, val_X, train_y, val_y, train_weights, val_weights = train_test_split(X, y, weights, random_state=1)

    for depth in [2, 30]:
        logger.info("Model for max_depth=" + str(depth))
        gg_vs_qq_model = DecisionTreeClassifier(random_state=1, max_leaf_nodes=depth)
        gg_vs_qq_model.fit(train_X,train_y, train_weights)
        dump(gg_vs_qq_model,"first_model.joblib")

        predicted_y = gg_vs_qq_model.predict(val_X)
        print(predicted_y.size)
        print(predicted_y.sum())
        print("Current depth:",depth)
        print(accuracy_score(val_y, predicted_y, sample_weight=val_weights))
        print(export_text(gg_vs_qq_model, feature_names=variables_of_interest))
        print(confusion_matrix(val_y, predicted_y, sample_weight=val_weights))
        plot_tree(gg_vs_qq_model, feature_names=variables_of_interest)
        plt.show()



if __name__ == "__main__":
    main()
