import pandas as pd
import uproot
import logging
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, PrecisionRecallDisplay
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
    file_name = {"lepton0":"",
                 "lepton2":"lepton2VOI_preprocessed.pkl"}

    # dataframe_path = "/home/daw/Documents/Physics/Particle phys/Higgs Project/"

    logger.info('Opening' + file_name[channel])
    df = pd.read_pickle(file_name[channel])
    logger.info('Opening finished')

    variables_of_interest = ["pTVH","pTV","ptL1","ptL2",'pTBBJ', 'pTBB', 'nJets','HT',"GSCMvh","etaL1","dRBB", "dPhiLL", "dPhiBB", "dEtaVBB"]

    y = df.is_ggZH
    X = df[variables_of_interest]
    weights = df.EventWeight

    train_X, val_X, train_y, val_y, train_weights, val_weights = train_test_split(X, y,weights, random_state=1)

    for estimators in [1]:
        logger.info("Model for max_depth=" +str(estimators) )
        gg_vs_qq_model = HistGradientBoostingClassifier(random_state=1, early_stopping=True, learning_rate=0.1)
        gg_vs_qq_model.fit(train_X,train_y, train_weights)
        dump(gg_vs_qq_model,"first_model_boosted.joblib")

        predicted_y = gg_vs_qq_model.predict(val_X)
        print(predicted_y.size)
        print(predicted_y.sum())
        print("Current depth:",estimators)
        print(accuracy_score(val_y, predicted_y, sample_weight=val_weights))
        print(confusion_matrix(val_y,predicted_y,sample_weight=val_weights))
        print("precision",precision_score(val_y,predicted_y,sample_weight=val_weights))
        display = PrecisionRecallDisplay.from_estimator(gg_vs_qq_model, val_X, val_y, name="Boosted", sample_weight=val_weights, pos_label=True)
        display.plot()
        plt.show()

        print("False positive:",)
        print("False negative:",1-np.sum(np.logical_not(np.logical_or(predicted_y,val_y)))/(predicted_y.size-sum(predicted_y)))

if __name__ == "__main__":
    main()
