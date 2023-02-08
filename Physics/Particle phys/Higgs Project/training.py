import pandas as pd
import uproot
import logging
import numpy
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from joblib import dump, load

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# def probability_of_correct_classification(model, val_X, val_y, val_weights):
#     for true_value in val_y:
#         if model.predict

def test_model():
    model = load("first_model.joblib")

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

    y = df.is_qqZH
    X = df[variables_of_interest]
    weights = df.EventWeight

    train_X, val_X, train_y, val_y, train_weights, val_weights = train_test_split(X, y,weights, random_state=1)

    gg_vs_qq_model = DecisionTreeClassifier(random_state=1)
    gg_vs_qq_model.fit(train_X,train_y, train_weights)
    dump(gg_vs_qq_model,"first_model.joblib")

    validate_predictions = gg_vs_qq_model.predict(val_X)
    print(validate_predictions.len())
    print(validate_predictions.sum())
    print(accuracy_score(val_y, validate_predictions, sample_weight=val_weights))



if __name__ == "__main__":
    main()
