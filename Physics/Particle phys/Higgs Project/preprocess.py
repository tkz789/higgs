import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    variables_of_interest = ["pTVH", "pTV", "ptL1", "ptL2", 'pTBBJ', 'pTBB', 'nJets', 'HT', "GSCMvh", "etaL1",
                         "dRBB", "dPhiLL", "dPhiBB", "dEtaVBB"]
    logger.info("Preprocessing")

    df = pd.read_pickle("lepton2VOI.pkl")

    df = df.loc[(df["Sample"] == "qqZllH125")|(df["Sample"] == "ggZllH125")]
    df['is_qqZH'] = df["Sample"] == "qqZllH125"
    print(df)

    logger.info("Pickling")
    df.to_pickle("lepton2VOI_preprocessed.pkl")

if __name__ == "__main__":
    main()