import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    variables_of_interest = ["pTVH", "pTV", "ptL1", "ptL2", 'pTBBJ', 'pTBB', 'nJets', 'HT', "GSCMvh", "etaL1",
                             "dRBB", "dPhiLL", "dPhiBB", "dEtaVBB"]
    logger.info("Preprocessing")

    channel = "lepton0"
    file_name = {"lepton0": "lepton0whole_qq_gg.pkl", "lepton2": "lepton2whole_qq_gg.pkl"}
    sample = {'lepton0': 'sample', 'lepton2': 'Sample'}
    sample_names = {"lepton0": ["qqZllH125", "ggZllH125", "ggZllH125cc", "qqZllH125cc"],
                    "lepton2": ["qqZllH125", "ggZllH125", "ggZllH125cc", "qqZllH125cc"]}

    df = pd.read_pickle(file_name[channel])

    df = df.loc[(df[sample[channel]] == sample_names[channel][0]) |
                (df[sample[channel]] == sample_names[channel][1]) |
                (df[sample[channel]] == sample_names[channel][2]) |
                (df[sample[channel]] == sample_names[channel][3])]
    df['is_ggZH'] = np.logical_or(df[sample[channel]] == "ggZllH125", df[sample[channel]] == "ggZllH125cc")
    df["absdPhiBB"] = np.abs(df["dPhiBB"])
    print(df)

    logger.info("Pickling")
    df.to_pickle(file_name[channel] + "_preprocessed.pkl")

if __name__ == "__main__":
    main()