import pandas as pd
import numpy as np
import logging
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import VarianceThreshold

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_classif(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

def main():
    file_location = "/home/daw/Documents/Physics/Particle phys/Higgs Project/"

    channels = ['lepton2']

    for channel in channels:
        logger.info('Opening')


        df = pd.read_pickle("lepton2whole_qq_gg")
        logger.info("Opening finished")
        print(df.head())
        selection = VarianceThreshold()
        df = selection.fit_transform(df)
        logger.info("Transformation finished")
        print(df.head())


        df['is_ggZH'] = np.logical_or(df["Sample"] == "ggZllH125", df["Sample"] == "ggZllH125cc")
        logger.info("Selection made")
        print(channel + " data frame\n",df.head())
        logger.info("Mutual_information")
        keys = df.columns.values
        keys.remove("Sample")
        keys.remove("is_ggZH")
        discrete_features = df[keys].dtypes == int
        mi_score = make_mi_scores(df[keys], df['is_ggZH'], discrete_features)
        logger.info("Pickling")
        mi_score.to_pickle("MIscore.pcl")
        print(mi_score)

if __name__ == "__main__":
    main()