import uproot
import pandas
import numpy as np
import logging
import awkward
from sklearn.feature_selection import mutual_info_classif

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    file_location = "/home/daw/Documents/Physics/Particle phys/Higgs Project/"

    file_name = {'lepton0': "0leptons.root",
                'lepton1': "1lepton.root",
                'lepton2': "2leptons.root"}
    channels = ['lepton2']

    for channel in channels:
        logger.info('Opening' + file_name[channel])
        branches = uproot.open(file_location + file_name[channel] + ':Nominal')
        logger.info('Opening finished')

        keys = branches.keys()
        logger.info("Converting to pandas dataframe")
        keys.remove("ZPV")
        df = awkward.to_dataframe(branches.arrays(keys))
        logger.info("Converting finished")

        df = df.loc[(df["Sample"] == "qqZllH125")|(df["Sample"] == "ggZllH125")|(df["Sample"] == "ggZllH125cc")|(df["Sample"] == "qqZllH125cc")]
        df['is_ggZH'] = np.logical_or(df["Sample"] == "ggZllH125", df["Sample"] == "ggZllH125cc")
        logger.info("Selection made")
        selected_keys = keys
        selected_keys.remove("Sample")
        print(channel + " data frame\n",df.head())
        logger.info("Mutual_information")
        print(mutual_info_classif(df[selected_keys],df.is_ggZH))

if __name__ == "__main__":
    main()