import awkward
import uproot
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    file_location = "/home/daw/Documents/Physics/Particle phys/Higgs Project/"

    file_name = {'lepton0': "0leptons.root",
                'lepton1': "1lepton.root",
                'lepton2': "2leptons.root"}
    channels = ['lepton2']
    for channel in channels:
        logger.info("Opening")
        branches = uproot.open(file_name[channel] + ":Nominal")
        logger.info("Finished")
        df = pd.DataFrame()
        df["Sample"] = awkward.to_dataframe(branches.arrays(["Sample"]))
        df = df.loc[(df["Sample"] == "qqZllH125") |
                    (df["Sample"] == "ggZllH125") |
                    (df["Sample"] == "ggZllH125cc") |
                    (df["Sample"] == "qqZllH125cc")]
        for key in branches.keys():
            if branches.typenames()[key] in ["int32_t", "float"]:
                logger.info("Converting" + key)
                selected_array = awkward.to_dataframe(branches.arrays([key, "Sample"]))
                print(selected_array.head())
                selected_array = selected_array.loc[(selected_array["Sample"] == "qqZllH125") |
                                                    (selected_array["Sample"] == "ggZllH125") |
                                                    (selected_array["Sample"] == "ggZllH125cc") |
                                                    (selected_array["Sample"] == "qqZllH125cc")]
                df[key] = selected_array[key]
        print(df.head())
        logger.info("Pickling")
        df.to_pickle(channel + "whole_qq_gg")




if __name__ == "__main__":
    main()