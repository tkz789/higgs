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
    sample = {'lepton0': 'sample', 'lepton2': 'Sample'}
    sample_names = {"lepton0": ["qqZvvH125", "ggZvvH125", "ggZvvH125cc", "qqZvvH125cc"],
                    "lepton2": ["qqZllH125", "ggZllH125", "ggZllH125cc", "qqZllH125cc"]}
    channels = ['lepton0']
    for channel in channels:
        logger.info("Opening")
        branches = uproot.open(file_name[channel] + ":Nominal")
        logger.info("Finished")
        df = pd.DataFrame()

        # Conversion of "Sample" channel containing qqZH
        df[sample[channel]] = awkward.to_dataframe(branches.arrays([sample[channel]]))
        df = df.loc[(df[sample[channel]] == sample_names[channel][0]) |
                    (df[sample[channel]] == sample_names[channel][1]) |
                    (df[sample[channel]] == sample_names[channel][2]) |
                    (df[sample[channel]] == sample_names[channel][3])]

        # Conversion of all other samples for qqZH processes
        for key in branches.keys():
            if branches.typenames()[key] in ["int32_t", "float"]:
                logger.info("Converting" + key)
                selected_array = awkward.to_dataframe(branches.arrays([key, sample[channel]]))
                print(selected_array.head())
                selected_array = selected_array.loc[(selected_array[sample[channel]] == sample_names[channel][0]) |
                                                    (selected_array[sample[channel]] == sample_names[channel][1]) |
                                                    (selected_array[sample[channel]] == sample_names[channel][2]) |
                                                    (selected_array[sample[channel]] == sample_names[channel][3])]
                df[key] = selected_array[key]
        print(df.head())
        logger.info("Pickling")
        df.to_pickle(channel + "whole_qq_gg")




if __name__ == "__main__":
    main()