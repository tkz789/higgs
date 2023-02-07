import uproot
import matplotlib.pyplot as plt
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# def create_histogram(branch):
#     ggZZ = ("qqZllH125", "")



def main():
    logger.info("Opening file")
    file_location = "/home/daw/Documents/Physics/Particle phys/Higgs Project/"

    file_name = {'lepton0': "0leptons.root",
                'lepton1': "1lepton.root",
                'lepton2': "2leptons.root"}

    # channels = ["lepton0", "lepton2"]
    channel = "lepton0"
    sample_name = {"lepton0": "sample", "lepton2": "Sample"}
    keywords_qq = {"lepton0": "qqZvvH125", "lepton2": "qqZllH125"}
    keywords_gg = {"lepton0": "ggZvvH125", "lepton2": "ggZllH125"}

    # dataframe_path = "/home/daw/Documents/Physics/Particle phys/Higgs Project/"


    # dataframes = {}
    logger.info('Opening' + file_name[channel])
    branches = uproot.open(file_location + file_name[channel] + ':Nominal')
    # print(branches.show())
    logger.info('Opening finished')
    # qqZZ_mask = branches['Sample'].array() == "qqZllH125"
    # plt.hist(branches['Nvtx'].array()[qqZZ_mask],
    #          bins=20, range=(0, 300))
    # plt.xlabel("Nvtx")
    # plt.show()

    for key in branches.keys():
        if branches.typenames()[key] in ("int32_t", "float"):
            logger.info("Creating histogram for " + key)

            # "Sample" and "qqZllH125" for lepton2, "sample" and "qqZvvH124" for lepton0
            qqZH_mask = branches[sample_name[channel]].array() == keywords_qq[channel]
            ggZH_mask = branches[sample_name[channel]].array() == keywords_gg[channel]
            plt.hist([branches[key].array()[qqZH_mask],
                      branches[key].array()[ggZH_mask]],
                     weights=[branches['EventWeight'].array()[qqZH_mask],
                              branches['EventWeight'].array()[ggZH_mask]],
                     bins = 100, density=True,
                     label=['qqZH','ggZH'])
            # plt.xlabel(key)
            # plt.savefig("histograms/qqZZ " + channel + '_' + key + '.png')
            # plt.close()

            plt.xlabel(key)
            plt.legend()
            # plt.show()
            plt.savefig("histograms/"+channel+"/ggZHvsqqZH/weighted" + channel + '_' + key + ' comp.png')

            plt.close()








if __name__ == "__main__":
    main()