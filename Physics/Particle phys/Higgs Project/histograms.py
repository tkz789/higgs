import uproot
import matplotlib.pyplot as plt
import logging


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
    channels = ["lepton0"]
    # dataframe_path = "/home/daw/Documents/Physics/Particle phys/Higgs Project/"

    branches = {}
    # dataframes = {}
    for channel in channels:
        logger.info('Opening' + file_name[channel])
        branches[channel] = uproot.open(file_location + file_name[channel] + ':Nominal')
        # print(branches[channel].show())
        logger.info('Opening finished')
        # qqZZ_mask = branches[channel]['Sample'].array() == "qqZllH125"
        # plt.hist(branches[channel]['Nvtx'].array()[qqZZ_mask],
        #          bins=20, range=(0, 300))
        # plt.xlabel("Nvtx")
        # plt.show()

        for key in branches[channel].keys():
            if branches[channel].typenames()[key] in ("int32_t", "float"):
                logger.info("Creating histogram for " + key)

                # "Sample" and "qqZllH125" for lepton2, "sample" and "qqZvvH124" for lepton0
                qqZH_mask = branches[channel]['sample'].array() == "qqZvvH125"
                qqZH_mask2 = branches[channel]['sample'].array() == "ggZvvH125"
                plt.hist([branches[channel][key].array()[qqZH_mask],
                          branches[channel][key].array()[qqZH_mask2]],
                         bins = 100, density=True)
                # plt.xlabel(key)
                # plt.savefig("histograms/qqZZ " + channel + '_' + key + '.png')
                # plt.close()

                plt.xlabel(key)
                # plt.savefig("histograms/ggZZ " + channel + '_' + key + '.png')
                plt.savefig("histograms/lepton0/ggZHvsqqZH/" + channel + '_' + key + ' comp.png')

                plt.close()








if __name__ == "__main__":
    main()