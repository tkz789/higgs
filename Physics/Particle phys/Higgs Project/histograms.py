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
    channels = ["lepton2"]
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

                qqZH_mask = branches[channel]['Sample'].array() == "qqZllH125"
                plt.hist(branches[channel][key].array()[qqZH_mask], bins = 100, density=True)
                # plt.xlabel(key)
                # plt.savefig("histograms/qqZZ " + channel + '_' + key + '.png')
                # plt.close()

                qqZH_mask = branches[channel]['Sample'].array() == "ggZllH125"
                plt.hist(branches[channel][key].array()[qqZH_mask], bins=100, density=True)
                plt.xlabel(key)
                # plt.savefig("histograms/ggZZ " + channel + '_' + key + '.png')
                plt.savefig("histograms/lepton2/ggZHvsqqZH/ " + channel + '_' + key + '.png')

                plt.close()








if __name__ == "__main__":
    main()