import uproot
import matplotlib.pyplot as plt
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    logger.info("Opening file")
    file_location = "/home/daw/Documents/Physics/Particle phys/Higgs Project/"

    file_name = {'lepton0': "0leptons.root",
                'lepton1': "1lepton.root",
                'lepton2': "2leptons.root"}

    channels = ["lepton0","lepton2"]
    keywordsqq = {"lepton0": "qqZvvH125", "lepton2": "qqZllH125"}
    keywordsgg = {"lepton0": "ggZvvH125", "lepton2": "ggZllH125"}

    choice = input("Choose 0 or 2")
    if choice == "0":
        channel = channels[0]
    elif choice == "2":
        channel = channels[1]
    else:
        channel = -1

    # dataframe_path = "/home/daw/Documents/Physics/Particle phys/Higgs Project/"

    branches = {}
    logger.info('Opening' + file_name[channel])
    branches[channel] = uproot.open(file_location + file_name[channel] + ':Nominal')
    print(branches[channel].typenames())
    logger.info('Opening finished')

    key = input("Choose the key from the list above")

    if branches[channel].typenames()[key] in ("int32_t", "float"):
        logger.info("Creating histogram for " + key)

        # "Sample" and "qqZllH125" for lepton2, "sample" and "qqZvvH124" for lepton0
        qqZH_mask = branches[channel]['Sample'].array() == keywordsqq[channel]
        ggZH_mask = branches[channel]['Sample'].array() == keywordsgg[channel]
        plt.hist([branches[channel][key].array()[qqZH_mask],
                  branches[channel][key].array()[ggZH_mask]],
                 weights=[branches[channel]['EventWeight'].array()[qqZH_mask],
                          branches[channel]['EventWeight'].array()[ggZH_mask]],
                 bins = 100, density=True,
                 label=['qqZH','ggZH'])
        # plt.xlabel(key)
        # plt.savefig("histograms/qqZZ " + channel + '_' + key + '.png')
        # plt.close()

        plt.xlabel(key)
        plt.legend()
        plt.show()
        plt.close()
        while True:
            low_range = int(input("Choose lower limit of the new histogram"))
            upper_range =int(input("Choose upper limit of the new histogram"))
            plt.hist([branches[channel][key].array()[qqZH_mask],
                      branches[channel][key].array()[ggZH_mask]],
                     weights=[branches[channel]['EventWeight'].array()[qqZH_mask],
                              branches[channel]['EventWeight'].array()[ggZH_mask]],
                     bins=100, density=True,
                     label=['qqZH', 'ggZH'],
                     range=[low_range, upper_range])
            plt.xlabel(key)
            plt.legend()
            plt.show()
            plt.close()








if __name__ == "__main__":
    main()