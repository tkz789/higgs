import uproot
import pandas as pd
import awkward
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def read_and_pickle(channels=None):
    """Possible inputs are ["lepton0","lepton2"]"""
    if channels is None:
        channels = ["lepton2"]


    file_location = "/home/daw/Documents/Physics/Particle phys/Higgs Project/"

    file_name = {'lepton0': "0leptons.root",
                'lepton1': "1lepton.root",
                'lepton2': "2leptons.root"}


    dataframe_path = "/home/daw/Documents/Physics/Particle phys/Higgs Project/"
    variables_of_interest = ["pTVH","pTV","ptL1","ptL2",'pTBBJ', 'pTBB', 'nJets','HT',"GSCMvh","etaL1","dRBB", "dPhiLL", "dPhiBB", "dEtaVBB"]
    additional_info = ["EventWeight", "Sample"]

    for channel in channels:
        logger.info('Opening' + file_name[channel])
        branches = uproot.open(file_location + file_name[channel] + ':Nominal')
        print(branches.show())
        logger.info('Opening finished')

        logger.info("Converting to pandas dataframe")
        dataframes = awkward.to_dataframe(branches.arrays(variables_of_interest + additional_info))
        logger.info("Converting finished")
        print(channel + " data frame\n",dataframes.head())

        logger.info("Pickling to file")
        dataframes.to_pickle(dataframe_path + channel + "VOI.pkl")




if __name__ == "__main__":
    read_and_pickle()
