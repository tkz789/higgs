import uproot
import pandas as pd
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
    variables_of_interest = ["pTVH","pTV","ptL1","ptL2",'pTBBJ', 'pTBB', 'nJets', 'mLB','HT',"GSCMvh","etaL1","dRBB", "dPhiLL", "dPhiBBcor", "dEtaVBB"]
    additional_info = ["EventWeight", "Sample"]

    for channel in channels:
        logger.info('Opening' + file_name[channel])
        branches = uproot.open(file_location + file_name[channel] + ':Nominal')
        print(branches.show())
        logger.info('Openieng finished')

        logger.info("Converting to pandas dataframe")
        dataframes = branches.arrays(variables_of_interest + additional_info,library='pd')
        logger.info("Converting finished")
        print(channel + " data frame\n",dataframes.head())

        logger.info("Pickling to file")
        dataframes.to_pickle(dataframe_path + channel + "VOI.pkl")




if __name__ == "__main__":
    read_and_pickle()
