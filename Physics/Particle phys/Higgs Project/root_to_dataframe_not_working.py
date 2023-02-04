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

    # channels = ("lepton0", "lepton2")
    channels = ('lepton0','')
    dataframe_path = "/home/daw/Documents/Physics/Particle phys/Higgs Project/"

    trees = {}
    dataframes = {}
    for channel in channels:
        logger.info('Opening' + file_name[channel])
        trees[channel] = uproot.open(file_location + file_name[channel] + ':Nominal')
        print(trees[channel].show())
        logger.info('Openieng finished')


        logger.info("Converting to pandas dataframe")
        dataframes[channel] = trees[channel].arrays(library='pd')
        logger.info("Converting finished")
        print(channel + " data frame\n",dataframes[channel].head())

        logger.info("Pickling to file")
        dataframes[channel].to_pickle(dataframe_path + channel + ".pkl")




if __name__ == "__main__":
    main()
