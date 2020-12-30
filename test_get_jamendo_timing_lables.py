
from configparser import ConfigParser
import argparse,glob,os

import jamendo_helpers

if __name__ == "__main__":

    #read in jamendo filenames
    word_onset_filenames = glob.glob(os.path.join('../annotations','*.wordonset.txt'))
    word_onset_file = word_onset_filenames[3] 

    labels = jamendo_helpers.get_jamendo_timing_labels(word_onset_file)
    print(labels)
