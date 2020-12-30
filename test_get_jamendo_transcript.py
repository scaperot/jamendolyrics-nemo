
from configparser import ConfigParser
import argparse,glob,os

import jamendo_helpers

if __name__ == "__main__":

    #read in jamendo filenames
    word_filenames = glob.glob(os.path.join('../lyrics','*.words.txt'))
    word_file = word_filenames[3] 
    print('transcript for:',word_file)
    transcript = jamendo_helpers.get_jamendo_transcript(word_file)
    print(transcript,'\n',len(transcript),'characters')
