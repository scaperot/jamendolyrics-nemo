import os,argparse
import jamendo_helpers
from configparser import ConfigParser


if __name__ == "__main__":

    #select a json filename and save to disk.
    parser = argparse.ArgumentParser(description='generate required json file for nemo to process and evaluate jamendo songs')
    parser.add_argument('-c','--config', required=True,type=str,default='jamendo_for_nemo.cfg',
            help='config file for setting up evaluation of nemo model with jamendo')

    args = parser.parse_args()
    
    config = ConfigParser(inline_comment_prefixes=["#"])
    config.read(args.config)

    filename = config.get('main','AUDIO_MANIFEST')
    print('Creating',filename)

    #if the file exists, do nothing...
    if not os.path.exists(filename):
        
        #check to see if directory exists...
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            print('WARNING: Creating directory:',directory)
            os.makedirs(directory)

        jamendo_helpers.generate_audio_manifest(filename)
        if os.path.exists(filename):
            print('Success.')

    else:
        print(filename,'already exists. Doing nothing.')


