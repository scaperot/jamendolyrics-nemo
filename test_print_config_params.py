
from configparser import ConfigParser
import argparse



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='print parameters of jamendo-nemo config file')
    parser.add_argument('-c','--config', required=True,type=str,default='jamendo_for_nemo.cfg',
            help='config file for setting up evaluation of nemo model with jamendo')

    args = parser.parse_args()
    
    config = ConfigParser(inline_comment_prefixes=["#"])
    config.read(args.config)

    audio_manifest_fname = config.get('main','AUDIO_MANIFEST')
    print('(used for nemo processing) Songs for evaluation and transcripts are located here:',audio_manifest_fname)
    metadata_path = config.get('main','METADATA_PATH')
    print('where transcripts, labels, predictions live: ',metadata_path)
    label_path = config.get('main','LABEL_PATH')
    print('where timing labels live (i.e. wordonset.txt): ',label_path)
    label_ext = config.get('main','LABEL_EXT')
    print('file extension to look for when reading in labels for evaluation (looks for all in directory): ',label_ext)
    prediction_path = config.get('main','PREDICTION_PATH')
    print('where outputs from model alignment predictions live: ',prediction_path)
    prediction_ext = config.get('main','PREDICTION_EXT')
    print('file extension to look for when reading in predictions for evaluation (looks for all in director):',prediction_ext)
    model_fname = config.get('main','MODEL')
    print('(used by nemo only) location of the pre-trained Nemo model:',model_fname)
    mp3_path = config.get('main','MP3_PATH')
    print('(used by jamendo only) MP3 songs for evaluation',mp3_path)
    delay = config.get('main','DELAY')
    print('(used by jamendo only) Jamendo delay factor to evaluate the error wrt to a static delay parameters.',delay)


