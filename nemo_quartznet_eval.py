#
# Taken from https://towardsdatascience.com/train-conversational-ai-in-3-lines-of-code-with-nemo-and-lightning-a6088988ae37
#
# Quartznet model architecture:
# https://github.com/NVIDIA/NeMo/blob/main/examples/asr/conf/quartznet_15x5.yaml
#
import re,time,argparse,json,sys
import os.path
import numpy as np

from ruamel.yaml import YAML

from omegaconf import OmegaConf
from omegaconf import DictConfig

import torch
import pytorch_lightning as pl
import nemo.collections.asr as nemo_asr
import ctc_segmentation as ctc 

from nemo.utils import logging

from configparser import ConfigParser

import jamendo_helpers
sys.path.append('../')
import Evaluate

try:
    from torch.cuda.amp import autocast
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def autocast(enabled=None):
        yield


def restore_asr(restore_path):
    quartznet_model = nemo_asr.models.EncDecCTCModel.restore_from(restore_path)
    return quartznet_model


def prediction_save_logprobs(asr_model,filename,prediction_dir='./predictions/'):
    

    asr_model.preprocessor._sample_rate = 22050
    logprobs_list = asr_model.transcribe([filename], logprobs=True)[0].cpu().numpy()

    print('Saving logprobs for song.')
    if not os.path.isdir(prediction_dir):
        print('prediction directory not found, trying to create it.')
        os.makedirs(prediction_dir)
        if not os.path.isabs(prediction_dir):
            #make string absolute path
            prediction_dir = os.path.abspath(prediction_dir)
    audiofile = strip_path(filename)
    pred_fname = prediction_dir+'/'+audiofile[:-4]+'_logprobs.npy'
    #fname = open(pred_fname,'w') #jamendolyrics convention
    np.save(pred_fname,logprobs_list)

def prediction_with_alignment(asr_model,filename,transcript,prediction_dir='./predictions/'):
    

    asr_model.preprocessor._sample_rate = 22050
    
    logprobs_list = asr_model.transcribe([filename], logprobs=True)
    alphabet  = [t for t in asr_model.cfg['labels']] + ['%'] # converting to list and adding blank character.

    # adapted example from here:
    # https://github.com/lumaku/ctc-segmentation
    config = ctc.CtcSegmentationParameters()
    config.frame_duration_ms = 20  #frame duration is the window of the predictions (i.e. logprobs prediction window) 
    config.blank = len(alphabet)-1 #index for character that is intended for 'blank' - in our case, we specify the last character in alphabet.


    ground_truth_mat, utt_begin_indices = ctc.prepare_text(config,transcript,alphabet)

    timings, char_probs, state_list     = ctc.ctc_segmentation(config,logprobs_list[0].cpu().numpy(),ground_truth_mat)
    
    # Obtain list of utterances with time intervals and confidence score
    segments                            = ctc.determine_utterance_segments(config, utt_begin_indices, char_probs, timings, transcript)

    print('Ground Truth Transcript:',transcript)
    print('CTC Segmentation Dense Sequnce:\n',''.join(state_list))

    #save onset per word.
    print('Saving timing prediction.')

    audiofile = strip_path(filename)
    pred_fname = prediction_dir+'/'+audiofile[:-4]+'_align.csv'
    fname = open(pred_fname,'w') #jamendolyrics convention
    for i in transcript.split():
       #
       # taking each word, and writing out the word timings from segments variable
       #
       # re.search performs regular expression operations.
       # .format inserts characters into {}.  
       # r'<string>' is considered a raw string.
       # char.start() gives you the start index of the starting character of the word (i) in transcript string
       # char.end() gives you the last index of the ending character** of the word (i) in transcript string
       # **the ending character is offset by one for the regex command, so a -1 is required to get the right 
       # index
       char = re.search(r'{}'.format(i),transcript)
       #       segments[index of character][start time of char=0]
       onset = segments[char.start()][0]
       #       segments[index of character][end time of char=1]
       term  = segments[char.end()-1][1]
       fname.write(str(onset)+','+str(term)+'\n')
    fname.close()


def read_manifest(filename):

    line_list = []
    with open(filename) as F:
        for line in F:
            val = json.loads(line)
            line_list.append(val)
    files = [t["audio_filepath"] for t in line_list]
    transcripts = [t["text"] for t in line_list]
    return files,transcripts

def strip_path(filename):
    return filename.split('/')[-1]

def prediction_one_song(model,audio_filename,lp_dir='tmp',lp_ext='_logprobs.py',word_dir='../lyrics',word_ext='.words.txt',prediction_dir='metadata',prediction_ext='_align.csv'):
    '''
    model  - nemo model object
    lp_dir - path with logprobabilities
    audio_filename - file name of audio song that is being proceesed

    '''
    basename = audio_filename[:-4] #crop extension (mp3 or wav)
    alphabet  = [t for t in model.cfg['labels']] + ['%'] # converting to list and adding blank character.

    # adapted example from here:
    # https://github.com/lumaku/ctc-segmentation
    config = ctc.CtcSegmentationParameters()
    config.frame_duration_ms = 20  #frame duration is the window of the predictions (i.e. logprobs prediction window) 
    config.blank = len(alphabet)-1 #index for character that is intended for 'blank' - in our case, we specify the last character in alphabet.
    logprobs_filenames      = glob.glob(os.path.join(lp_dir,basename+'*_logprobs.npy'))
    logprobs_list = []
    for f in logprobs_filenames:
        logprobs_list.append(np.load(f))
    
    logprobs = logprobs_list[0]
    for i in range(1,len(logprobs_list)):
        logprobs = np.concatenate((logprobs,logprobs_list[i]))

    #read in jamendo filenames
    word_file = '../lyrics/'+basename+'.words.txt'
    transcript = jamendo_helpers.get_jamendo_transcript(word_file)

    tstart = time.time()
    
    print('Prepare Text.',flush=True)
    ground_truth_mat, utt_begin_indices = ctc.prepare_text(config,transcript,alphabet)

    print('Segmentation.',flush=True)
    timings, char_probs, state_list     = ctc.ctc_segmentation(config,logprobs,ground_truth_mat)
    
    print('Get time intervals.',flush=True)
    # Obtain list of utterances with time intervals and confidence score
    segments                            = ctc.determine_utterance_segments(config, utt_begin_indices, char_probs, timings, transcript)
    tend = time.time()
    
    pred_fname = prediction_dir+'/'+basename+'_align.csv' #jamendolyrics convention
    fname = open(pred_fname,'w') 
    offset = 0  #offset is used to compensate for the re.search command which only finds the first 
                # match in the string.  so the transcript is iteratively cropped to ensure that the 
                # previous words in the transcript are not found again.
    for i in transcript.split():
       #
       # taking each word, and writing out the word timings from segments variable
       #
       # re.search performs regular expression operations.
       # .format inserts characters into {}.  
       # r'<string>' is considered a raw string.
       # char.start() gives you the start index of the starting character of the word (i) in transcript string
       # char.end() gives you the last index of the ending character** of the word (i) in transcript string
       # **the ending character is offset by one for the regex command, so a -1 is required to get the right 
       # index
       char = re.search(r'{}'.format(i),transcript[offset:])
       #       segments[index of character][start time of char=0]
       onset = segments[char.start()+offset][0]
       #       segments[index of character][end time of char=1]
       term  = segments[char.end()-1+offset][1]
       offset += char.end()
       fname.write(str(onset)+','+str(term)+'\n')
    fname.close()


if __name__ == '__main__':
    '''
    For a single song...
    1. create labeled word times 
        Input: song filename
        Output: .wordonset.txt
    2. chop it up to fit the model. 
        Input: song filename
        Output: .wav's
    3. predict logprobs for each segment in #2 
        Input: .wav's
        Output: .npy's
    4. concatenate the logprobs and run ctc_segmentation (_align.csv)
        Input: .npy's
        Output: _align.csv (one)
    6. run Jamendo Evaluate on results
        Input: _align.csv and .wordonset.txt
        Output: <printed alignment error>
    '''
    parser = argparse.ArgumentParser(description="Run Audio file(s) / Transcripts (from nemo audio manifest)  through a known model for lyric alignment predictions, and save to file.")
    parser.add_argument('-c','--config', required=False, default='jamendo_for_nemo.cfg', type=str,help='config file with model, audio, prediction setup information.')
    args = parser.parse_args()

    print('Using: ',args.config)

    config = ConfigParser(inline_comment_prefixes=["#"])
    config.read(args.config)

    audio_manifest_path = config.get('main','AUDIO_MANIFEST')
    print('Using: ',audio_manifest_path)

    
    exit_flag = False
    if not os.path.exists(audio_manifest_path):
        print(audio_manifest_path,'not found.  Exiting.')
        exit_flag = True
    
    model_filename      = config.get('main','MODEL')
    print('Using: ',model_filename)
    if not os.path.exists(model_filename):
        print(model_filename,'not found.  Exiting.')
        exit_flag = True
    
    prediction_path     = config.get('main','PREDICTION_PATH')
    print('Using: ',prediction_path)
    if not os.path.exists(prediction_path):
        print('prediction directory not found, trying to create it.')
        os.makedirs(prediction_dir)
        if not os.path.isabs(prediction_dir):
            #make string absolute path
            prediction_dir = os.path.abspath(prediction_dir)

    #make tmp for temporary files that are cropped to run through the model.
    tmp_dir = 'tmp'
    if not os.path.exists(tmp_dir):
        print('WARNING: Creating directory:',tmp_dir)
        os.makedirs(tmp_dir)

    if exit_flag: sys.exit()

    files,transcripts = read_manifest(audio_manifest_path)
    #load model 
    asr_model = restore_asr(model_filename)

    #FOR ONE FILE...make a for loop to do all songs...
    i = np.random.randint(len(files))
    song_fname = files[i]

    print('Cropping songs...')
    #225500 - 10.23 seconds is the time chosen for the size of model...
    #22050  - sample rate used for training the model
    _ , song_cnames_list = jamendo_helpers.crop_song(song_fname,tmp_dir,225500,22050)

    print('Testing',len(song_cnames_list),'files.')
    ptime = []
    for i in range(len(song_cnames_list)):
        print('Testing',strip_path(song_cnames_list[i]))
        prediction_save_logprobs(asr_model, song_cnames_list[i], tmp_dir)

    prediction_one_song(asr_model,audio_filename,lp_dir='tmp',lp_ext='_logprobs.py',word_dir='../lyrics',word_ext='.words.txt',prediction_dir='metadata',prediction_ext='_align.csv'):


    #print(np.mean(ptime),'to run prediction on 10s file.')

    #results = Evaluate.compute_metrics(config)
    #Evaluate.print_results(results)
