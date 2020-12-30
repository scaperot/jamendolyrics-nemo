
from configparser import ConfigParser
import argparse,glob,os,time,re
import numpy as np
import nemo.collections.asr as nemo_asr
import ctc_segmentation as ctc 
import jamendo_helpers

def get_find_word_timing(segments,transcript):
    '''
    1. split the transcript into N words. 
    2. For word i, search the transcript for the first occurance
    3. Record the tstart, tend times
    4. Set the offset=
    '''
    offset = 0
    for i in transcript.split():
       # find the first instance of a word in the transcript, and each iteration, look at less 
       # and less of the transcript to find the right value.
       char = re.search(r'{}'.format(i),transcript[offset:])
       # segments[index of character][start time of char=0]
       #   char.start() is relative to transcript[offset:], so to get the right index, 
       #   you need to add the offset back in.
       onset = segments[char.start()+offset][0]
       #       segments[index of character][end time of char=1]
       term  = segments[char.end()-1+offset][1]
       offset = char.end()-1
    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-c','--config', required=False, default='jamendo_for_nemo.cfg', type=str,help='config file with model, audio, prediction setup information.')
    parser.add_argument('-s','--song', required=False, default='Avercage_-_Embers.mp3', type=str,help='file to perform logprob concatenations.')
    args = parser.parse_args()

    print('Using: ',args.config)

    config_arg = ConfigParser(inline_comment_prefixes=["#"])
    config_arg.read(args.config)
    audio_file = args.song
    basename   = audio_file[:-4]
    model_path = config_arg.get('main','MODEL')
    prediction_path = config_arg.get('main','PREDICTION_PATH')

    asr_model = nemo_asr.models.EncDecCTCModel.restore_from(model_path)
    asr_model.preprocessor._sample_rate = 22050

    alphabet  = [t for t in asr_model.cfg['labels']] + ['%'] # converting to list and adding blank character.

    # adapted example from here:
    # https://github.com/lumaku/ctc-segmentation
    config = ctc.CtcSegmentationParameters()
    config.frame_duration_ms = 20  #frame duration is the window of the predictions (i.e. logprobs prediction window) 
    config.blank = len(alphabet)-1 #index for character that is intended for 'blank' - in our case, we specify the last character in alphabet.
    tmp_dir = 'tmp'
    logprobs_filenames      = glob.glob(os.path.join(tmp_dir,basename+'*_logprobs.npy'))
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
    
    pred_fname = prediction_path+'/'+audio_file[:-4]+'_align.csv'
    fname = open(pred_fname,'w') #jamendolyrics convention
    offset = 0
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


