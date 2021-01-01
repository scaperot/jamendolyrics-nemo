import numpy as np
import os, argparse,glob,csv,librosa
import DALI as dali_code
from DALI import utilities


import sys
sys.path.append('../../dali-dataset-tools')
import dali_helpers


'''
Take a single song and break up into ~10s chunks 
NOTE 1: This is a modification to Stoller's 'End to end lyrics alignment for polyphonic music 
  using an audio-to-character recogition model.', in which this code does not use context windows 
  on each side of the 10s prediction window.

Data Metadata Formata
- 225501 samples @22050Hz (10.2268s)

For Training: 
- shift 112750 samples @22050Hz (5.11s)

TODO: For Prediction
- shift by the size of the total samples (i.e. no overlap).

'''



def append_timing(audio_filename,timing_list):
    '''
    save wordonset.txt file of timing information for a file in jamendolyrics file format.

    Input:
    audio_filename (string) - filename of the audio file
    timing_list (list <float>) - word onset list

    1. basename - removes .wav from filename
    2. writes to basename.wordonset.txt (jamendolyric format)
    '''
    if audio_filename[-4:] != '.wav':
        print('append_timing: error, do not support other file types.')
        return False

    base_filename = audio_filename[:-4]
    txt_filename  = base_filename + '.wordonset.txt'
    
    f = open(txt_filename,'w')
    for i in timing_list:
        txt = '%.2f\n' % (i)
        f.write(txt)
    f.close()
    
    return

def append_transcript_nemo(json_filename,audio_filename,duration,transcript):
    '''
    append_transcript: save in nemo manifest format
       json_filename:  filename for appending
       audio_filename: absolute path to audio file
       duration:      length of song at full_filename
       transcript:    lyrics corresponding to full_filename
    '''
    jsonfile = open(json_filename, 'a')
    line_format = "{}\"audio_filepath\": \"{}\", \"duration\": {}, \"text\": \"{}\"{}"
    jsonfile.write(line_format.format(
            "{", audio_filename, duration, transcript, "}\n"))
    jsonfile.close()
    return

def get_jamendo_files():

    #read in jamendo filenames
    audio_filenames      = glob.glob(os.path.join('../mp3','*.mp3'))
    word_onset_filenames = glob.glob(os.path.join('../annotations','*.wordonset.txt'))
    word_filenames       = glob.glob(os.path.join('../lyrics','*words.txt'))

    audio_filenames.sort()
    word_onset_filenames.sort()
    word_filenames.sort()

    return audio_filenames, word_onset_filenames, word_filenames

def get_jamendo_timing_labels(wordonset_fname):
    with open(wordonset_fname) as f:
        time_rows = list(csv.reader(f, delimiter="\t"))
    return np.array([float(row[0]) for row in time_rows])

def get_jamendo_transcript(words_fname):
    with open(words_fname) as f:
        word_rows = list(csv.reader(f, delimiter="\t"))
    word_list = np.array([row[0] for row in word_rows])
    return ' '.join(word_list)

def generate_audio_manifest(audio_manifest_fname):
    '''
    generate NEMO audio manifest based on jamendolyrics song selection.

    1. get all the mp3 files from mp3/
    2. find all transcripts
    3. write .json that is familiar to nemo to AUDIO_MANIFEST
    '''
    mp3_filenames, _, word_filenames = get_jamendo_files()
    assert(len(mp3_filenames)==20),mp3_filenames.shape[0]

    for i in range(len(mp3_filenames)):
        audio_filename   = mp3_filenames[i]
        transcript = get_jamendo_transcript(word_filenames[i])
        append_transcript_nemo(audio_manifest_fname,audio_filename,10.2268,transcript)


    return True

def get_transcript_for_window_full_song(basename,ref_times,ref_words,window_secs):
    '''
    given a window in time where a song is cropped, find the associated transcript 
    by going through labels word for word and adding them to the transcript.

    additionally, for each window_index, find the offset from the start of the song
    to adjust the timing values relative to the start of the cropped song.

    Input:
    dali_annot (DALI object) - created using entry.annotations['annot']
    window_secs (tuple) - (start of window in secs,end of window in secs)
    window_index (int)  - index associated with the number of crops or associated segments created for the entire song.
                          window relative to the start of the song.  used to create timing offset. 
    
    Return:
    transcript (string), word onset timing (list of floats)
    '''
    transcript = ''
    onset_timing = []
    for i in range(len(ref_words)-1):
        transcript += (ref_word[i] + ' ')
        onset_timing.append( ref_times[i] )
    
    return transcript, onset_timing

def get_transcript_for_window(basename,ref_times,ref_words,window_secs,window_index):
    '''
    given a window in time where a song is cropped, find the associated transcript 
    by going through labels word for word and adding them to the transcript.

    additionally, for each window_index, find the offset from the start of the song
    to adjust the timing values relative to the start of the cropped song.

    Input:
    dali_annot (DALI object) - created using entry.annotations['annot']
    window_secs (tuple) - (start of window in secs,end of window in secs)
    window_index (int)  - index associated with the number of crops or associated segments created for the entire song.
                          window relative to the start of the song.  used to create timing offset. 
    
    Return:
    transcript (string), word onset timing (list of floats)
    '''
    transcript = ''
    onset_timing = []
    offset = dali_helpers.get_window_offset(window_index)
    for i in range(len(ref_words)-1):
        #find first full onset word
        word = ref_words[i]
        word_time  = ref_times[i] 
        next_word_time = ref_times[i+1]
        # word starts after  the start of window 
        # word ends   before the end   of window 
        if word_time > window_secs[0] and next_word_time < window_secs[1]:
            transcript += (word + ' ')
            word_time_with_offset = word_time - offset
            onset_timing.append( word_time_with_offset )
    
    return transcript, onset_timing, offset

def get_cropped_transcripts(audio_filename, ref_times,ref_words, song_ndx,sample_rate):
#def get_cropped_transcripts(song_id, dali_annot,song_ndx,sample_rate):
    '''
    Input:
    dali_annot (DALI object) - 
    song_ndx (mx2 numpy array) - values are samples relative to beginning of song (i.e. 0 is first sample) 
            row - [start of window, termination of window]
            m windows that were created with crop_song

    Return:
    song_transcripts (list), song_timing (list) is a list of lists with word onset timing associated with each transcript
    '''

    basename = audio_filename.split('/')[-1][:-4]

    #find the words and times in an array for faster access...?  i'm not sure if its faster.
    mcrops = song_ndx.shape[0]
    song_transcripts = []
    song_timing = []
    for j in range(mcrops):
        start = song_ndx[j,0] / sample_rate
        term  = song_ndx[j,1] / sample_rate
        window_secs  = np.array([start,term])


        transcript, timing_list, _ =  get_transcript_for_window(basename,ref_times,ref_words,window_secs,j)
        song_transcripts.append(transcript)
        song_timing.append(timing_list)

        print('window:',window_secs, ', crop num:',j,', transcript:',song_transcripts[j])
    return song_transcripts, song_timing

def calc_window_for_song(total_length,win_samples):
    '''
    calculate the start / end index for training windows
    slide window over song every (win_samples / 2) samples.

    Input:
    total_length (int) - total samples in a song
    win_samples (int) -  window size

    Return:
    start_ndx (m,) numpy array, the start of each window relative to the total samples of the song
    end_ndx (m,) numpy array, the end of each window relative to the total samples of song
    '''
    n   = np.arange(total_length)  # counter from 0 to max samples of x
    div = np.floor(total_length / win_samples).astype(int)
    rem = total_length % win_samples
    ndx = np.reshape(n[:-rem],(div,win_samples))
    start_ndx = np.reshape(ndx[:,0],(ndx[:,0].shape[0],1))
    end_ndx   = np.reshape(ndx[:,-1],(ndx[:,-1].shape[0],1))

    return np.concatenate((start_ndx, end_ndx),axis=1)

def crop_song(audio_filename, audio_path, win_samples, sample_rate):
    '''
    crop_song - takes a DALI song and crops it m times into win_length samples.

    Inputs: 
       song_id    - DALI song id
       entry      - DALI data entry
       audio_path - absolute path where audio files are stored (read/write)
       win_samples - number of samples for each crop
    Return:
       song_ndx   - (m,start_sample,stop_sample) indices for the m crops
       filename_list - absolute path for filenames saved with save_samples_wav

    1. load song with librosa
    2. calculate indices (i.e. sample index starting at 0) for windows of chunks. 
       a. 'win_rate' is win_length/2
       b. do not keep parts of the song less than win_length
    3. crop according to indices and save to audio_path in the form
        audio/<song_id>_##.wav 
        where ## is the number of chunks in the song.
   '''
    xin, sr = librosa.load(audio_filename, sr=sample_rate)
    x = dali_helpers.normalize_data(xin)

    song_ndx = calc_window_for_song(x.shape[0],win_samples)

    l = song_ndx.shape[0]
    basename = audio_filename.split('/')[-1][:-4]

    filename_list = []
    for i in range(l):
        filename_list.append( dali_helpers.save_samples_wav(basename, audio_path, i, x, (song_ndx[i,0],song_ndx[i,1]), sr) )

    return song_ndx, filename_list

def preprocess_song(audio_filename, ref_times, ref_words, audio_path, nemo_manifest_filename, sample_rate):
#def preprocess_song(song_id, dali_path, audio_path, dali_info, nemo_manifest_filename, sample_rate):
    '''
    
    '''
    win_size = 10.2268
    win_samples = np.floor(win_size * sample_rate).astype(int)
    print('window samples: ',win_samples)


    # slice up song and save to audio_path, return indices of samples
    song_ndx,filename_list = crop_song(audio_filename, ref_times, ref_words, audio_path, win_samples, sample_rate)


    # slice up the transcript for each cropped version of the song
    transcript_list, timing_list = get_cropped_transcripts(audio_filename, ref_times,ref_words, song_ndx,sample_rate)

    # save all cropped files in nemo format
    for i in range(len(transcript_list)):
        append_transcript_nemo(nemo_manifest_filename,filename_list[i],win_size,transcript_list[i])
        append_timing(filename_list[i],timing_list[i])

    return True


if __name__ == '__main__':
    '''
    choose a random song, crop audio files, and massage transcripts into nemo toolkit format
    '''
    audio_filenames, word_onset_filenames, word_filenames = get_jamendo_files()

    sample_rate = 22050
    nemo_manifest_filename = 'jamendo_for_nemo/jamendo_for_nemo.json'
    audio_path = 'jamendo_for_nemo'
    if not os.path.exists(audio_path):
        print('creating',audio_path)
        os.makedirs(os.path.abspath('.') + '/'+audio_path)
 
    # loop through all songs and preprocess 
    for i in range(len(word_onset_filenames)):
        ref_times = []
        ref_words = []
        print('Picking:',word_onset_filenames[i][:-14])


        audio_file      = audio_filenames[i]
        word_onset_file = word_onset_filenames[i]
        word_file       = word_filenames[i]

        ref_times = get_jamendo_timing_labels(word_onset_file)
        ref_words = get_jamendo_transcript(word_file)
        assert(ref_times.shape == ref_words.shape),"%d,%d" % (ref_times.shape[0], ref_words.shape[0])

        #preprocess a song
        if not preprocess_song(audio_file, ref_times, ref_words, audio_path, nemo_manifest_filename, sample_rate):
            print('ERROR PREPROCESSING.')

