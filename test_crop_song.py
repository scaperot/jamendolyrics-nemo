import jamendo_helpers,os

if __name__ == "__main__":
    tmp_dir = 'tmp'
    audio_files, _ , _ = jamendo_helpers.get_jamendo_files()
    
    #pick one file, save crops to tmp/

    #make tmp if doens't exist
    if not os.path.exists(tmp_dir):
        print('WARNING: Creating directory:',tmp_dir)
        os.makedirs(tmp_dir)

    _,file_list = jamendo_helpers.crop_song(audio_files[0],tmp_dir,225500,22050)




