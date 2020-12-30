import os
import jamendo_helpers


if __name__ == "__main__":

    
    jamendo_helpers.generate_audio_manifest('test.json')
    os.system('cat test.json')


