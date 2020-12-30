# jamendolyrics-nemo
code for running a NVIDIA pre-trained model through the jamendolyrics repository for evaluation

git clone https://github.com/f90/jamendolyrics
cd to jamendolyrics
clone this repostory git clone https://github.com/jamendolyrics-nemo
cd to jamendolyrics-nemo
modify the jamendo_for_nemo.cfg to point BASE_NAME to the path where your nemo model lives
run python jamendo_helpers.py to chop up audio into 10s chunks (assuming your model is good with that)
TODO: save logprobs, concatenate by song, run ctc_segmentation for each song, run Evaluate.py to see error results
