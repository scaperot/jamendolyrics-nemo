# jamendolyrics-nemo
code for running a NVIDIA pre-trained model through the jamendolyrics repository for evaluation

git clone https://github.com/f90/jamendolyrics </br>
cd to jamendolyrics</br>
clone this repostory git clone https://github.com/jamendolyrics-nemo</br>
cd to jamendolyrics-nemo</br>
modify the jamendo_for_nemo.cfg to point MODEL to the path where your nemo model lives</br>
run python jamendo_helpers.py to chop up audio into 10s chunks (assuming your model is good with that)</br>
create venv for nemo: </br>
python3 -m venv $HOME/tmp/nemo-venv/ </br></br>

enter vnenv: </br>
source $HOME/tmp/nemo-venv/bin/activate</br></br>

install nemo: </br>
pip install Cython, apt-get update && apt-get install -y libsndfile1 ffmpeg, pip install nemo_toolkit[all]==1.0.0b2</br></br>

install ctc_segmentation:</br>
pip install ctc_segmentation</br></br>


TODO: save logprobs, concatenate by song, run ctc_segmentation for each song, run Evaluate.py to see error results</br>
