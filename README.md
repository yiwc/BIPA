# Binocular Point Alignment

## Install

`
conda env create -f bs3.yaml
`

two packages you may need

'
git clone https://github.com/yiwc/HRClib.git

git clone https://github.com/yiwc/rl_environment.git
'

HRClib is for connect to the KINOVA robot.

rl_environment is for training envs.

## Usage on Robot

### 1 run yolo on the server

on the server, run

`
python BS3_yolo_server.py -n inss_v6
`

notice, the -n, indicates which yolo weights you want to load. See all the weights you have in the directory:
`
yolov4/darknet/backup/
`


### 2 Meanwhile, run RL agent on the local

Run the insertion action demo using pretrained model

`
python main_control_v6_yolo.py
`

on line 379, task_select="insert", or "screw'

## Debug on the code

run above, with "ENABLE_FAKE_FLOW_TEST=False" (Line:383, main_control_v6_yolo.py)

It will load fake images to test the work flow.

## Train RL agent

`python BS3_vec.py`

## Enjoy RL agent in the Sim

`python BS3_enjoy.py`
