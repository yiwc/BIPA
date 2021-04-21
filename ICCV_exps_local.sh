cuda=0
version=v3
e4=2000
tk=alpoderl2

#CUDA_VISIBLE_DEVICES=$cuda python3 BS2_vec.py -pre ICCV_$version -t $tk -e $e4 -c 32

tk=alpoderl2_fixcam
CUDA_VISIBLE_DEVICES=$cuda python3 BS2_vec.py -pre ICCV_$version -t $tk -e $e4 -c 32


cuda=0
version=v3
e4=300
tk=alpoderl2_nodemo
CUDA_VISIBLE_DEVICES=$cuda python3 BS2_vec.py -pre ICCV_$version -t $tk -e $e4 -c 32
