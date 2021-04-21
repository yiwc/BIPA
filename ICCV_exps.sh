cuda=3
version=v3
e4=40
tk=alpoderl2
#CUDA_VISIBLE_DEVICES=$cuda python3 BS3_vec.py -pre ICCV_$version -t $tk -e $e4 -c 40 -p IML --test_env $tk
#CUDA_VISIBLE_DEVICES=$cuda python3 BS3_vec.py -pre ICCV_$version -t $tk -e $e4 -c 40 -p NM --test_env $tk
#CUDA_VISIBLE_DEVICES=$cuda python3 BS3_vec.py -pre ICCV_$version -t $tk -e $e4 -c 40 -p PML --test_env $tk
#CUDA_VISIBLE_DEVICES=$cuda python3 BS3_vec.py -pre ICCV_$version -t $tk -e $e4 -c 40 -p MML --test_env $tk
#CUDA_VISIBLE_DEVICES=$cuda python3 BS3_vec.py -pre ICCV_$version -t $tk -e $e4 -c 40 -p MonIML --test_env $tk

#tk=alpoderl2_fixcam
#CUDA_VISIBLE_DEVICES=$cuda python3 BS3_vec.py -pre ICCV_$version -t $tk -e $e4 -c 40 -p IML --test_env $tk
#CUDA_VISIBLE_DEVICES=$cuda python3 BS3_vec.py -pre ICCV_$version -t $tk -e $e4 -c 40 -p NM --test_env $tk
#CUDA_VISIBLE_DEVICES=$cuda python3 BS3_vec.py -pre ICCV_$version -t $tk -e $e4 -c 40 -p PML --test_env $tk
#CUDA_VISIBLE_DEVICES=$cuda python3 BS3_vec.py -pre ICCV_$version -t $tk -e $e4 -c 40 -p MML --test_env $tk
#CUDA_VISIBLE_DEVICES=$cuda python3 BS3_vec.py -pre ICCV_$version -t $tk -e $e4 -c 40 -p MonIML --test_env $tk




version=v3
e4=500
CUDA_VISIBLE_DEVICES=$cuda python3 BS3_vec.py -pre ICCV_$version -t $tk -e $e4 -c 40 -p IML --test_env $tk