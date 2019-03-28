#!/bin/bash
if [ $# -lt 6 ]; then
    echo "Format: ./train.sh jsonfile initmodel numgpus nsteps modeldir outname"    
    exit
fi

get_latest_chkpt()
{
    LATESTMODEL=$(ls logs/ | grep bags | sort | tail -1)
    LF=$(ls logs/$LATESTMODEL/*.h5 | sort | tail -1)
    cp $LF $1
}

echo "Using relative paths in JSON? (y/n):"
read re

if [ "$re" == "y" ]; then
    relf="--relpath"
else
    relf=""
fi

python coco.py train --json $1 --model $2 --num_gpus $3 --nsteps $4 --aug --stage 1 $relf
get_latest_chkpt "$5/$6_heads_aug.h5" 

python coco.py train --json $1 --model $LF --num_gpus $3 --nsteps $4 --stage 1 $relf
get_latest_chkpt "$5/$6_heads.h5" 

python coco.py train --json $1 --model $LF --num_gpus $3 --nsteps $4 --aug --stage 3 $relf
get_latest_chkpt "$5/$6_full_aug.h5" 

python coco.py train --json $1 --model $LF --num_gpus $3 --nsteps $4 --stage 3 $relf
get_latest_chkpt "$5/$6_full.h5" 

