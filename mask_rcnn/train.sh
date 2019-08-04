#!/bin/bash
if [ $# -lt 9 ]; then
    echo "Format: ./train.sh jsonfile initmodel numgpus nsteps modeldir outname projectname startstage(0,1,2,3) use_masks(Y/N) [loadconfig]"    
    exit
fi

if [ $9 = "Y" ]; then
    msk=""
elif [ $9 = "N" ]; then
    msk="--ignore_masks"
else
    echo "Invalid character for use_masks (use Y or N only)"
    exit
fi

if [ $# -gt 9 ]; then
    lc="--loadconfig ${10}"
else
    lc=""
fi

get_latest_chkpt()
{
    LATESTMODEL=$(ls logs/ | grep $2 | sort | tail -1)
    LF=$(ls logs/$LATESTMODEL/*.h5 | sort | tail -1)
    cp $LF $1
}

if [ $8 -lt 1 ]; then
    echo "STARTING WITH HEADS WITH AUGMENTATION"
    python coco.py train --json $1 --model $2 --num_gpus $3 --nsteps $4 --aug --stage 1 --project $7 $lc $msk
    get_latest_chkpt "$5/$6_heads_aug.h5" $7
else
    LF=$2
fi

if [ $8 -lt 2 ]; then
    echo "STARTING WITH HEADS WITHOUT AUGMENTATIONS"
    python coco.py train --json $1 --model $LF --num_gpus $3 --nsteps $4 --stage 1 --project $7 $lc $msk
    get_latest_chkpt "$5/$6_heads.h5" $7
else
    LF=$2
fi

if [ $8 -lt 3 ]; then
    echo "TRAINING FULL NETWORK WITH AUGMENTATIONS"
    python coco.py train --json $1 --model $LF --num_gpus $3 --nsteps $4 --aug --stage 3 --project $7 $lc $msk
    get_latest_chkpt "$5/$6_full_aug.h5" $7
else
    LF=$2
fi

if [ $8 -lt 4 ]; then
    echo "TRAINING FULL NETWORK WITHOUT AUGMENTATIONS"
    python coco.py train --json $1 --model $LF --num_gpus $3 --nsteps $4 --stage 3 --project $7 $lc $msk
    get_latest_chkpt "$5/$6_full.h5" $7
fi
