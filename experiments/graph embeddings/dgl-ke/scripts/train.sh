DATA_PATH=~/data
SAVE_PATH=~/data/results
DATASET=hc1208

FILE_NAME=train.txt
FORMAT=raw_udd_htr

LOG_INTERVAL=1000
BATCH_SIZE=1000
BATCH_SIZE_EVAL=16
NEG_SAMPLE_SIZE=200

dglke_train --model_name TransE --dataset $DATASET --data_path $DATA_PATH --format FORMAT --data_files [$FILE_NAME] \
--save_path $SAVE_PATH --batch_size $BATCH_SIZE --log_interval $LOG_INTERVAL --batch_size_eval $BATCH_SIZE_EVAL --neg_sample_size $NEG_SAMPLE_SIZE \
--gpu 0 --test --mix_cpu_gpu --async_update