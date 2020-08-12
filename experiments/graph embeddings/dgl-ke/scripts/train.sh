DATA_PATH=~/data
SAVE_PATH=~/data/results
LOG_INTERVAL=100

BATCH_SIZE=1000
BATCH_SIZE_EVAL=16
NEG_SAMPLE_SIZE=200

dglke_train --model_name TransE --data_path $DATA_PATH --save_path $SAVE_PATH --batch_size $BATCH_SIZE --log_interval $LOG_INTERVAL \
--batch_size_eval $BATCH_SIZE_EVAL --neg_sample_size $NEG_SAMPLE_SIZE --test --mix_cpu_gpu --async_update