for task in 'CB' 'WSC' 'RTE'
do
    TF_CPP_MIN_LOG_LEVEL=2 TF_ENABLE_ONEDNN_OPTS=0 CUDA_VISIBLE_DEVICES=0 python main_superglue.py --task $task --data_dir ./superglue_data/ --do_lower_case --wandb_logging --reward_function 'binary'
    TF_CPP_MIN_LOG_LEVEL=2 TF_ENABLE_ONEDNN_OPTS=0 CUDA_VISIBLE_DEVICES=0 python main_superglue.py --task $task --data_dir ./superglue_data/ --do_lower_case --wandb_logging --use_comp_loss --reward_function 'binary'
    TF_CPP_MIN_LOG_LEVEL=2 TF_ENABLE_ONEDNN_OPTS=0 CUDA_VISIBLE_DEVICES=0 python main_superglue.py --task $task --data_dir ./superglue_data/ --do_lower_case --wandb_logging --reward_function 'real'
    TF_CPP_MIN_LOG_LEVEL=2 TF_ENABLE_ONEDNN_OPTS=0 CUDA_VISIBLE_DEVICES=0 python main_superglue.py --task $task --data_dir ./superglue_data/ --do_lower_case --wandb_logging --use_comp_loss --reward_function 'real'
done
