# MPDistil
Code for ICLR 2024 paper **[A Good Learner can Teach Better: Teacher-Student Collaborative Knowledge Distillation](https://openreview.net/forum?id=Ixi4j6LtdX)**

Knowledge distillation (KD) is a technique used to transfer knowledge from a larger ''teacher'' model into a smaller ''student'' model. Recent advancements in meta-learning-based knowledge distillation (MetaKD) emphasize that the fine-tuning of teacher models should be aware of the student's need to achieve better knowledge distillation. However, existing MetaKD methods often lack incentives for the teacher model to improve itself. In this study, we introduce MPDistil, a meta-policy distillation technique, that utilizes novel optimization strategies to foster both collaboration and competition during the fine-tuning of the teacher model in the meta-learning step. Additionally, we propose a curriculum learning framework for the student model in a competitive setup, in which the student model aims to outperform the teacher model by self-training on various tasks. Exhaustive experiments on SuperGLUE and GLUE benchmarks demonstrate the efficacy of MPDistil compared to 20 conventional KD and advanced MetaKD baselines, showing significant performance enhancements in the student model -- e.g., a distilled 6-layer BERT model outperforms a 12-layer BERT model on five out of six SuperGLUE tasks. Furthermore, MPDistil, while applied to a large language teacher model (DeBERTa-v2-xxlarge), significantly narrows the performance gap of its smaller student counterpart (DeBERTa-12) by just 4.6% on SuperGLUE. We further demonstrate how higher rewards and customized training curricula strengthen the student model and enhance generalizability.

## Methodology
<p align="center">
  <img width="720" alt="methodology" src="https://github.com/notmyname16/MPDistil/assets/88495622/e9444194-08fd-43ad-880b-94232302f449">
</p>

## Results

<p align="center">
<img width="640" alt="table 1" src="https://github.com/notmyname16/MPDistil/assets/88495622/defec7c5-e7f1-40dc-a967-14d37e4b3367">
</p>

<p align="center">
<img width="640" alt="table 4" src="https://github.com/notmyname16/MPDistil/assets/88495622/6ebc8132-07c5-4ec9-bb28-c7b14e258e0b">
</p>



## How to run

To run MPDistil with the collaborative teacher-student loss and binary reward formulation, run

```
python main_superglue.py \
 --task $task \
 --data_dir ./data/ \
 --reward_function 'binary'
```

To run with competitive loss, run

```
python main_superglue.py \
 --task $task \
 --data_dir ./data/ \
 --use_comp_loss \
 --reward_function 'binary'
```

Similarly, we can chose to run with 'real' reward function.

### Citation
If you find this repo useful, please cite our paper
```
@inproceedings{
 sengupta2024a,
 title={A Good Learner can Teach Better: Teacher-Student Collaborative Knowledge Distillation},
 author={Ayan Sengupta and Shantanu Dixit and Md Shad Akhtar and Tanmoy Chakraborty},
 booktitle={The Twelfth International Conference on Learning Representations},
 year={2024},
 url={https://openreview.net/forum?id=Ixi4j6LtdX}
}
```
