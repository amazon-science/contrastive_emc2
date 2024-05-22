## EMC2: Efficient MCMC Negative Sampling for Contrastive Learning with Global Convergence

**Authors: Chung-Yiu Yau, Hoi-To Wai, Parameswaran Raman, Soumajyoti Sarkar, Mingyi Hong**

This repository contains the PyTorch code used to perform the experiments of the [paper](https://arxiv.org/pdf/2404.10575). 

Check system_setup.md if you find your environment not compatible with the code.

Use the following commands to reproduce the main experiment results.

## ResNet-18 on STL-10:
- SimCLR

`main_simclr.py --beta 14.28 --lr 1e-4 --disable_batchnorm --pos_batch_size 32 --compute_batch_size 256 --epoch 100 --num_workers 4 --eval_freq 5 --data_root /path/to/data/directory`

- Embedding Cache

`main_gumbel.py --beta 14.28 --lr 1e-4 --rho 0.01024 --neg_batch_size 1 --pos_batch_size 32 --compute_batch_size 512 --disable_batchnorm --disable_proj --sampler gumbel_max --transform_batch_size 1 --num_workers 2 --eval_freq 5 --data_root /path/to/data/directory`

- SogCLR

`main_sogclr.py --beta 14.28 --lr 1e-4 --disable_batchnorm --pos_batch_size 32 --compute_batch_size 256 --epoch 100 --num_workers 4 --eval_freq 5 --data_root /path/to/data/directory`

- EMC<sup>2</sup>

`main_mcmc.py --beta 14.28 --lr 1e-4 --disable_batchnorm --pos_batch_size 32 --compute_batch_size 256 --epoch 100 --sampler mcmc --mcmc_burn_in 0.5 --num_workers 4 --eval_freq 5 --data_root /path/to/data/directory`

## ResNet-50 on ImageNet-100:
- SimCLR

`main_simclr.py --beta 14.28 --lr 1.2 --disable_batchnorm --dataset imagenet100 --model_name resnet50 --dim 128 --mlp_dim 2048 --projector_hidden_layers 1 --weight_decay 1e-6 --optimizer lars --epoch 800 --eval_freq 10 --num_workers 4 --data_root /path/to/data/directory`

- SogCLR

`main_sogclr.py --beta 14.28 --lr 1.2 --disable_batchnorm --dataset imagenet100 --model_name resnet50 --dim 128 --mlp_dim 2048 --projector_hidden_layers 1 --weight_decay 1e-6 --optimizer lars --epoch 800 --eval_freq 10 --num_workers 4 --data_root /path/to/data/directory`

- EMC<sup>2</sup>

`main_mcmc.py --beta 14.28 --lr 1.2 --disable_batchnorm --dataset imagenet100 --model_name resnet50 --dim 128 --mlp_dim 2048 --projector_hidden_layers 1 --weight_decay 1e-6 --optimizer lars --sampler mcmc --mcmc_burn_in 0.5 --epoch 800 --eval_freq 10 --num_workers 4 --data_root /path/to/data/directory`

## ResNet-18 on STL-10 Subset with SGD:
Use preaugmentation.py to generate the pre-augmented STL-10 with 2 augmentations per image.

- SimCLR

`main_simclr.py --beta 5 --lr 1e-3 --disable_batchnorm --pos_batch_size 4 --compute_batch_size 8 --epoch 10 --num_workers 1 --eval_freq 25 --check_gradient_error --finite_aug --n_aug 2 --optimizer sgd --eval_iteration --data_root /path/to/data/directory`

- Embedding Cache

`main_gumbel.py --beta 5 --lr 1e-3 --rho 0.1 --neg_batch_size 1 --pos_batch_size 4 --compute_batch_size 8 --disable_batchnorm --disable_proj --sampler gumbel_max --transform_batch_size 1 --epoch 10 --num_workers 1 --eval_freq 25 --check_gradient_error --finite_aug --n_aug 2 --optimizer sgd --eval_iteration --data_root /path/to/data/directory`

- SogCLR

`main_sogclr.py --beta 5 --lr 1e-3 --disable_batchnorm --pos_batch_size 4 --compute_batch_size 8 --epoch 10 --num_workers 1 --eval_freq 25 --check_gradient_error --finite_aug --n_aug 2 --optimizer sgd --eval_iteration --data_root /path/to/data/directory`

- EMC<sup>2</sup>

`main_mcmc.py --beta 5 --lr 1e-3 --disable_batchnorm --pos_batch_size 4 --compute_batch_size 8 --epoch 10 --sampler mcmc --mcmc_burn_in 0.5 --num_workers 1 --eval_freq 25 --check_gradient_error --finite_aug --n_aug 2 --optimizer sgd --eval_iteration --data_root /path/to/data/directory`


## Citation

Please consider citing our paper if you use our code:
```text
@misc{yau2024emc2,
      title={EMC$^2$: Efficient MCMC Negative Sampling for Contrastive Learning with Global Convergence}, 
      author={Chung-Yiu Yau and Hoi-To Wai and Parameswaran Raman and Soumajyoti Sarkar and Mingyi Hong},
      year={2024},
      eprint={2404.10575},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.

