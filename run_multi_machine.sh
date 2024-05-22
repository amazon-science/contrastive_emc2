mpirun -np 64 \
--hostfile hostfile \
-x MASTER_ADDR=172.31.39.125 \
-x MASTER_PORT=29502 \
-x PATH \
-x OMP_NUM_THREADS=2 \
-x MKL_NUM_THREADS=2 \
-bind-to none -map-by slot \
-mca pml ob1 -mca btl ^openib \
--mca orte_base_help_aggregate 0 \
--mca btl_tcp_if_include eth0,lo \
/opt/conda/envs/pytorch/bin/python main_unimodal.py --beta 10 --lr 1.2 --rho 0.05 --cache_frac 1 --P 16384 --weight_decay 1e-6 --optimizer lars --pos_batch_size 256 --compute_batch_size 4096 --neg_batch_size 16 --projector_hidden_layers 0 --dim 2048 --mlp_dim 2048 --dataset imagenet100 --model_name resnet50 --grad_accum
