
# Start training on single machine
Use `single_machine_main()` in `main_unimodal.py`.

To run single machine script on fresh AWS instance:
1. copy data to data/stl10_binary
2. `screen -S train`
3. `source activate pytorch`
4. `pip install wandb`
5. `wandb login`, copy & paste the api key from wandb
6. make sure that at the bottom of main_mcmc.py, main_gumebl.py, main_simclr.py or main_sogclr.py, call the function `single_machine_main()`.
7. `python main_unimodal.py --beta 14.28 --lr 3e-4 --rho 0.05 --neg_batch_size 1 --disable_batchnorm`


# Setup mpirun for multiple machines
Use `run.sh` and `mpi_main()` in `main_unimodal.py`.

Instructions to setup mpi cluster [https://mpitutorial.com/tutorials/running-an-mpi-cluster-within-a-lan/]:
1. `sudo vim /etc/hosts` to add names for private ip, e.g., 172.31.35.243 worker1
2. `cat oscar-gpu.pem > ~/.ssh/id_dsa`
3. `chmod -R 700 ~/.ssh && chmod 600 ~/.ssh/*`
4. `ssh worker1`
5. repeat the above for all masters

Setup AWS node:
- Use Deep Learning AMI GPU PyTorch 2.0.1 (Amazon Linux 2) 20230627 (id: ami-027207f9b19741557) (version 20232822 also works, but every node should use the same image.)
- Use instance g4dn.metal
- Security group: default + SSH(Linux/Mac)

Setup /etc/hosts:
- Copy Private IPv4 addresses of each worker into /etc/hosts on master
- For example, append this line `172.31.35.243 worker1`

Dataset:
1. Download datasets ( ~/data/imagenet and ~/data/stl10_binary) on one node by (imagenet link: train set https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar, val set https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar) and (stl10) running the training script with world_size=1 to let pytorch download.
   1. use the following bash code to extract ILSVRC2012_img_train.tar
   ```
    mkdir train val
    mv ILSVRC2012_img_train.tar train
    mv ILSVRC2012_img_val.tar val
    tar -xvf ILSVRC2012_img_train.tar
    for fn in n*.tar
    do
        d=($(echo $fn | tr "." "\n"))
        mkdir $d
        tar -xvf $fn --directory $d
    done
    tar -xvf ILSVRC2012_img_val.tar
   ```
   2. The folder structure of imagenet should looks like this:
    ```
    ~/data
    └── imagenet
        ├── train
        │   ├── n01440764
        │   │   ├── ...
        │   │   └── ...JPEG
        │   ├── n01443537
        │   └── ...
        └── val
            ├── n01440764
            │   ├── ...
            │   └── ...JPEG
            ├── n01443537
            └── ...
    ```
    3. run `cd ~ && find data/imagenet/train -type f | wc -l` to check if number of samples is correct: should show 1281168
    4. run `cd ~ && find data/imagenet/val -type f | wc -l` to check if number of samples is correct: should show 50000


Usually the first time starting run.sh on a mpi cluster can take 1~2 mins


To run multi machine script on fresh AWS instances:
1. On master node:
   1. `source activate pytorch`
   2. `pip install wandb`
   3. `wandb login`, copy & paste the api key from wandb
   4. `sudo vim /etc/hosts` to add names for private ip, e.g., 172.31.35.243 worker1
   5. `cat oscar-gpu.pem > ~/.ssh/id_dsa` to put ssh key into the system
   6. `chmod -R 700 ~/.ssh && chmod 600 ~/.ssh/*`
   7. `ssh worker1`, `ssh worker2`, ..., `ssh workern`
2. On every worker:
   1. `mkdir data`
3. On master node:
   1. for every worker i:
      1. `scp -r data/imagenet worker${i}:~/data/imagenet` (if scp large directory like imagenet, run an empty python to avoid the receive-side machine going to sleep)
   2. make sure that at the bottom of main_unimodal.py, main_simclr.py or main_sogclr.py, call the function `mpi_main()`.
   3. edit run.sh at
      1. mpirun -np ____Put world size (total number of GPUs) here____
      1. -x MASTER_ADDR= ___Put master IP Address here___
   4. edit sync.sh to sync project code to all workers (except master itself)
