export CUDA_VISIBLE_DEVICES=1,2,3,4
python -m torch.distributed.run --nproc_per_node=4 train_paligemma.py
