for i in {0..13}; do \
    /mnt5/nimrod/nerf-pytorch/venv/bin/python3 selfdeblur_dip_defocus.py --data_path /home/nimrod/Projects/depth_from_defocus/data/mask_synt_data/downsample/small_dataset_2022_08_01 --gpu 0 --index $i; \
done