Examples for training and running reconstruction methods.

## Cascades of Independently Recurrent Inference Machines (CIRIM)

To enforce explicitly formulated Data Consistency, turn off the --nodc option. For the CIRIM is preferable to have on.

* ### **Train** a CIRIM with 8 cascades:
    ```shell
    python -m scripts.train_cirim --data_path data/ --exp_dir models/ --sense_path coil_sensitivity_maps/
    --mask_type gaussian2d --accelerations 4 10 --center_fractions .7 .7 --num_cascades 8 --time_steps 8
    --recurrent_layer IndRNN --conv_filters 64 64 2 --conv_kernels 5 3 3 --conv_dilations 1 2 1
    --conv_bias True True False --recurrent_filters 64 64 0 --recurrent_kernels 3 3 3 --recurrent_dilations 1 1 0
    --recurrent_bias True True False --depth 2 --conv_dim 2 --loss_fn "l1" --no_dc --keep_eta --output_type "SENSE"
    --fft_type orthogonal --lr 0.001 --batch_size 1 --num_epochs 100 --num_workers 4
    ```
* ### **Run** a CIRIM with 8 cascades:
    ```shell
    python -m scripts.run_cirim --data_path data/ --checkpoint models/ --out_dir recons/
    --sense_path coil_sensitivity_maps/
    --mask_type gaussian2d --accelerations 4 10 --center_fractions .7 .7 --num_cascades 8 --time_steps 8
    --recurrent_layer IndRNN --conv_filters 64 64 2 --conv_kernels 5 3 3 --conv_dilations 1 2 1
    --conv_bias True True False --recurrent_filters 64 64 0 --recurrent_kernels 3 3 3 --recurrent_dilations 1 1 0
    --recurrent_bias True True False --depth 2 --conv_dim 2 --loss_fn "l1" --no_dc --keep_eta --output_type "SENSE"
    --fft_type orthogonal --lr 0.001 --batch_size 1 --num_epochs 100 --num_workers 4
    ```

## Recurrent Inference Machines (RIM)

* ### **Train** a RIM:
    ```shell
    python -m scripts.train_cirim --data_path data/ --exp_dir models/ --sense_path coil_sensitivity_maps/
    --mask_type gaussian2d --accelerations 4 10 --center_fractions .7 .7 --num_cascades 1 --time_steps 8
    --recurrent_layer GRU --conv_filters 64 64 2 --conv_kernels 5 3 3 --conv_dilations 1 2 1
    --conv_bias True True False --recurrent_filters 64 64 0 --recurrent_kernels 3 3 3 --recurrent_dilations 1 1 0
    --recurrent_bias True True False --depth 2 --conv_dim 2 --loss_fn "l1" --no_dc --keep_eta --output_type "SENSE"
    --fft_type orthogonal --lr 0.001 --batch_size 1 --num_epochs 100 --num_workers 4
    ```
* ### **Run** a CIRIM with 8 cascades:
    ```shell
    python -m scripts.run_cirim --data_path data/ --checkpoint models/ --out_dir recons/
    --sense_path coil_sensitivity_maps/
    --mask_type gaussian2d --accelerations 4 10 --center_fractions .7 .7 --num_cascades 1 --time_steps 8
    --recurrent_layer GRU --conv_filters 64 64 2 --conv_kernels 5 3 3 --conv_dilations 1 2 1
    --conv_bias True True False --recurrent_filters 64 64 0 --recurrent_kernels 3 3 3 --recurrent_dilations 1 1 0
    --recurrent_bias True True False --depth 2 --conv_dim 2 --loss_fn "l1" --no_dc --keep_eta --output_type "SENSE"
    --fft_type orthogonal --lr 0.001 --batch_size 1 --num_epochs 100 --num_workers 4
    ```

## End-to-End Variational Network (E2EVN)

To omit explicitly formulated Data Consistency, turn on the --nodc option. For the E2EVN is preferable to have it off.

* ### **Train** an E2EVN with 12 cascades:
    ```shell
    python -m scripts.train_e2evn --data_path data/ --exp_dir models/ --sense_path coil_sensitivity_maps/
    --mask_type gaussian2d --accelerations 4 10 --center_fractions .7 .7 --num_cascades 12 --pools 2 --chans 14
    --unet_padding_size 11 --loss_fn "l1" --no_dc --keep_eta --output_type "SENSE" --fft_type orthogonal --lr 0.001
    --batch_size 1 --num_epochs 100 --num_workers 4
    ```
* ### **Run** an E2EVN with 12 cascades:
    ```shell
    python -m scripts.run_e2evn --data_path data/ --checkpoint models/ --out_dir recons/
    --sense_path coil_sensitivity_maps/
    --mask_type gaussian2d --accelerations 4 10 --center_fractions .7 .7 --num_cascades 12 --pools 2 --chans 14
    --unet_padding_size 11 --loss_fn "l1" --no_dc --keep_eta --output_type "SENSE" --fft_type orthogonal --lr 0.001
    --batch_size 1 --num_epochs 100 --num_workers 4
    ```

## UNet

* ### **Train** an UNet with 64 channels and without dropout:
    ```shell
    python -m scripts.train_unet --data_path data/ --exp_dir models/ --sense_path coil_sensitivity_maps/
    --mask_type gaussian2d --accelerations 4 10 --center_fractions .7 .7 --in_chans 2 --out_chans 2 --chans 64
    --num_pools 2 --unet_padding_size 11 --drop_prob 0.0 --loss_fn "l1" --no_dc --keep_eta --output_type "SENSE"
    --fft_type orthogonal --lr 0.001 --batch_size 1 --num_epochs 100 --num_workers 4
    ```
* ### **Run** an UNet with 64 channels and without dropout:
    ```shell
    python -m scripts.run_unet --data_path data/ --checkpoint models/  --out_dir recons/
    --sense_path coil_sensitivity_maps/
    --mask_type gaussian2d --accelerations 4 10 --center_fractions .7 .7 --in_chans 2 --out_chans 2 --chans 64
    --num_pools 2 --unet_padding_size 11 --drop_prob 0.0 --loss_fn "l1" --no_dc --keep_eta --output_type "SENSE"
    --fft_type orthogonal --lr 0.001 --batch_size 1 --num_epochs 100 --num_workers 4
    ```

## Compressed Sensing

* ### **Run** Parallel Imaging Compressed Sensing reconstruction:
    ```shell
    python -m scripts.run_pics --data_path data/ --out_dir recons/ --sense_path coil_sensitivity_maps/
    --mask_type gaussian2d --accelerations 4 10 --center_fractions .7 .7 --num_iters 60 --reg_wt .005
    --fft_type orthogonal --num_procs 4
    ```

## Zero-Filled

* ### **Run** Zero-Filled SENSE reconstruction:
    ```shell
    python -m scripts.run_zf --data_path data/ --out_dir recons/ --sense_path coil_sensitivity_maps/
    --mask_type gaussian2d --accelerations 4 10 --center_fractions .7 .7 --fft_type orthogonal --num_procs 4
    ```
