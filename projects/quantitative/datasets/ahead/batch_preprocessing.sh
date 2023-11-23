nice -n 10 python preprocessing.py \
    --data_path /data/projects/recon/data/public/ahead --output_path /data/projects/recon/data/public/ahead_preprocessing \
    --plane axial --slice_range 120 171 --TEs 3.0 11.5 20.0 28.5 --fully_sampled True --shift False \
    --fft_centered False --fft_normalization backward --spatial_dims -2 -1
