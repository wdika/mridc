## Coil Sensitivity maps

If you are working with data containing a fully-sampled center, you can estimate the coil sensitivity maps using the
estimate_csm.py script.

## Evaluation

Evaluate the performance of a method by comparing the reconstructions to the ground truth, using the
[reconstruction evaluation script](mridc/evaluate.py).

Example for evaluating zero-filled reconstruction and returning the mean +/- std performance,

```bash
python -m mridc.evaluate target_path/ predictions_path/ output_path/ --method zero-filled --acceleration 10  
--type mean_std
```

Example for evaluating zero-filled reconstruction and returning a csv with the scores per slice
(to use for further visualization),

```bash
python -m mridc.evaluate target_path/ predictions_path/ output_path/ --method zero-filled --acceleration 10  
--type all_slices
```

## Visualize results

Simple drop to png your reconstruction using the save2png.py script.