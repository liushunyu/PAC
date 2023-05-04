# PAC

### Get the data
Once grid2op is installed, you can get the data directly from the internet. This download will happen automatically the first time you will create the environment.

```
import grid2op
env = grid2op.make("l2rpn_case14_sandbox")
env = grid2op.make("l2rpn_wcci_2020")
```

### Get the statistics
```
python parameter.py
```

All data and statistics should be placed in `PAC/data_grid2op`


### Run
```
python train.py
```