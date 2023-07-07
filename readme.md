# PAC

#### Install dependencies

See `requirment.txt` file for more information about how to install the dependencies.



#### Get the data and statistics

Once grid2op is installed, you can get the data directly from the internet. This download will happen automatically the first time you create the environment.

```
import grid2op
env = grid2op.make("l2rpn_case14_sandbox")
env = grid2op.make("l2rpn_wcci_2020")
```

After getting the data, you should run the following command to calculate the statistics.

```
python parameter.py
```

All data and statistics should be placed in the `PAC/data_grid2op` directory.



#### Run

```
python train.py
```