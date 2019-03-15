I removed all the examples from the original repository and then consolidated
the code into the `pcgan` folder, which can now be used as a Python package.
The `experiments` folder contains two subfolders:  `benchmark` and `gaspy`. The
`benchmark` folder contains some scripts and notebooks that I used to verify
the integrity of my Python environment and `pcgan` port---i.e., can I actually
run the original code?

The `gaspy` folder is named as such after the source of my data,
[GASpy](https://github.com/ulissigroup/GASpy). This folder contains
`parse_gaspy_data.ipynb` notebook I used to convert the GASpy data into a
format readable by `pcgan`. It also contains the `gaspoint.ipynb`, which is
what I used to experiment with different PC-GANS. The current and final
settings in that notebook are what I used to create the contents of the
`experiments/gaspy/entropy` folder, which contains the final result.
