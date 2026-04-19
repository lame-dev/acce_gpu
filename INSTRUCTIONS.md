# Compile and run instructions
Compilation and run of the test cases are managed through the makefile present in the same directory

To compile the source code on DAS-5, load `cuda12.6/toolkit` first: 

```bash
module load cuda12.6/toolkit
```

Run the makefile help setting to see all available options:
```bash
make help
```

Example of compilation and execution:
```bash
make all
make test_seq_sm
make test_cuda_sm
```

## Scalability plot

Run both sequential and CUDA versions across multiple problem sizes and produce a speedup plot:

```bash
python3 run_scalability.py
python3 plot_scalability.py
```

Options:
- `--sizes 64 128 256 512 1024` to choose specific grid sizes
- `--runs 5` to set the number of repetitions per size
- `--skip-seq` or `--skip-cuda` to skip one variant
- `--no-prun` to run directly on the current node (when already on a GPU node)

The runner writes `scalability_results.csv`; the plotter reads it and saves `scalability_plot.png`.

## Execution time breakdown plot

Build with profiling instrumentation, run across test scenarios, and produce a stacked bar chart:

```bash
make clean && make profile
python3 run_profile.py
python3 plot_profile.py
```

`run_profile.py` options:
- `--runs 3` to set repetitions per scenario
- `--scenarios debug large_mountains` to run only specific scenarios
- `--no-prun` to run directly on a GPU node

`plot_profile.py` options:
- `-sec large_mountains` to render only a single scenario
- `--output my_plot.png` to change the output filename

The runner writes `profile_results.csv`; the plotter reads it and saves `profile_plot.png`.

## Automated benchmark

Running the automated benchmark:
```bash
make -f mk/benchmark.mk benchmark
```
> this creates a `results/` folder that will contain `{scenario}/{run_number}/` that not only contain the output of that run but also the correctness check and nvprof output (ran after the fact with the same input)

Generating Speedup Table and NVPROF table
```bash
python3 analyze_scenario.py result/
```
> if `--verbose` is passed, scenario by scenario tables will be displayed after
