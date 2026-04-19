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

Running the automated benchmark:
```bash
make -f mk/benchmark.mk benchmark
```
> this creates a `results/` folder that will contain `{scenario}/{run_number}/` that not only contain the output of that run but also the correctness check and nvprof output (ran after the fact with the same input)

Generating Speedup Table and NVPROF table
```bash
pyhton3 analyze_scenario.py result/
```
> if `--verbose` is passed, scenario by scenario tables will be displayed after
