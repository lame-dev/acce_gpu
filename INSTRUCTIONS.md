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
