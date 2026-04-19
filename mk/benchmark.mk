BENCHMARK_RUNS ?= 5
BENCHMARK_SCENARIOS ?= test_files/debug.in test_files/small_mountains.in test_files/custom_clouds.in test_files/medium_lower_dam.in test_files/medium_higher_dam.in test_files/large_mountains.in

BENCHMARK_SEQ_BIN ?= ./flood_seq
BENCHMARK_CUDA_BIN ?= ./flood_cuda_ex

BENCHMARK_PRUN ?= prun
BENCHMARK_PRUN_TIME ?= 15:00
BENCHMARK_PRUN_NATIVE ?= -C gpunode

BENCHMARK_PROFILER ?= nvprof

BENCHMARK_RESULTS_DIR ?= results
BENCHMARK_RESULTS_DIR_SEQ ?= results/sequential
BENCHMARK_RESULTS_DIR_CUDA ?= results/cuda

benchmark:
	@set -eu; \
	for scenario in $(BENCHMARK_SCENARIOS); do \
		scenario_name=$$(basename "$$scenario" .in); \
		scenario_dir="$(BENCHMARK_RESULTS_DIR)/$$scenario_name"; \
		args="$$(cat "$$scenario")"; \
		echo "[benchmark] scenario=$$scenario_name input=$$scenario"; \
		run=1; \
		while [ $$run -le $(BENCHMARK_RUNS) ]; do \
			run_id=$$(printf '%02d' $$run); \
			run_dir="$$scenario_dir/$$run_id"; \
			seq_dir="$$run_dir/sequential"; \
			cuda_dir="$$run_dir/cuda"; \
			mkdir -p "$$seq_dir" "$$cuda_dir"; \
			$(BENCHMARK_PRUN) -t $(BENCHMARK_PRUN_TIME) -np 1 -native '$(BENCHMARK_PRUN_NATIVE)' $(BENCHMARK_SEQ_BIN) $$args > "$$seq_dir/result.out" 2> "$$seq_dir/error.txt"; \
			$(BENCHMARK_PRUN) -t $(BENCHMARK_PRUN_TIME) -np 1 -native '$(BENCHMARK_PRUN_NATIVE)' $(BENCHMARK_CUDA_BIN) $$args > "$$cuda_dir/result.out" 2> "$$cuda_dir/error.txt"; \
			python3 test_files/check_correctness.py "$$seq_dir/result.out" "$$cuda_dir/result.out" > "$$run_dir/check_correctness.txt" 2>&1; \
			$(BENCHMARK_PRUN) -t $(BENCHMARK_PRUN_TIME) -np 1 -native '$(BENCHMARK_PRUN_NATIVE)' $(BENCHMARK_PROFILER) $(BENCHMARK_CUDA_BIN) $$args > "$$run_dir/nvprof_out.txt" 2> "$$run_dir/nvprof_eval.txt"; \
			echo "[benchmark]        wrote $$seq_dir/result.out $$cuda_dir/result.out $$run_dir/check_correctness.txt $$run_dir/nvprof_eval.txt"; \
			run=$$((run + 1)); \
		done; \
	done

.PHONY: benchmark
