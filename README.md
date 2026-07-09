# Matrix Decomposition

## Compiling 

To compile on the cluster:
```bash
mkdir -p build && cd build
cmake -DCMAKE_CUDA_ARCHITECTURES=70 -DCMAKE_BUILD_TYPE=Release ..
make -j8
```
The test then run with the following job script:
```bash
#!/bin/bash
#SBATCH -p nvltv
#SBATCH -o job-%j.out
#SBATCH -N 1
#SBATCH -t 00:45:00
module load cuda
./build/tests/matdecomp_gpu_tests
```

## Plotting benchmarks

The GPU benchmarks emit Google Benchmark JSON. To turn a JSON run into figures:

```bash
# 1. Generate JSON (from the build directory or repo root)
build/benchmarks/matdecomp_xxx_benchmarks \
  --benchmark_format=json --benchmark_out=results.json

# 2. One-time: create the plotting venv
python -m venv .venv
./venv/bin/activate
pip install -r plot/requirements.txt

# 3. Render PNGs into plot/
python plot/plot_benchmarks.py results.json
```

Produces, in `plot/`: a log-runtime contour over (M, N) per algorithm
(`contour_<algo>_<dtype>.png`), a square-matrix comparison
(`square_comparison.png`), and skinny-matrix comparisons at N/M = 6.25%, 12.5%
and 25% (`skinny_comparison_*.png`). Algorithms are discovered from the JSON, so
adding more benchmarks needs no script change.
