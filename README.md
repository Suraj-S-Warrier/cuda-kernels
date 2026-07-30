# CUDA Learning Repository

This repository is a hands-on CUDA learning project focused on building, understanding, and optimizing GPU kernels while working through the PMPP book (Programming Massively Parallel Processors).

The main goal is to move beyond just writing CUDA code and develop a deeper intuition for:

- GPU architecture and execution model
- thread/block organization
- memory access patterns
- performance bottlenecks and optimization strategies
- profiling and benchmarking with NVIDIA tools

## What this project is trying to achieve

This repo serves as a personal playground for experimenting with CUDA kernels and comparing different implementation strategies.

The workflow is intended to be:

1. Implement a kernel from scratch
2. Verify correctness
3. Profile it using Nsight Compute / Nsight Systems
4. Analyze performance characteristics
5. Compare results with more optimized approaches or standard GPU libraries

## Current examples included

The repository currently contains several CUDA programs covering a range of concepts:

- Vector addition
- Sum reduction
- Coalesced reduction
- Tiled matrix multiplication
- Optimized tiled matrix multiplication
- Granularity adjustment experiments

These examples are useful for exploring ideas such as:

- coalesced memory access
- shared memory usage
- tiling and blocking
- occupancy and launch configuration
- kernel-level performance tuning

It also contains screenshots of certain profiling tasks done to pinpoint the bottlenecks and improve performance.

## Learning focus

This project is meant to build practical experience in:

- writing CUDA kernels
- understanding warp execution and memory coalescing
- improving kernel throughput
- measuring performance with profiling tools
- comparing handcrafted kernels against library-based solutions

## Profiling and benchmarking

As the project grows, the plan is to:

- benchmark performance against standard libraries such as cuBLAS, CUB, or Thrust where appropriate
- document results and insights in a clear, structured way

## Build and run

CUDA source files in this repository can be compiled with NVCC. For example:

```bash
nvcc sum_reduction_coalesced.cu -o sum_reduction_coalesced.exe
./sum_reduction_coalesced.exe
```

You can also build using the provided VS Code task if available.

## Repository structure

- .cu files: CUDA kernel implementations
- results.txt: experiment or benchmark output

## Future direction

The next steps for this repo are to:

- add more CUDA kernels and algorithms
- keep improving optimization strategies
- profile more workloads systematically
- compare custom implementations with standard libraries
- document performance findings and takeaways

This repository is a work in progress and is intended to grow as a practical record of CUDA learning, experimentation, and performance exploration.
