# OpenAI Triton


# Contents
 - [Notes](#Notes)
 - [Autotune](#Autotune)
 - [Examples](#Examples)
 - [Resources](#Resources)
 - 


### Notes
1. **All** operations in triton kernels are vectorized: Loading data, operating on data, storing data, and creating masks.
2. In Triton, we decompose the computation only in 1 level: Into blocks. There is no further decomposition into threads.
3. We don't need to and are not able to manage the shared memory. Triton does that automatically.
4. `os.environ['TRITON_INTERPRET'] = '1'`

### Autotune
1. To print auto-tune logs: `os.environ["TRITON_PRINT_AUTOTUNING"] = "1"`
2. To only print the best config: `print(f"{my_triton_kernel.best_config}")`

### Examples
1. Slicing on tl vectors not allowed:
   ```
   my_vec = tl.load(x_ptr + tl.arange(0, 10))
   ele = my_vec[1]                     # <--- Not allowed
   ele = tl.load(x_ptr + 1)
   ```


### Resources
- [A_Practitioners_Guide_to_Triton.ipynb](https://github.com/gpu-mode/lectures/blob/main/lecture_014/A_Practitioners_Guide_to_Triton.ipynb)
