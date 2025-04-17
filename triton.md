# OpenAI Triton


# Contents
 - [Notes](#Notes)
 - [Autotune](#Autotune)
 - [Examples](#Examples)
 - [Errors](#Errors)
 - [Resources](#Resources)
 - 


### Notes
1. **All** operations in triton kernels are vectorized: Loading data, operating on data, storing data, and creating masks.
2. In Triton, we decompose the computation only in 1 level: Into blocks. There is no further decomposition into threads.
3. We don't need to and are not able to manage the shared memory. Triton does that automatically.
4. `os.environ['TRITON_INTERPRET'] = '1'`
5. Dtype triton and torch: [link](https://github.com/ROCm/triton/blob/9a32ed046673bbe8e67fcce688103dbe43f1f7aa/python/perf-kernels/streamk/utils/utils.py#L11-L31)

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
2. Using `tl.constexpr` where appropriate:
   ```
   my_kernel[grid](..., use_fp8=True)

   @triton.jit
   def my_kernel(...,
                 ...,
                 use_fp8,
                ):
       ## some code
       if use_fp8:
           ## some other code which won't be executed
           ## because you're confusing pointer with bool value!!

      ## end code
   
   ## Solution: make use_fp8 as tl.constexpr
   ```

### Errors
1. **Kernel output is correct in the Interpret mode but not in the regular mode. (where kernel correctness has been verified independently with unittest)**. This probably because the input tensors during integrating the kernel e2e are problematic. One big reason being Tensor not being contiguous! 

### Resources
- [A_Practitioners_Guide_to_Triton.ipynb](https://github.com/gpu-mode/lectures/blob/main/lecture_014/A_Practitioners_Guide_to_Triton.ipynb)
