# OpenAI Triton


# Contents
 - [Notes](#Notes)
 - [Resources](#Resources)


### Notes
1. **All** operations in triton kernels are vectorized: Loading data, operating on data, storing data, and creating masks.
2. In Triton, we decompose the computation only in 1 level: Into blocks. There is no further decomposition into threads.
3. We don't need to and are not able to manage the shared memory. Triton does that automatically.



### Resources
- [A_Practitioners_Guide_to_Triton.ipynb](https://github.com/gpu-mode/lectures/blob/main/lecture_014/A_Practitioners_Guide_to_Triton.ipynb)
