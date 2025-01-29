# About LLMs and related things

## Table of Contents
**[1. Quantization](#quantization)**<br>


<br> 

### Quantization
- Consider a 2D matrix MxN where M = num_tokens and N = num_channels
- 3 modes:
    - **Per-tensor** scaling: Use a single scaling factor for the entire 2D matrix
    - **Per-token** scaling: Use "M" scaling factors, one for each row
    - **Per-channel** scaling: Use "N" scaling factors, one for each channel


| Name        | precision    |                                                  |
|-------------|--------------|--------------------------------------------------|
| SmoothQuant | Int8 (W8A8)  | Requires pre-processing the weights              |
| GPTQ, AWQ   | W4A16        | AWQ: Preserves & doesn't quantize all the weights |
| Weight only | W4A16, W8A16 |                                                  |
