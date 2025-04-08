# About LLMs and related things

## Table of Contents
**[1. Quantization](#quantization)**<br>
**[2. Training](#training)**<br>


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

### Training
- 3 stages:
    - **Pre-training**: Most compute heavy. Model learns broad/basic things like language structure.
    - **Post-training**: Fine-tuning, domain specific training.
    - **Instruct-training**: Train the model to better parse a human's prompt and understand what's asked. e.g.
      ```
      Prompt: "What's the capital of France?"

      # without instruct-training
      "What's the capital of Canada? What's the capital of Spain? ... "
      
      # with instruct-tuning
      "The capital of France is Paris....."
      ```
