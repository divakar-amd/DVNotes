# HIP/ROCm

### HIP & ROCm intro
- HIP: Heterogenous Interface for portability
- Rocm vs HIP: Rocm is the big open-source stack which encomposes HIP among other things
- Compute Unit ~ executes SIMD on GPU.
- CPU is latency oriented. GPU is throughput oriented (total workload)
- MI300X:
	- CUs : 304
   	- Streaming processors = CU * 64 = 19,456
	- Mem : 192 GB
 - Host & Device maintain their own memory separately 

### Memory & Device Management
- hipGetDeviceCount(int* count) returns no. of available "compute devices". - Some AMD GPUs (like those in laptop) are currently not considered "compute devices"
- hipGetDeviceProperties(hipDeviceProp_t* prop, int deviceId)
- hipMalloc(void** ptr, size_t size)
	- the ptr is defined on the cpu but the address returned is GPU address
- hipFree(void* ptr)
	- the ptr should be pointing to GPU address. Don't use this api to free CPU memory
- hipMemset(void* dst, int value, size_t sizeBytes) : useful for zeroing out device mem
- Error handling
	- HIP APIs return hipError_t
 	- cont char* hipGetErrorString(hipError_t hipError)
    	- hipError_t hipGetLastError(void)
 
### Simple kernel
- __ global __ functions have to return `void`
  	-  __ global __ void simplekernel(float *a){ ... }
-  However, device functions can return other stuff:
  	- __ device __ float otherKernel(float *a){ ... }
