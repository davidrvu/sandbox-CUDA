# sandbox-CUDA
Playground for CUDA


## CUDA C/C++ -> DEVICE -> Code Generation  

compute_20,sm_20;compute_30,sm_30;compute_35,sm_35;compute_37,sm_37;compute_50,sm_50;compute_52,sm_52;  


### PARA NVIDIA QUADRO M6000: compute_52,sm_52;  


### Supported on CUDA 7 and later  
SM20 – Older cards such as GeForce GT630  
SM30 – Kepler architecture (generic – Tesla K40/K80)  
Adds support for unified memory programming  
SM35 – More specific Tesla K40/K80.  

### Adds support for dynamic parallelism. Shows no real benefit over SM30 in my experience.  
SM50 – Tesla/Quadro M series  
SM52 – Quadro M6000 , GTX 980/Titan  
SM53 – Tegra TX1  

### Supported on CUDA 8 and later  
SM60 – GP100/Pascal P100 – DGX-1 (Generic Pascal)  
SM61 – GTX 1080, 1070, 1060  
SM62 – Future versions – Unknown for now, but opinions vary between the new Drive-PX2 or the GTX 1080Ti and Titan-P  