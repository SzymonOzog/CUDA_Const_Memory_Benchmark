# Const memory benchmarking
Trying to understand the why and when to use constant memory I stumbled across Lei Mao's blogpost https://leimao.github.io/blog/CUDA-Constant-Memory/

I decided to modify the script a little bit to get some more understanding for it:
- changed the vector datatypes to float
- cleared the cache between subsequent runs
- modified the script to run for multiple input sizes 
- graphed the results 

## One access per block
![ratio_block](https://github.com/user-attachments/assets/6d953dff-79e3-4035-905f-83654318ac6c)

## One access per thread 
![ratio_thread](https://github.com/user-attachments/assets/811aa71b-d886-468b-b8f6-ab2b1235468b)

## One access per warp 
![ratio_warp](https://github.com/user-attachments/assets/dd97dcd1-be2a-4d45-b892-dee8bbef4b54)

## Pseudo random 
![ratio_rand](https://github.com/user-attachments/assets/d04b1163-22a9-4c3b-934b-eff86c4afb0a)
