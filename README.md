# C++ version of nanoGPT

This implementation is based on Andrej Kharpaty's [repo](https://github.com/karpathy/nanogpt) and kikirizki's [repo](https://github.com/kikirizki/nanogpt_cpp). This project is implemented in C++ to refresh my skills with the language after a long time and to challenge myself to better understand the inner workings of GPT models at a lower level, beyond the abstractions provided by Python frameworks.

## Download dataset

From tiny_.txt in the github 

## Compile
```bash
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/usr/local/libtorch ..
cmake --build . --config Release
```

## Roadmap
- [ ] GGLM implementation
- [x] Libtorch implementation (WIP)
- [ ] ncnn implementation
