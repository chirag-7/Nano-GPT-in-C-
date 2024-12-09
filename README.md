# C++ version of nanoGPT

This implementation is based on Andrej Kharpaty's [repo](https://github.com/karpathy/nanogpt) and kikirizki's [repo](https://github.com/kikirizki/nanogpt_cpp). Instead of using Shakespeare's works, this project explores the unique speaking and writing style of Jim Simons, the renowned mathematician-turned-hedge fund manager, by training on a curated dataset of his interviews, speeches, and other publicly available materials. This project is implemented in C++ to refresh my skills with the language after a long time and to challenge myself to better understand the inner workings of GPT models at a lower level, beyond the abstractions provided by Python frameworks.

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
