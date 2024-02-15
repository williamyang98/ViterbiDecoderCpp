# Introduction
Files for setting up a aarch64 qemu emulator on ubuntu

Instructions taken from [here](https://azeria-labs.com/arm-on-x86-qemu-user/)

## Instructions
1. ```cd examples```
2. Install dependencies: ```./toolchains/arm_install_packages.sh```
3. Configure cmake: ```cmake . -B build --preset gcc-arm-simulator -DCMAKE_BUILD_TYPE=Release```
4. Build: ```cmake --build build --config Release```
5. Run: ```./toolchains/arm_run.sh ./build/run_tests```