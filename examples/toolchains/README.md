# Introduction
Files for setting up a aarch64 qemu emulator on ubuntu

Instructions taken from [here](https://azeria-labs.com/arm-on-x86-qemu-user/)

## Instructions
1. <code>cd examples</code>
2. <code>./toolchains/arm_install_packages.sh</code>
3. <code>./toolchains/arm_build.sh</code>
4. <code>ninja -C build-arm</code>
5. <code>./toolchains/arm_run.sh ./build-arm/run_tests</code>