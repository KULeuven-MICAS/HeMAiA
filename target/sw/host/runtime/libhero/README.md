# Libhero

Libhero is the librarie used by the OpenMP RTL (or any higher) layer, to interact with the devices.
It wraps calls to the drivers into normalized functions and implement a default HERO device library to allow the higher part of the stack to be cross-platform. 

Libhero uses [o1heap](https://github.com/pavel-kirienko/o1heap) to manage device memory allocation.

Since the current HeMAiA project is focusing on the baremetal development, this libhero removes the Virtual Memory Management components.