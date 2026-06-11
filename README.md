# vktest

this project is a Vulkan 1.3 meshlet renderer that loads glTF models, builds CPU meshlets with meshoptimizer, and renders many randomized model instances using GPU-driven indirect drawing. Each frame, a compute shader performs per-meshlet frustum sphere culling and meshlet cone backface culling, writes the surviving meshlet instances and indirect draw commands, then the graphics pass renders them with simple Lambert lighting and an ImGui stats overlay.

## Build

```
cmake -S . -B build
cmake --build build --config Release
```
