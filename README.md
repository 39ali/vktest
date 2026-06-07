# vktest

Minimal Vulkan glTF foundation inspired by Sascha Willems' `gltfloading.cpp`.

The app creates a GLFW window, initializes Vulkan, loads a glTF file through
tinygltf, flattens scene nodes into one vertex buffer and one index buffer, and
uploads both through host-visible staging buffers into device-local buffers.

## Build

```powershell
cmake -S . -B build
cmake --build build --config Debug
```

Run with the bundled Duck glTF sample:

```powershell
.\build\Debug\vktest.exe
```

Or pass another `.gltf`/`.glb`:

```powershell
.\build\Debug\vktest.exe C:\path\to\model.gltf
```

The first configure may download GLFW, glm, and tinygltf if they are not already
available through CMake packages.
