# vktest

this project is a Vulkan 1.3 meshlet renderer that loads glTF models, builds CPU meshlets with meshoptimizer, and renders many randomized model instances using GPU-driven indirect drawing. Each frame, a compute shader performs per-meshlet frustum sphere culling and meshlet cone backface culling, writes the surviving meshlet instances and indirect draw commands, then the graphics pass renders them with simple Lambert lighting and an ImGui stats overlay.

- HZB

### future work:

- Use an early/late occlusion pass like
  Early pass renders last-frame-visible meshlets to build useful depth. Late pass tests the rest against that depth. This is much more stable.

- Store/use raw reverse-Z depth instead of linear depth
  It avoids linearization cost and gives better precision distribution, requires changing depth compare logic.

- Use reduction sampler behavior
  sample one point with a min-reduction sampler. We currently sample four corners manually. We could make the HZB/sampler strategy match the chosen depth convention better.

- Clamp or reject tiny screen bounds (contribution culling)
  Very small meshlets can flicker because one texel decides visibility. For tiny screen bounds, either skip HZB or use a stronger bias.

## Build

```
cmake -S . -B build
cmake --build build --config Release
```
