cmake_minimum_required(VERSION 3.24)

project(vktest LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

find_package(Vulkan REQUIRED)

find_package(glfw3 QUIET)
if (NOT glfw3_FOUND)
  FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.4
  )
  set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(glfw)
endif()

find_package(glm QUIET)
if (NOT glm_FOUND)
  FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG 1.0.1
  )
  FetchContent_MakeAvailable(glm)
endif()

FetchContent_Declare(
  tinygltf
  GIT_REPOSITORY https://github.com/syoyo/tinygltf.git
  GIT_TAG v2.9.3
)
set(TINYGLTF_BUILD_LOADER_EXAMPLE OFF CACHE BOOL "" FORCE)
set(TINYGLTF_BUILD_GL_EXAMPLES OFF CACHE BOOL "" FORCE)
set(TINYGLTF_BUILD_VALIDATOR_EXAMPLE OFF CACHE BOOL "" FORCE)
set(TINYGLTF_BUILD_BUILDER_EXAMPLE OFF CACHE BOOL "" FORCE)
set(TINYGLTF_HEADER_ONLY ON CACHE BOOL "" FORCE)
set(TINYGLTF_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(tinygltf)

FetchContent_Declare(
  meshoptimizer
  GIT_REPOSITORY https://github.com/zeux/meshoptimizer.git
  GIT_TAG v0.23
)
set(MESHOPT_BUILD_DEMO OFF CACHE BOOL "" FORCE)
set(MESHOPT_BUILD_GLTFPACK OFF CACHE BOOL "" FORCE)
set(MESHOPT_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(meshoptimizer)

FetchContent_Declare(
  imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG v1.91.9b
)
FetchContent_MakeAvailable(imgui)

FetchContent_Declare(
  vma
  GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
  GIT_TAG v3.3.0
)
FetchContent_MakeAvailable(vma)

find_program(GLSLC_EXECUTABLE glslc HINTS "$ENV{VULKAN_SDK}/Bin")
if (NOT GLSLC_EXECUTABLE)
  message(FATAL_ERROR "Could not find glslc. Install the Vulkan SDK or add glslc to PATH.")
endif()

set(SHADER_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/gltf.vert
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/gltf.frag
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/meshlet_cull.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/hzb_reduce.comp
)

set(SHADER_INCLUDES
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/shared_structs.h
)

set(SPIRV_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SPIRV_BINARY_DIR})

foreach(shader ${SHADER_SOURCES})
  get_filename_component(shader_name ${shader} NAME)
  set(spirv ${SPIRV_BINARY_DIR}/${shader_name}.spv)
  add_custom_command(
    OUTPUT ${spirv}
    COMMAND ${GLSLC_EXECUTABLE} -I ${CMAKE_CURRENT_SOURCE_DIR}/shaders
            ${shader} -o ${spirv}
    DEPENDS ${shader} ${SHADER_INCLUDES}
    VERBATIM
  )
  list(APPEND SPIRV_BINARIES ${spirv})
endforeach()

add_custom_target(shaders DEPENDS ${SPIRV_BINARIES})

add_executable(vktest
  src/main.cpp
  src/app.cpp
  src/camera.cpp
  src/renderer.cpp
  src/resource.cpp
  src/vk_context.cpp
  ${imgui_SOURCE_DIR}/imgui.cpp
  ${imgui_SOURCE_DIR}/imgui_draw.cpp
  ${imgui_SOURCE_DIR}/imgui_tables.cpp
  ${imgui_SOURCE_DIR}/imgui_widgets.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
)
add_dependencies(vktest shaders)

target_include_directories(vktest PRIVATE
  ${tinygltf_SOURCE_DIR}
  ${imgui_SOURCE_DIR}
  ${imgui_SOURCE_DIR}/backends
  ${vma_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders
)
target_link_libraries(vktest PRIVATE Vulkan::Vulkan meshoptimizer)

if (TARGET glfw)
  target_link_libraries(vktest PRIVATE glfw)
else()
  target_link_libraries(vktest PRIVATE glfw)
endif()

if (TARGET glm::glm)
  target_link_libraries(vktest PRIVATE glm::glm)
endif()

target_compile_definitions(vktest PRIVATE
  GLFW_INCLUDE_VULKAN
  GLM_FORCE_RADIANS
  GLM_FORCE_DEPTH_ZERO_TO_ONE
)

add_custom_command(TARGET vktest POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${SPIRV_BINARY_DIR}
    $<TARGET_FILE_DIR:vktest>/shaders
  COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/assets
    $<TARGET_FILE_DIR:vktest>/assets
)
