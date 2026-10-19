cmake_minimum_required(VERSION 3.20)
project(ToyEngine)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# GLFW options - we don't need its docs/tests/examples built
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

add_subdirectory(external/glfw)

add_executable(ToyEngine src/main.cpp)
target_link_libraries(ToyEngine PRIVATE glfw opengl32)
target_include_directories(ToyEngine PRIVATE external/stb)

if (MSVC)
    target_compile_options(ToyEngine PRIVATE /EHsc)
endif()

find_package(Vulkan REQUIRED)

add_executable(ToyEngineVk src/vk_main.cpp)
target_link_libraries(ToyEngineVk PRIVATE glfw Vulkan::Vulkan)

if (MSVC)
    target_compile_options(ToyEngineVk PRIVATE /EHsc)
endif()


find_program(GLSLC glslc HINTS $ENV{VULKAN_SDK}/Bin)
file(GLOB SHADER_SOURCES ${CMAKE_SOURCE_DIR}/shaders/*.vert ${CMAKE_SOURCE_DIR}/shaders/*.frag)
foreach(SHADER ${SHADER_SOURCES})
    get_filename_component(SHADER_NAME ${SHADER} NAME)
    set(SPV ${CMAKE_SOURCE_DIR}/shaders/${SHADER_NAME}.spv)
    add_custom_command(OUTPUT ${SPV}
        COMMAND ${GLSLC} ${SHADER} -o ${SPV}
        DEPENDS ${SHADER})
    list(APPEND SPV_FILES ${SPV})
endforeach()
add_custom_target(Shaders DEPENDS ${SPV_FILES})
add_dependencies(ToyEngineVk Shaders)

add_executable(EcsBench src/ecs_bench.cpp)
target_compile_features(EcsBench PRIVATE cxx_std_17)
target_compile_options(EcsBench PRIVATE /EHsc /O2)


# Dear ImGui as a small static library
set(IMGUI_DIR ${CMAKE_SOURCE_DIR}/external/imgui)
add_library(imgui STATIC
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp)
target_include_directories(imgui PUBLIC ${IMGUI_DIR} ${IMGUI_DIR}/backends)
target_link_libraries(imgui PUBLIC glfw Vulkan::Vulkan)
target_link_libraries(ToyEngineVk PRIVATE imgui)

# ImGuizmo (3D transform gizmos), built into the imgui library
set(IMGUIZMO_DIR ${CMAKE_SOURCE_DIR}/external/ImGuizmo/src)
target_sources(imgui PRIVATE ${IMGUIZMO_DIR}/ImGuizmo.cpp)
target_include_directories(imgui PUBLIC ${IMGUIZMO_DIR})

target_sources(ToyEngineVk PRIVATE src/gltf_loader.cpp)
target_include_directories(ToyEngineVk PRIVATE ${CMAKE_SOURCE_DIR}/external/cgltf)

target_include_directories(ToyEngineVk PRIVATE ${CMAKE_SOURCE_DIR}/external/stb)

target_include_directories(ToyEngineVk PRIVATE ${CMAKE_SOURCE_DIR}/external/json)

target_compile_options(EcsBench PRIVATE $<$<CONFIG:Release>:/O2>)

file(GLOB RECAST_SOURCES
     ${CMAKE_SOURCE_DIR}/external/recastnavigation/Recast/Source/*.cpp
     ${CMAKE_SOURCE_DIR}/external/recastnavigation/Detour/Source/*.cpp)
add_library(RecastDetour STATIC ${RECAST_SOURCES})
target_include_directories(RecastDetour PUBLIC
     ${CMAKE_SOURCE_DIR}/external/recastnavigation/Recast/Include
     ${CMAKE_SOURCE_DIR}/external/recastnavigation/Detour/Include)

target_sources(ToyEngineVk PRIVATE src/navmesh.cpp)
target_link_libraries(ToyEngineVk PRIVATE RecastDetour)