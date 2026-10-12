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