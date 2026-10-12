#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>

#include <iostream>
#include <vector>
#include <set>
#include <optional>
#include <cstring>
#include <stdexcept>

#include "math3d.h"
#include "bench_scene.h"
#include "components.h"
#include "ecs.h"
#include "ecs_systems.h"

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_vulkan.h"

// Benchmark sizes for this renderer. Instancing makes 100,000 cubes practical, so the list
// goes one step past the shared BENCH_COUNTS (which the software renderer also uses).
inline constexpr int VK_BENCH_COUNTS[] = { 1, 10, 100, 1000, 10000, 100000 };
inline constexpr int VK_BENCH_NUM_COUNTS = (int)(sizeof(VK_BENCH_COUNTS) / sizeof(VK_BENCH_COUNTS[0]));
inline constexpr int MAX_CUBES = 100000;
inline constexpr int MAX_INSTANCES = MAX_CUBES + 1;   // + the floor

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;
const int MAX_FRAMES_IN_FLIGHT = 2;

const std::vector<const char*> validationLayers = { "VK_LAYER_KHRONOS_validation" };
const std::vector<const char*> deviceExtensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };

#ifdef NDEBUG
const bool enableValidation = false;
#else
const bool enableValidation = true;
#endif

struct QueueFamilyIndices {
    std::optional<uint32_t> graphics;
    std::optional<uint32_t> present;
    bool complete() const { return graphics.has_value() && present.has_value(); }
};

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT,
    const VkDebugUtilsMessengerCallbackDataEXT* data,
    void*) {
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        std::cerr << "[Vulkan] " << data->pMessage << "\n";
    }
    return VK_FALSE;
}

struct SwapchainSupport {
    VkSurfaceCapabilitiesKHR caps{};
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

struct MeshVertex {
    Vec3 pos;
    Vec3 color;
    Vec3 normal;
    float u, v;
};

struct LightGPU {
    float position[4]; // xyz = world position
    float color[4];    // rgb = color, a = intensity
};

// Layout matches the GLSL block under std140 rules: 64 + 16 + 3*32 + 16 = 192 bytes
struct SceneUBO {
    Mat4 viewProj;
    float cameraPos[4];
    LightGPU lights[3];
    int32_t info[4];   // x = number of active lights
};
static_assert(sizeof(SceneUBO) == 192, "SceneUBO must match the GLSL std140 layout");

struct PushData {
    Mat4 model;
    float params[4];   // x = checker on/off, y = shininess
};

class App {
public:
    bool benchMode = false; // set from --bench before run()

    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    GLFWwindow* window = nullptr;
    VkInstance instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue graphicsQueue = VK_NULL_HANDLE;
    VkQueue presentQueue = VK_NULL_HANDLE;

    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    std::vector<VkImage> swapchainImages;
    VkFormat swapchainFormat{};
    VkExtent2D swapchainExtent{};
    std::vector<VkImageView> swapchainImageViews;
    VkRenderPass renderPass = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline graphicsPipeline = VK_NULL_HANDLE;
    VkPipeline wireframePipeline = VK_NULL_HANDLE;   // same shaders, polygon mode LINE (if the GPU supports it)
    std::vector<VkFramebuffer> swapchainFramebuffers;

    VkCommandPool commandPool = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> commandBuffers;              // one per frame in flight
    std::vector<VkSemaphore> imageAvailableSemaphores;        // one per frame in flight
    std::vector<VkSemaphore> renderFinishedSemaphores;        // one per swapchain image
    std::vector<VkFence> inFlightFences;                      // one per frame in flight
    uint32_t currentFrame = 0;

    // Depth buffer
    VkFormat depthFormat{};
    VkImage depthImage = VK_NULL_HANDLE;
    VkDeviceMemory depthMemory = VK_NULL_HANDLE;
    VkImageView depthView = VK_NULL_HANDLE;

    // Mesh data (CPU side, then uploaded)
    std::vector<MeshVertex> meshVertices;
    std::vector<uint32_t> meshIndices;
    uint32_t cubeIndexCount = 0, floorFirstIndex = 0, floorIndexCount = 0;
    VkBuffer vertexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory vertexMemory = VK_NULL_HANDLE;
    VkBuffer indexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory indexMemory = VK_NULL_HANDLE;

    // Uniforms (one buffer + descriptor set per frame in flight)
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> descriptorSets;
    std::vector<VkBuffer> uniformBuffers;
    std::vector<VkDeviceMemory> uniformMemory;
    std::vector<void*> uniformMapped;

    // The world: every cube is an entity in the ECS. Each frame the draw list built from it
    // is copied into this frame slot's instance buffer (one DrawItem per instance).
    Registry world;
    std::vector<DrawItem> drawList;
    std::vector<VkBuffer> instanceBuffers;           // one per frame in flight
    std::vector<VkDeviceMemory> instanceMemory;
    std::vector<void*> instanceMapped;
    uint32_t drawnCubes = 0;                          // cubes in this frame's draw list (the floor follows them)
    bool instanced = true;                            // false = one draw call per cube (benchmark comparison)
    double lastSimTime = 0.0;                         // for the spin system's dt
    double lastSimMs = 0.0;                           // ECS update + draw list + upload, last frame
    double benchSimMsSum = 0.0;

    // Debug overlay (Dear ImGui). Disabled in --bench so it can't affect the measurements.
    bool imguiEnabled = false;
    VkDescriptorPool imguiPool = VK_NULL_HANDLE;
    bool wireframeSupported = false;
    bool wireframe = false;         // toggle: draw triangle edges only
    bool lightingOn = true;         // toggle: off = flat albedo, no lighting
    bool simPaused = false;         // toggle: stop the spin system
    bool camFrozen = false;         // toggle: stop the camera sway
    float camTime = 0.0f;           // advances only while the camera is not frozen
    uint32_t drawCallsLastFrame = 0;
    double frameDt = 0.0, prevFrameTime = 0.0;
    double msSpin = 0.0, msDrawList = 0.0, msUpload = 0.0;   // per-system CPU timings, last frame
    double smoothFrameMs = 0.0, smoothCpuMs = 0.0, smoothGpuMs = 0.0, lastGpuMs = 0.0;
    static constexpr int HISTORY = 120;
    float frameHistory[HISTORY] = {};
    int historyPos = 0;

    // Benchmark state
    int cubeCount = 1;
    int benchStage = 0, benchFrames = 0;
    bool benchMeasuring = false;
    double benchStageStart = 0.0, benchMeasureStart = 0.0;

    // Timing: GPU time via timestamp queries (2 per frame in flight), CPU time for record + submit
    VkQueryPool queryPool = VK_NULL_HANDLE;
    double timestampPeriodNs = 1.0;
    uint64_t framesSubmitted = 0;
    double lastCpuMs = 0.0;
    double benchCpuMsSum = 0.0, benchGpuMsSum = 0.0;
    int benchGpuSamples = 0;

    double lastTitleTime = 0.0;
    int frameCount = 0;

    void initWindow() {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // Vulkan: no OpenGL context
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);   // resizing handled later with swapchain recreation
        window = glfwCreateWindow(WIDTH, HEIGHT, "Toy Engine (Vulkan)", nullptr, nullptr);
        if (!window) throw std::runtime_error("Failed to create window");
    }

    void initVulkan() {
        createInstance();
        setupDebugMessenger();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapchain();
        createImageViews();
        createRenderPass();
        createDescriptorSetLayout();
        createGraphicsPipeline();
        createCommandPool();
        createDepthResources();
        createFramebuffers();
        createMeshBuffers();
        createInstanceBuffers();
        createUniformBuffers();
        createDescriptorSets();
        createCommandBuffers();
        createSyncObjects();
        createQueryPool();
        if (!benchMode) initImGui();
    }

    void mainLoop() {
        if (benchMode) {
            cubeCount = VK_BENCH_COUNTS[0];
            instanced = false;   // first half of the run: one draw call per cube
            printf("mode,cubes,triangles,frame_ms,fps,sim_ms,cpu_ms,gpu_ms\n");
        }
        spawnScene();
        prevFrameTime = glfwGetTime();   // so the first frame's dt isn't "time since startup"
        benchStageStart = glfwGetTime();

        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            drawFrame();
        }
        vkDeviceWaitIdle(device); // let the GPU finish before we start destroying things
    }

    void cleanup() {
        if (imguiEnabled) {
            ImGui_ImplVulkan_Shutdown();
            ImGui_ImplGlfw_Shutdown();
            ImGui::DestroyContext();
            vkDestroyDescriptorPool(device, imguiPool, nullptr);
        }
        for (auto s : imageAvailableSemaphores) vkDestroySemaphore(device, s, nullptr);
        for (auto s : renderFinishedSemaphores) vkDestroySemaphore(device, s, nullptr);
        for (auto f : inFlightFences) vkDestroyFence(device, f, nullptr);
        vkDestroyQueryPool(device, queryPool, nullptr);
        vkDestroyCommandPool(device, commandPool, nullptr); // also frees its command buffers

        for (auto fb : swapchainFramebuffers) vkDestroyFramebuffer(device, fb, nullptr);
        vkDestroyPipeline(device, graphicsPipeline, nullptr);
        if (wireframePipeline != VK_NULL_HANDLE) vkDestroyPipeline(device, wireframePipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyRenderPass(device, renderPass, nullptr);

        vkDestroyImageView(device, depthView, nullptr);
        vkDestroyImage(device, depthImage, nullptr);
        vkFreeMemory(device, depthMemory, nullptr);

        for (auto iv : swapchainImageViews) vkDestroyImageView(device, iv, nullptr);
        vkDestroySwapchainKHR(device, swapchain, nullptr);

        for (size_t i = 0; i < uniformBuffers.size(); i++) {
            vkDestroyBuffer(device, uniformBuffers[i], nullptr);
            vkFreeMemory(device, uniformMemory[i], nullptr);
        }
        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

        for (size_t i = 0; i < instanceBuffers.size(); i++) {
            vkDestroyBuffer(device, instanceBuffers[i], nullptr);
            vkFreeMemory(device, instanceMemory[i], nullptr);
        }

        vkDestroyBuffer(device, indexBuffer, nullptr);
        vkFreeMemory(device, indexMemory, nullptr);
        vkDestroyBuffer(device, vertexBuffer, nullptr);
        vkFreeMemory(device, vertexMemory, nullptr);

        vkDestroyDevice(device, nullptr);
        if (enableValidation) destroyDebugMessenger();
        vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyInstance(instance, nullptr);
        glfwDestroyWindow(window);
        glfwTerminate();
    }

    // ---------- Instance ----------

    bool checkValidationSupport() {
        uint32_t count = 0;
        vkEnumerateInstanceLayerProperties(&count, nullptr);
        std::vector<VkLayerProperties> layers(count);
        vkEnumerateInstanceLayerProperties(&count, layers.data());
        for (const char* name : validationLayers) {
            bool found = false;
            for (auto& l : layers) if (strcmp(name, l.layerName) == 0) { found = true; break; }
            if (!found) return false;
        }
        return true;
    }

    std::vector<const char*> requiredExtensions() {
        uint32_t count = 0;
        const char** glfwExt = glfwGetRequiredInstanceExtensions(&count);
        std::vector<const char*> ext(glfwExt, glfwExt + count);
        if (enableValidation) ext.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        return ext;
    }

    static void fillDebugInfo(VkDebugUtilsMessengerCreateInfoEXT& info) {
        info = {};
        info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        info.pfnUserCallback = debugCallback;
    }

    void createInstance() {
        if (enableValidation && !checkValidationSupport())
            throw std::runtime_error("Validation layers requested but not available (is the Vulkan SDK installed?)");

        VkApplicationInfo app{};
        app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app.pApplicationName = "Toy Engine";
        app.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
        app.pEngineName = "ToyEngine";
        app.engineVersion = VK_MAKE_VERSION(0, 1, 0);
        app.apiVersion = VK_API_VERSION_1_3;

        auto extensions = requiredExtensions();

        VkInstanceCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        info.pApplicationInfo = &app;
        info.enabledExtensionCount = (uint32_t)extensions.size();
        info.ppEnabledExtensionNames = extensions.data();

        // Chaining a messenger into pNext lets validation cover instance creation itself
        VkDebugUtilsMessengerCreateInfoEXT debugInfo{};
        if (enableValidation) {
            info.enabledLayerCount = (uint32_t)validationLayers.size();
            info.ppEnabledLayerNames = validationLayers.data();
            fillDebugInfo(debugInfo);
            info.pNext = &debugInfo;
        }

        if (vkCreateInstance(&info, nullptr, &instance) != VK_SUCCESS)
            throw std::runtime_error("Failed to create Vulkan instance");
    }

    // ---------- Debug messenger ----------

    void setupDebugMessenger() {
        if (!enableValidation) return;
        VkDebugUtilsMessengerCreateInfoEXT info;
        fillDebugInfo(info);
        auto fn = (PFN_vkCreateDebugUtilsMessengerEXT)
            vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
        if (!fn || fn(instance, &info, nullptr, &debugMessenger) != VK_SUCCESS)
            throw std::runtime_error("Failed to set up debug messenger");
    }

    void destroyDebugMessenger() {
        auto fn = (PFN_vkDestroyDebugUtilsMessengerEXT)
            vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
        if (fn) fn(instance, debugMessenger, nullptr);
    }

    // ---------- Surface ----------

    void createSurface() {
        if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS)
            throw std::runtime_error("Failed to create window surface");
    }

    // ---------- Physical device ----------

    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice dev) {
        QueueFamilyIndices idx;
        uint32_t count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &count, nullptr);
        std::vector<VkQueueFamilyProperties> families(count);
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &count, families.data());

        for (uint32_t i = 0; i < count; i++) {
            if (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) idx.graphics = i;
            VkBool32 presentSupport = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(dev, i, surface, &presentSupport);
            if (presentSupport) idx.present = i;
            if (idx.complete()) break;
        }
        return idx;
    }

    bool supportsDeviceExtensions(VkPhysicalDevice dev) {
        uint32_t count = 0;
        vkEnumerateDeviceExtensionProperties(dev, nullptr, &count, nullptr);
        std::vector<VkExtensionProperties> available(count);
        vkEnumerateDeviceExtensionProperties(dev, nullptr, &count, available.data());
        std::set<std::string> required(deviceExtensions.begin(), deviceExtensions.end());
        for (auto& e : available) required.erase(e.extensionName);
        return required.empty();
    }

    int scoreDevice(VkPhysicalDevice dev) {
        if (!findQueueFamilies(dev).complete()) return 0;
        if (!supportsDeviceExtensions(dev)) return 0;
        SwapchainSupport sc = querySwapchainSupport(dev);
        if (sc.formats.empty() || sc.presentModes.empty()) return 0;
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(dev, &props);
        int score = 1;
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) score += 1000;
        return score;
    }

    void pickPhysicalDevice() {
        uint32_t count = 0;
        vkEnumeratePhysicalDevices(instance, &count, nullptr);
        if (count == 0) throw std::runtime_error("No Vulkan-capable GPU found");
        std::vector<VkPhysicalDevice> devices(count);
        vkEnumeratePhysicalDevices(instance, &count, devices.data());

        int best = 0;
        for (auto d : devices) {
            int s = scoreDevice(d);
            if (s > best) { best = s; physicalDevice = d; }
        }
        if (physicalDevice == VK_NULL_HANDLE) throw std::runtime_error("No suitable GPU found");

        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(physicalDevice, &props);
        std::cout << "Using GPU: " << props.deviceName << "\n";
    }

    // ---------- Logical device ----------

    void createLogicalDevice() {
        QueueFamilyIndices q = findQueueFamilies(physicalDevice);
        std::set<uint32_t> uniqueFamilies = { q.graphics.value(), q.present.value() };

        float priority = 1.0f;
        std::vector<VkDeviceQueueCreateInfo> queueInfos;
        for (uint32_t family : uniqueFamilies) {
            VkDeviceQueueCreateInfo qi{};
            qi.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            qi.queueFamilyIndex = family;
            qi.queueCount = 1;
            qi.pQueuePriorities = &priority;
            queueInfos.push_back(qi);
        }

        VkPhysicalDeviceFeatures supported{};
        vkGetPhysicalDeviceFeatures(physicalDevice, &supported);
        VkPhysicalDeviceFeatures features{};
        if (supported.fillModeNonSolid) {          // needed for the wireframe toggle
            features.fillModeNonSolid = VK_TRUE;
            wireframeSupported = true;
        }

        VkDeviceCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        info.queueCreateInfoCount = (uint32_t)queueInfos.size();
        info.pQueueCreateInfos = queueInfos.data();
        info.pEnabledFeatures = &features;
        info.enabledExtensionCount = (uint32_t)deviceExtensions.size();
        info.ppEnabledExtensionNames = deviceExtensions.data();

        if (vkCreateDevice(physicalDevice, &info, nullptr, &device) != VK_SUCCESS)
            throw std::runtime_error("Failed to create logical device");

        vkGetDeviceQueue(device, q.graphics.value(), 0, &graphicsQueue);
        vkGetDeviceQueue(device, q.present.value(), 0, &presentQueue);
    }

    // ---------- Swapchain ----------

    SwapchainSupport querySwapchainSupport(VkPhysicalDevice dev) {
        SwapchainSupport s;
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(dev, surface, &s.caps);
        uint32_t n = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(dev, surface, &n, nullptr);
        s.formats.resize(n);
        vkGetPhysicalDeviceSurfaceFormatsKHR(dev, surface, &n, s.formats.data());
        vkGetPhysicalDeviceSurfacePresentModesKHR(dev, surface, &n, nullptr);
        s.presentModes.resize(n);
        vkGetPhysicalDeviceSurfacePresentModesKHR(dev, surface, &n, s.presentModes.data());
        return s;
    }

    void createSwapchain() {
        SwapchainSupport sup = querySwapchainSupport(physicalDevice);

        // Format: prefer 8-bit BGRA with sRGB encoding
        VkSurfaceFormatKHR fmt = sup.formats[0];
        for (auto& f : sup.formats) {
            if (f.format == VK_FORMAT_B8G8R8A8_SRGB &&
                f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) { fmt = f; break; }
        }

        // Present mode: IMMEDIATE while benchmarking (no vsync cap), otherwise MAILBOX if available,
        // else FIFO (vsync, always supported)
        VkPresentModeKHR wanted = benchMode ? VK_PRESENT_MODE_IMMEDIATE_KHR : VK_PRESENT_MODE_MAILBOX_KHR;
        VkPresentModeKHR mode = VK_PRESENT_MODE_FIFO_KHR;
        for (auto m : sup.presentModes)
            if (m == wanted) { mode = m; break; }

        // Extent: the size of the images, normally the window size in pixels
        VkExtent2D extent;
        if (sup.caps.currentExtent.width != UINT32_MAX) {
            extent = sup.caps.currentExtent;
        } else {
            int w, h;
            glfwGetFramebufferSize(window, &w, &h);
            extent.width  = std::clamp((uint32_t)w, sup.caps.minImageExtent.width,  sup.caps.maxImageExtent.width);
            extent.height = std::clamp((uint32_t)h, sup.caps.minImageExtent.height, sup.caps.maxImageExtent.height);
        }

        uint32_t imageCount = sup.caps.minImageCount + 1;
        if (sup.caps.maxImageCount > 0 && imageCount > sup.caps.maxImageCount)
            imageCount = sup.caps.maxImageCount;

        VkSwapchainCreateInfoKHR info{};
        info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        info.surface = surface;
        info.minImageCount = imageCount;
        info.imageFormat = fmt.format;
        info.imageColorSpace = fmt.colorSpace;
        info.imageExtent = extent;
        info.imageArrayLayers = 1;
        info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

        QueueFamilyIndices q = findQueueFamilies(physicalDevice);
        uint32_t families[] = { q.graphics.value(), q.present.value() };
        if (q.graphics != q.present) {
            info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            info.queueFamilyIndexCount = 2;
            info.pQueueFamilyIndices = families;
        } else {
            info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }

        info.preTransform = sup.caps.currentTransform;
        info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        info.presentMode = mode;
        info.clipped = VK_TRUE;
        info.oldSwapchain = VK_NULL_HANDLE;

        if (vkCreateSwapchainKHR(device, &info, nullptr, &swapchain) != VK_SUCCESS)
            throw std::runtime_error("Failed to create swapchain");

        vkGetSwapchainImagesKHR(device, swapchain, &imageCount, nullptr);
        swapchainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(device, swapchain, &imageCount, swapchainImages.data());
        swapchainFormat = fmt.format;
        swapchainExtent = extent;

        std::cout << "Swapchain: " << extent.width << "x" << extent.height
                  << ", " << imageCount << " images, present mode "
                  << (mode == VK_PRESENT_MODE_MAILBOX_KHR ? "MAILBOX" :
                      mode == VK_PRESENT_MODE_IMMEDIATE_KHR ? "IMMEDIATE" : "FIFO") << "\n";
    }

    void createImageViews() {
        swapchainImageViews.resize(swapchainImages.size());
        for (size_t i = 0; i < swapchainImages.size(); i++) {
            VkImageViewCreateInfo info{};
            info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            info.image = swapchainImages[i];
            info.viewType = VK_IMAGE_VIEW_TYPE_2D;
            info.format = swapchainFormat;
            info.components = { VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                                VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY };
            info.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            if (vkCreateImageView(device, &info, nullptr, &swapchainImageViews[i]) != VK_SUCCESS)
                throw std::runtime_error("Failed to create image view");
        }
    }

    // ---------- Render pass ----------

    VkFormat findDepthFormat() {
        for (VkFormat f : { VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT }) {
            VkFormatProperties p;
            vkGetPhysicalDeviceFormatProperties(physicalDevice, f, &p);
            if (p.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) return f;
        }
        throw std::runtime_error("No supported depth format");
    }

    void createRenderPass() {
        depthFormat = findDepthFormat();

        VkAttachmentDescription color{};
        color.format = swapchainFormat;
        color.samples = VK_SAMPLE_COUNT_1_BIT;
        color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        color.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        color.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        color.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentDescription depth{};
        depth.format = depthFormat;
        depth.samples = VK_SAMPLE_COUNT_1_BIT;
        depth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;        // reset to "far" every frame
        depth.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;  // never shown, so no need to keep it
        depth.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depth.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depth.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depth.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentReference colorRef{ 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
        VkAttachmentReference depthRef{ 1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorRef;
        subpass.pDepthStencilAttachment = &depthRef;

        VkSubpassDependency dep{};
        dep.srcSubpass = VK_SUBPASS_EXTERNAL;
        dep.dstSubpass = 0;
        dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                           VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                           VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        dep.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                           VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

        VkAttachmentDescription attachments[2] = { color, depth };

        VkRenderPassCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        info.attachmentCount = 2;
        info.pAttachments = attachments;
        info.subpassCount = 1;
        info.pSubpasses = &subpass;
        info.dependencyCount = 1;
        info.pDependencies = &dep;

        if (vkCreateRenderPass(device, &info, nullptr, &renderPass) != VK_SUCCESS)
            throw std::runtime_error("Failed to create render pass");
    }

    // ---------- Graphics pipeline ----------

    static std::vector<char> readFile(const std::string& path) {
        std::ifstream file(path, std::ios::ate | std::ios::binary);
        if (!file.is_open())
            throw std::runtime_error("Failed to open " + path + " (run from the project root)");
        size_t size = (size_t)file.tellg();
        std::vector<char> buf(size);
        file.seekg(0);
        file.read(buf.data(), size);
        return buf;
    }

    VkShaderModule createShaderModule(const std::vector<char>& code) {
        VkShaderModuleCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        info.codeSize = code.size();
        info.pCode = reinterpret_cast<const uint32_t*>(code.data());
        VkShaderModule mod;
        if (vkCreateShaderModule(device, &info, nullptr, &mod) != VK_SUCCESS)
            throw std::runtime_error("Failed to create shader module");
        return mod;
    }

    void createGraphicsPipeline() {
        VkShaderModule vert = createShaderModule(readFile("shaders/mesh.vert.spv"));
        VkShaderModule frag = createShaderModule(readFile("shaders/mesh.frag.spv"));

        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vert;
        stages[0].pName = "main";
        stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = frag;
        stages[1].pName = "main";

        // Describes how the bytes in the vertex buffer map to shader inputs
        // Binding 0: per-vertex mesh data. Binding 1: per-INSTANCE data (the model matrix),
        // advanced once per instance instead of once per vertex.
        static_assert(offsetof(DrawItem, model) == 0, "the model matrix must be first in DrawItem");
        VkVertexInputBindingDescription bindings[2] = {
            { 0, sizeof(MeshVertex), VK_VERTEX_INPUT_RATE_VERTEX },
            { 1, sizeof(DrawItem),   VK_VERTEX_INPUT_RATE_INSTANCE },
        };
        // A mat4 attribute occupies 4 consecutive locations (4..7), one column each.
        // Our Mat4 is column-major with 16 contiguous floats, so the columns sit at +0/16/32/48.
        VkVertexInputAttributeDescription attrs[8] = {
            { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, (uint32_t)offsetof(MeshVertex, pos) },
            { 1, 0, VK_FORMAT_R32G32B32_SFLOAT, (uint32_t)offsetof(MeshVertex, color) },
            { 2, 0, VK_FORMAT_R32G32B32_SFLOAT, (uint32_t)offsetof(MeshVertex, normal) },
            { 3, 0, VK_FORMAT_R32G32_SFLOAT,    (uint32_t)offsetof(MeshVertex, u) },
            { 4, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 0 },
            { 5, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 16 },
            { 6, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 32 },
            { 7, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 48 },
        };
        VkPipelineVertexInputStateCreateInfo vertexInput{};
        vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInput.vertexBindingDescriptionCount = 2;
        vertexInput.pVertexBindingDescriptions = bindings;
        vertexInput.vertexAttributeDescriptionCount = 8;
        vertexInput.pVertexAttributeDescriptions = attrs;

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        // Viewport and scissor are set while recording commands (dynamic)
        VkPipelineViewportStateCreateInfo viewport{};
        viewport.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewport.viewportCount = 1;
        viewport.scissorCount = 1;

        VkDynamicState dynStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        VkPipelineDynamicStateCreateInfo dynamic{};
        dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamic.dynamicStateCount = 2;
        dynamic.pDynamicStates = dynStates;

        VkPipelineRasterizationStateCreateInfo raster{};
        raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        raster.polygonMode = VK_POLYGON_MODE_FILL;
        raster.lineWidth = 1.0f;
        // Benchmark draws every triangle, matching the software renderer (which has no culling)
        raster.cullMode = benchMode ? VK_CULL_MODE_NONE : VK_CULL_MODE_BACK_BIT;
        raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; // our meshes are CCW; the Y flip keeps that true on screen

        VkPipelineMultisampleStateCreateInfo msaa{};
        msaa.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        msaa.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo depthStencil{};
        depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencil.depthTestEnable = VK_TRUE;
        depthStencil.depthWriteEnable = VK_TRUE;
        depthStencil.depthCompareOp = VK_COMPARE_OP_LESS; // smaller depth = closer = wins (same rule as Phase 1)

        VkPipelineColorBlendAttachmentState blendAttachment{};
        blendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                         VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        blendAttachment.blendEnable = VK_FALSE;

        VkPipelineColorBlendStateCreateInfo blend{};
        blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        blend.attachmentCount = 1;
        blend.pAttachments = &blendAttachment;

        // Pipeline layout: one descriptor set (scene UBO) + one push constant block (model matrix + params)
        VkPushConstantRange push{ VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushData) };
        VkPipelineLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layoutInfo.setLayoutCount = 1;
        layoutInfo.pSetLayouts = &descriptorSetLayout;
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges = &push;
        if (vkCreatePipelineLayout(device, &layoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS)
            throw std::runtime_error("Failed to create pipeline layout");

        VkGraphicsPipelineCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        info.stageCount = 2;
        info.pStages = stages;
        info.pVertexInputState = &vertexInput;
        info.pInputAssemblyState = &inputAssembly;
        info.pViewportState = &viewport;
        info.pRasterizationState = &raster;
        info.pMultisampleState = &msaa;
        info.pDepthStencilState = &depthStencil;
        info.pColorBlendState = &blend;
        info.pDynamicState = &dynamic;
        info.layout = pipelineLayout;
        info.renderPass = renderPass;
        info.subpass = 0;

        VkResult result = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &graphicsPipeline);

        // Second pipeline: identical except it rasterizes triangle edges (debug toggle).
        // `info` still points at `raster`, so changing the mode there is all it takes.
        if (result == VK_SUCCESS && wireframeSupported) {
            raster.polygonMode = VK_POLYGON_MODE_LINE;
            result = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &wireframePipeline);
        }

        // Shader modules are only needed during pipeline creation
        vkDestroyShaderModule(device, frag, nullptr);
        vkDestroyShaderModule(device, vert, nullptr);

        if (result != VK_SUCCESS)
            throw std::runtime_error("Failed to create graphics pipeline");
    }

    // ---------- Framebuffers ----------

    void createFramebuffers() {
        swapchainFramebuffers.resize(swapchainImageViews.size());
        for (size_t i = 0; i < swapchainImageViews.size(); i++) {
            VkImageView views[2] = { swapchainImageViews[i], depthView };
            VkFramebufferCreateInfo info{};
            info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            info.renderPass = renderPass;
            info.attachmentCount = 2;
            info.pAttachments = views;
            info.width = swapchainExtent.width;
            info.height = swapchainExtent.height;
            info.layers = 1;
            if (vkCreateFramebuffer(device, &info, nullptr, &swapchainFramebuffers[i]) != VK_SUCCESS)
                throw std::runtime_error("Failed to create framebuffer");
        }
    }

    // ---------- Commands ----------

    void createCommandPool() {
        QueueFamilyIndices q = findQueueFamilies(physicalDevice);
        VkCommandPoolCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; // re-record each frame
        info.queueFamilyIndex = q.graphics.value();
        if (vkCreateCommandPool(device, &info, nullptr, &commandPool) != VK_SUCCESS)
            throw std::runtime_error("Failed to create command pool");
    }

    void createCommandBuffers() {
        commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        VkCommandBufferAllocateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        info.commandPool = commandPool;
        info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        info.commandBufferCount = (uint32_t)commandBuffers.size();
        if (vkAllocateCommandBuffers(device, &info, commandBuffers.data()) != VK_SUCCESS)
            throw std::runtime_error("Failed to allocate command buffers");
    }

    void recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex) {
        VkCommandBufferBeginInfo begin{};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        if (vkBeginCommandBuffer(cmd, &begin) != VK_SUCCESS)
            throw std::runtime_error("Failed to begin command buffer");

        // GPU timing: reset this slot's two queries (outside the render pass), then stamp the start
        vkCmdResetQueryPool(cmd, queryPool, currentFrame * 2, 2);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool, currentFrame * 2);

        VkClearValue clears[2]{};
        clears[0].color = {{ 0.06f, 0.08f, 0.12f, 1.0f }};
        clears[1].depthStencil = { 1.0f, 0 }; // depth starts at "farthest"

        VkRenderPassBeginInfo rp{};
        rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rp.renderPass = renderPass;
        rp.framebuffer = swapchainFramebuffers[imageIndex];
        rp.renderArea.offset = { 0, 0 };
        rp.renderArea.extent = swapchainExtent;
        rp.clearValueCount = 2;
        rp.pClearValues = clears;

        vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          (wireframe && wireframePipeline != VK_NULL_HANDLE) ? wireframePipeline : graphicsPipeline);
        drawCallsLastFrame = 0;

        VkViewport viewport{ 0.0f, 0.0f, (float)swapchainExtent.width, (float)swapchainExtent.height, 0.0f, 1.0f };
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        VkRect2D scissor{ { 0, 0 }, swapchainExtent };
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1,
                                &descriptorSets[currentFrame], 0, nullptr);

        // Binding 0 = mesh vertices, binding 1 = this frame's per-instance model matrices
        VkBuffer vbs[2] = { vertexBuffer, instanceBuffers[currentFrame] };
        VkDeviceSize offsets[2] = { 0, 0 };
        vkCmdBindVertexBuffers(cmd, 0, 2, vbs, offsets);
        vkCmdBindIndexBuffer(cmd, indexBuffer, 0, VK_INDEX_TYPE_UINT32);

        VkShaderStageFlags pushStages = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

        // Cubes. The model matrices come from the instance buffer (built from the ECS this frame);
        // the push constant now only carries material parameters, so it is pushed once.
        PushData cubePush{};
        cubePush.model = Mat4::identity();   // unused by the shaders since Step 3.4
        cubePush.params[0] = 0.0f;           // no checker
        cubePush.params[1] = 48.0f;          // shininess
        vkCmdPushConstants(cmd, pipelineLayout, pushStages, 0, sizeof(PushData), &cubePush);
        if (instanced) {
            // ONE draw call: the GPU runs the cube's 36 indices drawnCubes times, instance 0..N-1
            vkCmdDrawIndexed(cmd, cubeIndexCount, drawnCubes, 0, 0, 0);
            drawCallsLastFrame += 1;
        } else {
            // N draw calls of one instance each; firstInstance selects which matrix to read
            for (uint32_t i = 0; i < drawnCubes; i++)
                vkCmdDrawIndexed(cmd, cubeIndexCount, 1, 0, 0, i);
            drawCallsLastFrame += drawnCubes;
        }

        // Floor: static, checkered, less glossy. Its matrix is the last instance (index drawnCubes).
        PushData floorPush{};
        floorPush.model = Mat4::identity();
        floorPush.params[0] = 1.0f;  // checker on
        floorPush.params[1] = 16.0f;
        vkCmdPushConstants(cmd, pipelineLayout, pushStages, 0, sizeof(PushData), &floorPush);
        vkCmdDrawIndexed(cmd, floorIndexCount, 1, floorFirstIndex, 0, drawnCubes);
        drawCallsLastFrame += 1;

        // The debug overlay draws last, inside the same render pass, on top of the scene
        if (imguiEnabled) ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);

        vkCmdEndRenderPass(cmd);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, currentFrame * 2 + 1);
        if (vkEndCommandBuffer(cmd) != VK_SUCCESS)
            throw std::runtime_error("Failed to record command buffer");
    }

    // ---------- Memory & buffer helpers ----------

    uint32_t findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags props) {
        VkPhysicalDeviceMemoryProperties mem;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &mem);
        for (uint32_t i = 0; i < mem.memoryTypeCount; i++) {
            if ((typeBits & (1u << i)) && (mem.memoryTypes[i].propertyFlags & props) == props) return i;
        }
        throw std::runtime_error("No suitable memory type");
    }

    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props,
                      VkBuffer& buffer, VkDeviceMemory& memory) {
        VkBufferCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        info.size = size;
        info.usage = usage;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateBuffer(device, &info, nullptr, &buffer) != VK_SUCCESS)
            throw std::runtime_error("Failed to create buffer");

        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(device, buffer, &req);
        VkMemoryAllocateInfo alloc{};
        alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        alloc.allocationSize = req.size;
        alloc.memoryTypeIndex = findMemoryType(req.memoryTypeBits, props);
        if (vkAllocateMemory(device, &alloc, nullptr, &memory) != VK_SUCCESS)
            throw std::runtime_error("Failed to allocate buffer memory");
        vkBindBufferMemory(device, buffer, memory, 0);
    }

    // One-off command buffer for upload work (blocks until the GPU finishes: fine at startup)
    void copyBuffer(VkBuffer src, VkBuffer dst, VkDeviceSize size) {
        VkCommandBufferAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandPool = commandPool;
        ai.commandBufferCount = 1;
        VkCommandBuffer cmd;
        vkAllocateCommandBuffers(device, &ai, &cmd);

        VkCommandBufferBeginInfo begin{};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &begin);
        VkBufferCopy region{ 0, 0, size };
        vkCmdCopyBuffer(cmd, src, dst, 1, &region);
        vkEndCommandBuffer(cmd);

        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &cmd;
        vkQueueSubmit(graphicsQueue, 1, &submit, VK_NULL_HANDLE);
        vkQueueWaitIdle(graphicsQueue);
        vkFreeCommandBuffers(device, commandPool, 1, &cmd);
    }

    // CPU data -> staging buffer -> fast GPU-only buffer
    void createDeviceLocalBuffer(const void* data, VkDeviceSize size, VkBufferUsageFlags usage,
                                 VkBuffer& buffer, VkDeviceMemory& memory) {
        VkBuffer staging;
        VkDeviceMemory stagingMem;
        createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     staging, stagingMem);
        void* mapped;
        vkMapMemory(device, stagingMem, 0, size, 0, &mapped);
        memcpy(mapped, data, (size_t)size);
        vkUnmapMemory(device, stagingMem);

        createBuffer(size, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, buffer, memory);
        copyBuffer(staging, buffer, size);

        vkDestroyBuffer(device, staging, nullptr);
        vkFreeMemory(device, stagingMem, nullptr);
    }

    // ---------- Depth resources ----------

    void createDepthResources() {
        VkImageCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        info.imageType = VK_IMAGE_TYPE_2D;
        info.extent = { swapchainExtent.width, swapchainExtent.height, 1 };
        info.mipLevels = 1;
        info.arrayLayers = 1;
        info.format = depthFormat;
        info.tiling = VK_IMAGE_TILING_OPTIMAL;
        info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        info.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        info.samples = VK_SAMPLE_COUNT_1_BIT;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateImage(device, &info, nullptr, &depthImage) != VK_SUCCESS)
            throw std::runtime_error("Failed to create depth image");

        VkMemoryRequirements req;
        vkGetImageMemoryRequirements(device, depthImage, &req);
        VkMemoryAllocateInfo alloc{};
        alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        alloc.allocationSize = req.size;
        alloc.memoryTypeIndex = findMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (vkAllocateMemory(device, &alloc, nullptr, &depthMemory) != VK_SUCCESS)
            throw std::runtime_error("Failed to allocate depth memory");
        vkBindImageMemory(device, depthImage, depthMemory, 0);

        VkImageViewCreateInfo view{};
        view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view.image = depthImage;
        view.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view.format = depthFormat;
        view.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };
        if (vkCreateImageView(device, &view, nullptr, &depthView) != VK_SUCCESS)
            throw std::runtime_error("Failed to create depth view");
    }

    // ---------- Mesh ----------

    // Corners must be counter-clockwise when viewed from outside
    void addQuad(Vec3 a, Vec3 b, Vec3 c, Vec3 d, Vec3 normal, Vec3 color) {
        uint32_t base = (uint32_t)meshVertices.size();
        meshVertices.push_back({ a, color, normal, 0.0f, 0.0f });
        meshVertices.push_back({ b, color, normal, 1.0f, 0.0f });
        meshVertices.push_back({ c, color, normal, 1.0f, 1.0f });
        meshVertices.push_back({ d, color, normal, 0.0f, 1.0f });
        for (uint32_t i : { 0u, 1u, 2u, 0u, 2u, 3u }) meshIndices.push_back(base + i);
    }

    void createMeshBuffers() {
        const float h = 0.5f;
        // Cube: one flat color per face so rotation and occlusion are easy to see
        addQuad({-h,-h, h}, { h,-h, h}, { h, h, h}, {-h, h, h}, { 0, 0, 1}, {0.9f, 0.3f, 0.3f}); // front  red
        addQuad({ h,-h,-h}, {-h,-h,-h}, {-h, h,-h}, { h, h,-h}, { 0, 0,-1}, {0.3f, 0.9f, 0.9f}); // back   cyan
        addQuad({ h,-h, h}, { h,-h,-h}, { h, h,-h}, { h, h, h}, { 1, 0, 0}, {0.3f, 0.9f, 0.3f}); // right  green
        addQuad({-h,-h,-h}, {-h,-h, h}, {-h, h, h}, {-h, h,-h}, {-1, 0, 0}, {0.9f, 0.3f, 0.9f}); // left   magenta
        addQuad({-h, h, h}, { h, h, h}, { h, h,-h}, {-h, h,-h}, { 0, 1, 0}, {0.3f, 0.3f, 0.9f}); // top    blue
        addQuad({-h,-h,-h}, { h,-h,-h}, { h,-h, h}, {-h,-h, h}, { 0,-1, 0}, {0.9f, 0.9f, 0.3f}); // bottom yellow
        cubeIndexCount = (uint32_t)meshIndices.size(); // 36

        // Floor (same extents as Phase 1), wound counter-clockwise seen from above
        floorFirstIndex = (uint32_t)meshIndices.size();
        addQuad({-1,-0.5f,-0.5f}, { 1,-0.5f,-0.5f}, { 1,-0.5f,-5.0f}, {-1,-0.5f,-5.0f},
                { 0, 1, 0}, {0.6f, 0.6f, 0.6f});
        floorIndexCount = 6;

        createDeviceLocalBuffer(meshVertices.data(), sizeof(MeshVertex) * meshVertices.size(),
                                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertexBuffer, vertexMemory);
        createDeviceLocalBuffer(meshIndices.data(), sizeof(uint32_t) * meshIndices.size(),
                                VK_BUFFER_USAGE_INDEX_BUFFER_BIT, indexBuffer, indexMemory);
    }

    // ---------- Debug overlay (Dear ImGui) ----------

    void initImGui() {
        // ImGui needs its own descriptor pool (its font texture is a combined image sampler)
        VkDescriptorPoolSize poolSize{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 8 };
        VkDescriptorPoolCreateInfo pool{};
        pool.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        pool.maxSets = 8;
        pool.poolSizeCount = 1;
        pool.pPoolSizes = &poolSize;
        if (vkCreateDescriptorPool(device, &pool, nullptr, &imguiPool) != VK_SUCCESS)
            throw std::runtime_error("Failed to create ImGui descriptor pool");

        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        io.IniFilename = nullptr;            // don't write imgui.ini next to the exe
        ImGui::StyleColorsDark();

        ImGui_ImplGlfw_InitForVulkan(window, true);   // true = install GLFW input callbacks

        ImGui_ImplVulkan_InitInfo init{};
        init.ApiVersion = VK_API_VERSION_1_3;
        init.Instance = instance;
        init.PhysicalDevice = physicalDevice;
        init.Device = device;
        init.QueueFamily = findQueueFamilies(physicalDevice).graphics.value();
        init.Queue = graphicsQueue;
        init.DescriptorPool = imguiPool;
        init.RenderPass = renderPass;
        init.Subpass = 0;
        init.MinImageCount = 2;
        init.ImageCount = (uint32_t)swapchainImages.size();
        init.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
        if (!ImGui_ImplVulkan_Init(&init))
            throw std::runtime_error("Failed to initialize the ImGui Vulkan backend");
        imguiEnabled = true;
    }

    // Builds this frame's overlay. Runs once per frame, before command recording.
    void buildUI() {
        // Smoothed values so the numbers are readable instead of flickering
        const double a = 0.05;
        smoothFrameMs = smoothFrameMs == 0.0 ? frameDt * 1000.0 : smoothFrameMs + a * (frameDt * 1000.0 - smoothFrameMs);
        smoothCpuMs   += a * (lastCpuMs - smoothCpuMs);
        smoothGpuMs   += a * (lastGpuMs - smoothGpuMs);
        frameHistory[historyPos] = (float)(frameDt * 1000.0);
        historyPos = (historyPos + 1) % HISTORY;

        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowBgAlpha(0.88f);
        ImGui::Begin("Toy Engine", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

        if (ImGui::CollapsingHeader("Stats", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Text("%.0f FPS   (%.2f ms/frame)", smoothFrameMs > 0 ? 1000.0 / smoothFrameMs : 0.0, smoothFrameMs);
            ImGui::PlotLines("##frames", frameHistory, HISTORY, historyPos, nullptr, 0.0f, 20.0f, ImVec2(300, 50));
            ImGui::Text("Entities: %zu cubes", world.size());
            ImGui::Text("Draw calls: %u   Triangles: %u", drawCallsLastFrame,
                        drawnCubes * (cubeIndexCount / 3) + floorIndexCount / 3);
            ImGui::Separator();
            ImGui::Text("CPU frame: %.3f ms   GPU: %.3f ms", smoothCpuMs, smoothGpuMs);
            ImGui::Text("  spinSystem:      %.3f ms", msSpin);
            ImGui::Text("  buildDrawList:   %.3f ms", msDrawList);
            ImGui::Text("  instance upload: %.3f ms", msUpload);
        }

        if (ImGui::CollapsingHeader("Scene", ImGuiTreeNodeFlags_DefaultOpen)) {
            int n = cubeCount;
            if (ImGui::SliderInt("Cubes", &n, 1, MAX_CUBES, "%d", ImGuiSliderFlags_Logarithmic) && n != cubeCount) {
                cubeCount = n;
                spawnScene();
            }
            ImGui::Checkbox("Instanced drawing", &instanced);
            ImGui::Checkbox("Pause simulation", &simPaused);
        }

        if (ImGui::CollapsingHeader("Debug views", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (!wireframeSupported) ImGui::BeginDisabled();
            ImGui::Checkbox("Wireframe", &wireframe);
            if (!wireframeSupported) ImGui::EndDisabled();
            ImGui::Checkbox("Lighting", &lightingOn);
            ImGui::Checkbox("Freeze camera", &camFrozen);
        }

        ImGui::End();
        ImGui::Render();
    }

    // ---------- Instances & the ECS world ----------

    // One host-visible buffer per frame in flight, big enough for MAX_INSTANCES model matrices.
    // Host-visible (not device-local) because we rewrite it every frame.
    void createInstanceBuffers() {
        drawList.reserve(MAX_INSTANCES);
        instanceBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        instanceMemory.resize(MAX_FRAMES_IN_FLIGHT);
        instanceMapped.resize(MAX_FRAMES_IN_FLIGHT);
        VkDeviceSize size = sizeof(DrawItem) * (VkDeviceSize)MAX_INSTANCES;
        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            createBuffer(size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         instanceBuffers[i], instanceMemory[i]);
            vkMapMemory(device, instanceMemory[i], 0, size, 0, &instanceMapped[i]);
        }
    }

    // (Re)build the world: one entity per cube. Components: where it is, how it spins, what to draw.
    void spawnScene() {
        world = Registry();
        if (cubeCount > MAX_CUBES) cubeCount = MAX_CUBES;
        int n = cubeCount;
        for (int i = 0; i < n; i++) {
            Entity e = world.create();
            Transform t;
            if (benchMode || n > 1) {   // a grid of cubes (benchmark, or the slider in the overlay)
                float x, y, z, s;
                benchCubeParams(i, n, x, y, z, s);
                t.position = { x, y, z };
                t.scale = s;
                t.yaw = i * 0.1f;
            } else {
                t.position = { 0.0f, 0.0f, -1.5f };
            }
            world.add<Transform>(e, t);
            world.add<Spin>(e, Spin{ 1.0f });
            world.add<MeshRenderer>(e, MeshRenderer{ 0, 0 });   // mesh 0 = the cube
        }
        lastSimTime = glfwGetTime();
    }

    // The CPU side of a frame: run the systems, build the draw list, upload it.
    void simulateAndUpload(uint32_t frame) {
        double now = glfwGetTime();
        float dt = (float)std::min(now - lastSimTime, 0.1);   // clamp so a stall can't make cubes jump
        lastSimTime = now;
        if (simPaused) dt = 0.0f;

        double t0 = glfwGetTime();
        spinSystem(world, dt);
        double t1 = glfwGetTime();
        buildDrawList(world, drawList);
        drawnCubes = (uint32_t)drawList.size();
        drawList.push_back(DrawItem{ Mat4::identity(), 1, 0 });   // the floor (not an entity yet)
        double t2 = glfwGetTime();

        memcpy(instanceMapped[frame], drawList.data(), drawList.size() * sizeof(DrawItem));
        double t3 = glfwGetTime();
        msSpin = (t1 - t0) * 1000.0; msDrawList = (t2 - t1) * 1000.0; msUpload = (t3 - t2) * 1000.0;
    }

    // ---------- Uniforms ----------

    void createDescriptorSetLayout() {
        VkDescriptorSetLayoutBinding b{};
        b.binding = 0;
        b.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        b.descriptorCount = 1;
        b.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        info.bindingCount = 1;
        info.pBindings = &b;
        if (vkCreateDescriptorSetLayout(device, &info, nullptr, &descriptorSetLayout) != VK_SUCCESS)
            throw std::runtime_error("Failed to create descriptor set layout");
    }

    void createUniformBuffers() {
        uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        uniformMemory.resize(MAX_FRAMES_IN_FLIGHT);
        uniformMapped.resize(MAX_FRAMES_IN_FLIGHT);
        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            createBuffer(sizeof(SceneUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         uniformBuffers[i], uniformMemory[i]);
            // Stay mapped for the whole run: we rewrite it every frame
            vkMapMemory(device, uniformMemory[i], 0, sizeof(SceneUBO), 0, &uniformMapped[i]);
        }
    }

    void createDescriptorSets() {
        VkDescriptorPoolSize size{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, (uint32_t)MAX_FRAMES_IN_FLIGHT };
        VkDescriptorPoolCreateInfo pool{};
        pool.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool.poolSizeCount = 1;
        pool.pPoolSizes = &size;
        pool.maxSets = (uint32_t)MAX_FRAMES_IN_FLIGHT;
        if (vkCreateDescriptorPool(device, &pool, nullptr, &descriptorPool) != VK_SUCCESS)
            throw std::runtime_error("Failed to create descriptor pool");

        std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, descriptorSetLayout);
        VkDescriptorSetAllocateInfo alloc{};
        alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc.descriptorPool = descriptorPool;
        alloc.descriptorSetCount = (uint32_t)MAX_FRAMES_IN_FLIGHT;
        alloc.pSetLayouts = layouts.data();
        descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
        if (vkAllocateDescriptorSets(device, &alloc, descriptorSets.data()) != VK_SUCCESS)
            throw std::runtime_error("Failed to allocate descriptor sets");

        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            VkDescriptorBufferInfo bi{ uniformBuffers[i], 0, sizeof(SceneUBO) };
            VkWriteDescriptorSet w{};
            w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet = descriptorSets[i];
            w.dstBinding = 0;
            w.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            w.descriptorCount = 1;
            w.pBufferInfo = &bi;
            vkUpdateDescriptorSets(device, 1, &w, 0, nullptr);
        }
    }

    void updateUniformBuffer(uint32_t frame) {
        float t = (float)glfwGetTime();
        Vec3 eye = { 0, 0.8f, 1.0f };

        // Gentle camera sway around the scene (interactive mode only; benchmarks keep a fixed camera).
        // camTime stops advancing when the camera is frozen, so unfreezing resumes smoothly.
        if (!benchMode) {
            if (!camFrozen) camTime += (float)frameDt;
            float angle = 0.35f * std::sin(camTime * 0.5f);
            const Vec3 target = { 0, -0.3f, -3.0f };
            float dx = eye.x - target.x, dz = eye.z - target.z;
            eye.x = target.x + dx * std::cos(angle) + dz * std::sin(angle);
            eye.z = target.z - dx * std::sin(angle) + dz * std::cos(angle);
        }

        Mat4 view = Mat4::lookAt(eye, { 0, -0.3f, -3.0f }, { 0, 1, 0 });
        Mat4 proj = Mat4::perspective(3.14159265f / 4.0f,
                                      (float)swapchainExtent.width / swapchainExtent.height, 0.1f, 100.0f);

        // OpenGL-style clip space -> Vulkan clip space:
        //   Y is flipped (Vulkan's Y points down), and depth is remapped from [-1,1] to [0,1]
        Mat4 fix = Mat4::identity();
        fix.m[1][1] = -1.0f;
        fix.m[2][2] = 0.5f;
        fix.m[3][2] = 0.5f;

        SceneUBO ubo{};
        ubo.viewProj = fix * proj * view;
        ubo.cameraPos[0] = eye.x; ubo.cameraPos[1] = eye.y; ubo.cameraPos[2] = eye.z;

        // Light 0: warm white, upper right, in front of the scene
        ubo.lights[0] = { { 1.5f, 1.5f, 0.0f, 1.0f }, { 1.0f, 0.9f, 0.75f, 4.0f } };
        // Light 1: cool blue fill from the far left
        ubo.lights[1] = { { -1.8f, 1.0f, -2.5f, 1.0f }, { 0.4f, 0.6f, 1.0f, 3.0f } };
        // Light 2: orbits the cube, so you can watch the lighting respond to movement
        ubo.lights[2] = { { std::sin(t * 1.2f) * 1.3f, 0.7f, -1.5f + std::cos(t * 1.2f) * 1.3f, 1.0f },
                          { 1.0f, 0.35f, 0.3f, 3.0f } };
        ubo.info[0] = 3;
        ubo.info[1] = lightingOn ? 1 : 0;

        memcpy(uniformMapped[frame], &ubo, sizeof(ubo));
    }

    // ---------- Timing ----------

    void createQueryPool() {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(physicalDevice, &props);
        timestampPeriodNs = props.limits.timestampPeriod; // nanoseconds per timestamp tick

        VkQueryPoolCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        info.queryType = VK_QUERY_TYPE_TIMESTAMP;
        info.queryCount = MAX_FRAMES_IN_FLIGHT * 2; // start + end per frame slot
        if (vkCreateQueryPool(device, &info, nullptr, &queryPool) != VK_SUCCESS)
            throw std::runtime_error("Failed to create query pool");
    }

    // ---------- Synchronization ----------

    void createSyncObjects() {
        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphores.resize(swapchainImages.size());
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

        VkSemaphoreCreateInfo semInfo{};
        semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT; // start signaled so frame 1 doesn't wait forever

        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            if (vkCreateSemaphore(device, &semInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
                vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS)
                throw std::runtime_error("Failed to create sync objects");
        }
        for (size_t i = 0; i < swapchainImages.size(); i++) {
            if (vkCreateSemaphore(device, &semInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS)
                throw std::runtime_error("Failed to create render-finished semaphore");
        }
    }

    // ---------- The frame ----------

    void drawFrame() {
        double frameStart = glfwGetTime();
        frameDt = frameStart - prevFrameTime;
        prevFrameTime = frameStart;

        // 1. Wait until the GPU is done with the previous use of this frame slot
        vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

        // GPU time of this slot's previous frame (it has finished, since the fence is signaled)
        if (framesSubmitted >= (uint64_t)MAX_FRAMES_IN_FLIGHT) {
            uint64_t ts[2];
            VkResult qr = vkGetQueryPoolResults(device, queryPool, currentFrame * 2, 2, sizeof(ts), ts,
                                                sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
            if (qr == VK_SUCCESS) {
                lastGpuMs = (double)(ts[1] - ts[0]) * timestampPeriodNs * 1e-6;
                if (benchMeasuring) { benchGpuMsSum += lastGpuMs; benchGpuSamples++; }
            }
        }

        // 2. Ask the swapchain which image we may draw into
        uint32_t imageIndex;
        VkResult acq = vkAcquireNextImageKHR(device, swapchain, UINT64_MAX,
                                             imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);
        if (acq == VK_ERROR_OUT_OF_DATE_KHR) return; // window resize/minimize: not handled yet (window is fixed-size)
        if (acq != VK_SUCCESS && acq != VK_SUBOPTIMAL_KHR)
            throw std::runtime_error("Failed to acquire swapchain image");

        // Only reset the fence once we know we'll submit work (avoids a deadlock on early return)
        vkResetFences(device, 1, &inFlightFences[currentFrame]);

        // The fence wait above guarantees the GPU is done with this slot's uniform buffer
        double cpuStart = glfwGetTime();
        simulateAndUpload(currentFrame);
        lastSimMs = (glfwGetTime() - cpuStart) * 1000.0;
        updateUniformBuffer(currentFrame);
        if (imguiEnabled) buildUI();

        // 3. Record this frame's commands
        VkCommandBuffer cmd = commandBuffers[currentFrame];
        vkResetCommandBuffer(cmd, 0);
        recordCommandBuffer(cmd, imageIndex);

        // 4. Submit: wait for the image to be available, signal when rendering is done
        VkSemaphore waitSems[] = { imageAvailableSemaphores[currentFrame] };
        VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
        VkSemaphore signalSems[] = { renderFinishedSemaphores[imageIndex] };

        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.waitSemaphoreCount = 1;
        submit.pWaitSemaphores = waitSems;
        submit.pWaitDstStageMask = waitStages;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &cmd;
        submit.signalSemaphoreCount = 1;
        submit.pSignalSemaphores = signalSems;
        if (vkQueueSubmit(graphicsQueue, 1, &submit, inFlightFences[currentFrame]) != VK_SUCCESS)
            throw std::runtime_error("Failed to submit draw command buffer");
        framesSubmitted++;
        lastCpuMs = (glfwGetTime() - cpuStart) * 1000.0; // CPU cost of recording + submitting this frame

        // 5. Present: show the image once rendering has finished
        VkPresentInfoKHR present{};
        present.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        present.waitSemaphoreCount = 1;
        present.pWaitSemaphores = signalSems;
        present.swapchainCount = 1;
        present.pSwapchains = &swapchain;
        present.pImageIndices = &imageIndex;
        vkQueuePresentKHR(presentQueue, &present);

        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;

        frameCount++;
        double now = glfwGetTime();
        if (now - lastTitleTime >= 1.0) {
            std::string t = "Toy Engine (Vulkan) - " + std::to_string(frameCount) + " FPS";
            glfwSetWindowTitle(window, t.c_str());
            frameCount = 0;
            lastTitleTime = now;
        }

        if (benchMode) {
            double benchNow = glfwGetTime();
            if (!benchMeasuring) {
                if (benchNow - benchStageStart >= BENCH_WARMUP_SEC) {
                    benchMeasuring = true; benchMeasureStart = benchNow; benchFrames = 0;
                    benchCpuMsSum = 0.0; benchGpuMsSum = 0.0; benchGpuSamples = 0; benchSimMsSum = 0.0;
                }
            } else {
                benchFrames++;
                benchCpuMsSum += lastCpuMs;
                benchSimMsSum += lastSimMs;
                if (benchNow - benchMeasureStart >= BENCH_MEASURE_SEC) {
                    double ms = (benchNow - benchMeasureStart) / benchFrames * 1000.0;
                    printf("%s,%d,%d,%.3f,%.1f,%.3f,%.3f,%.3f\n", instanced ? "instanced" : "per-draw",
                           cubeCount, cubeCount * 12 + 2, ms, 1000.0 / ms,
                           benchSimMsSum / benchFrames, benchCpuMsSum / benchFrames,
                           benchGpuSamples ? benchGpuMsSum / benchGpuSamples : 0.0);
                    fflush(stdout);
                    // Stages 0..5 = per-draw at each size, stages 6..11 = instanced at each size
                    if (++benchStage >= 2 * VK_BENCH_NUM_COUNTS) {
                        glfwSetWindowShouldClose(window, GLFW_TRUE);
                    } else {
                        cubeCount = VK_BENCH_COUNTS[benchStage % VK_BENCH_NUM_COUNTS];
                        instanced = benchStage >= VK_BENCH_NUM_COUNTS;
                        spawnScene();
                        benchMeasuring = false;
                        benchStageStart = benchNow;
                    }
                }
            }
        }
    }
};

int main(int argc, char** argv) {
    try {
        App app;
        app.benchMode = (argc > 1 && std::strcmp(argv[1], "--bench") == 0);
        app.run();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
