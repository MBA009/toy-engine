#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <cstdlib>
#include <fstream>
#include <iterator>
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
#include "camera.h"
#include "picking.h"
#include "mesh_data.h"
#include "gltf_loader.h"
#include "scene_io.h"
#include "file_watch.h"
#include "navmesh.h"
#include "ai_systems.h"
#include "demo_level.h"

#include "imgui.h"
#include "ImGuizmo.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_vulkan.h"

// Benchmark sizes for this renderer. Instancing makes 100,000 cubes practical, so the list
// goes one step past the shared BENCH_COUNTS (which the software renderer also uses).
inline constexpr int VK_BENCH_COUNTS[] = { 1, 10, 100, 1000, 10000, 100000 };
inline constexpr int VK_BENCH_NUM_COUNTS = (int)(sizeof(VK_BENCH_COUNTS) / sizeof(VK_BENCH_COUNTS[0]));
inline constexpr int MAX_CUBES = 100000;
inline constexpr int MAX_INSTANCES = MAX_CUBES + 1;   // + the floor

const uint32_t WIDTH = 800;       // benchmark window: fixed so results stay comparable with Phase 2
const uint32_t HEIGHT = 600;
const uint32_t EDITOR_WIDTH = 1280;   // interactive window: room for the editor panels
const uint32_t EDITOR_HEIGHT = 720;
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
    float params[4];   // x = checker on/off, y = shininess, z = alpha cutoff (0 = opaque)
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
    VkPipeline noCullPipeline = VK_NULL_HANDLE;       // same, but back faces are drawn too (glTF "double sided" materials)
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

    // Every mesh lives in the one big vertex buffer and one big index buffer; this table says where.
    // 0 = the built-in cube, 1 = the floor, 2+ = meshes imported from glTF files.
    struct MeshInfo {
        std::string name;
        uint32_t firstIndex = 0, indexCount = 0;
        int32_t vertexOffset = 0;      // added to every index when drawing, so imported meshes keep local indices
        Aabb bounds;                   // in the mesh's own space (used for picking)
        // The mesh is drawn as one draw call per submesh (= per material). Indices are global positions in the
        // shared index buffer, like firstIndex above. material = index into materialSets.
        struct Sub { uint32_t firstIndex, indexCount, material; };
        std::vector<Sub> subs;
        bool retired = false;          // replaced by a hot reload: still in the buffers, but no longer offered anywhere
        uint32_t textureCount = 0;     // for the Assets panel
    };
    std::vector<MeshInfo> meshes;

    // This frame's draw list grouped by mesh: instances [first, first+count) all use mesh `mesh`
    struct DrawRange { uint32_t mesh, first, count; };
    std::vector<DrawRange> drawRanges;
    uint32_t drawnTriangles = 0;

    // Assets panel
    std::vector<std::string> assetFiles;                          // .gltf / .glb files found under assets/
    double lastAssetScan = -100.0;
    std::string assetStatus;                                      // last import result, shown in the panel
    std::vector<std::pair<std::string, uint32_t>> loadedAssets;   // file path -> mesh id, so a file imports once
    std::vector<std::vector<std::string>> loadedAssetDeps;        // same index: the files that import read (watched for changes)
    VkBuffer vertexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory vertexMemory = VK_NULL_HANDLE;
    VkBuffer indexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory indexMemory = VK_NULL_HANDLE;

    // Textures & materials. A material is a descriptor set (set 1) holding one combined image sampler.
    // Material 0 is the default: a 1x1 white texture, so untextured meshes multiply by 1 and look as before.
    struct GpuTexture { uint32_t mipLevels = 1; VkImage image = VK_NULL_HANDLE; VkDeviceMemory memory = VK_NULL_HANDLE; VkImageView view = VK_NULL_HANDLE; };
    static constexpr uint32_t MAX_MATERIALS = 512;
    std::vector<GpuTexture> textures;
    VkSampler textureSampler = VK_NULL_HANDLE;
    struct MaterialProps { bool doubleSided = false; float alphaCutoff = 0.0f; };
    std::vector<MaterialProps> materialProps;    // indexed like materialSets
    std::vector<uint32_t> materialTexture;       // materialSets[i] shows textures[materialTexture[i]] (to rewrite on sampler change)
    int filterMode = 2;                          // 0 = nearest, 1 = bilinear (base level only), 2 = trilinear (mipmaps)
    int anisotropy = 8;                          // requested max anisotropy (clamped to the GPU's limit)
    float maxAnisotropy = 1.0f;                  // the GPU's limit; 1 = anisotropic filtering unsupported
    VkDescriptorSetLayout materialSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool materialPool = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> materialSets;

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
    // Camera: free-fly (hold the right mouse button), or the old automatic sway as a demo mode
    FlyCamera cam;
    bool camAutoSway = false;
    float camTime = 0.0f;           // drives the auto-sway
    bool looking = false;           // right mouse button held: mouse-look + WASD/QE active

    // Selection / picking
    Entity selected = NULL_ENTITY;
    float selectedDistance = 0.0f;                 // distance from the camera to the last picked hit
    Mat4 pickInvViewProj = Mat4::identity();       // from the last rendered frame
    bool havePickMatrix = false;
    bool prevLeftDown = false, prevDeleteDown = false;
    bool prevHotkey[3] = {};                       // W / E / R, for edge detection

    // Transform gizmo (ImGuizmo)
    ImGuizmo::OPERATION gizmoOp = ImGuizmo::TRANSLATE;
    bool gizmoSnap = false;
    bool gizmoOver = false;                        // the mouse is over a gizmo handle
    bool gizmoActive = false;                      // a gizmo handle is being dragged
    Mat4 lastView = Mat4::identity(), lastProj = Mat4::identity();   // OpenGL-style, before the Vulkan fix-up
    float simTime = 0.0f;                          // time that only advances while the simulation runs
    bool demoScene = false;                        // interactive default: a few named cubes
    // Inspector state
    char nameBuf[128] = {};
    bool nameEditing = false;                      // the name text box has focus: don't overwrite what's being typed

    // Lights, editable from the Inspector (these feed the scene uniform buffer every frame)
    struct EditorLight { float pos[3]; float color[3]; float intensity; };
    EditorLight lightCfg[3] = {
        { {  1.5f, 1.5f,  0.0f }, { 1.0f, 0.90f, 0.75f }, 4.0f },   // warm white, upper right
        { { -1.8f, 1.0f, -2.5f }, { 0.4f, 0.60f, 1.00f }, 3.0f },   // cool blue fill, far left
        { {  0.0f, 0.7f, -1.5f }, { 1.0f, 0.35f, 0.30f }, 3.0f },   // orbits around its position
    };
    bool light2Orbit = true;

    bool prevSaveDown = false;
    char scenePath[260] = "scenes/scene.json";     // file the Scene panel saves to / loads from
    std::string sceneStatus;                       // result of the last save / load, shown in the Scene panel
    // Navigation mesh (Step 7.1)
    NavMesh navMesh;
    NavParams navParams;
    bool showNavmesh = true;
    std::string navStatus = "not built yet";
    // Path query (Step 7.2): two points on the navmesh and the route between them
    enum class PathTool { Off, SetStart, SetEnd, SendAgent, AddPatrol, Place, MovePlayer };
    PathTool pathTool = PathTool::Off;      // which point the next click on the navmesh places
    float pathStart[3] = { 0, 0, 0 }, pathEnd[3] = { 0, 0, 0 };
    bool haveStart = false, haveEnd = false;
    NavPath navPath;
    std::string pathStatus;
    bool showAgentPaths = true;
    BehaviorTree combatTree = makeCombatTree();   // shared by every agent that has an AIBrain
    bool runAI = true;
    GameState game;               // win/lose state of the current run (only meaningful when the scene has a player and a goal)
    bool showAILabels = true;   // state text + health bar above each agent with a brain
    PathFn navPathFn = [this](const Vec3& from, const Vec3& to, std::vector<Vec3>& corners) {
        if (!navMesh.valid()) return false;
        const float f[3] = { from.x, from.y, from.z }, t[3] = { to.x, to.y, to.z };
        NavPath p; std::string err;
        if (!navMesh.findPath(f, t, p, err) || p.count() == 0) return false;
        corners.clear();
        for (size_t i = 0; i < p.count(); i++) corners.push_back({ p.point(i)[0], p.point(i)[1], p.point(i)[2] });
        return true;
    };
    int agentSerial = 0;
    // Hot reload (Step 6.4): poll file timestamps; editor mode only
    bool hotReload = true;
    FileWatcher shaderWatch, assetWatch;
    double lastHotPoll = 0.0;
    int shaderReloads = 0;
    std::string shaderStatus = "watching shaders/*.vert, *.frag";
    int cubeSerial = 0;                            // numbers the cubes added from the Hierarchy panel
    Entity lastScrolledTo = NULL_ENTITY;           // so the entity list scrolls only when the selection changes
    double lastMouseX = 0.0, lastMouseY = 0.0;
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
        window = benchMode ? glfwCreateWindow(WIDTH, HEIGHT, "Toy Engine (Vulkan)", nullptr, nullptr)
                           : glfwCreateWindow(EDITOR_WIDTH, EDITOR_HEIGHT, "Toy Engine (Vulkan)", nullptr, nullptr);
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
        createTextureResources();
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
        demoScene = !benchMode;
        spawnScene();
        // The editor reopens the last saved scene (the demo scene above stays if there is no file or it is broken;
        // the Scene panel then shows why). Benchmarks always start from their own generated scene.
        if (imguiEnabled && std::filesystem::exists(scenePath)) loadScene(scenePath);
        prevFrameTime = glfwGetTime();   // so the first frame's dt isn't "time since startup"
        benchStageStart = glfwGetTime();

        if (imguiEnabled) {
            for (const char* f : { "shaders/mesh.vert", "shaders/mesh.frag", "shaders/mesh.vert.spv", "shaders/mesh.frag.spv" })
                shaderWatch.watch(f);
        }
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            pollHotReload();
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
        vkDestroyPipeline(device, noCullPipeline, nullptr);
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

        vkDestroyDescriptorPool(device, materialPool, nullptr);      // frees every material's descriptor set
        vkDestroyDescriptorSetLayout(device, materialSetLayout, nullptr);
        vkDestroySampler(device, textureSampler, nullptr);
        for (const GpuTexture& t : textures) {
            vkDestroyImageView(device, t.view, nullptr);
            vkDestroyImage(device, t.image, nullptr);
            vkFreeMemory(device, t.memory, nullptr);
        }

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
        if (supported.samplerAnisotropy) {         // anisotropic texture filtering (Step 6.3)
            features.samplerAnisotropy = VK_TRUE;
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(physicalDevice, &props);
            maxAnisotropy = std::max(1.0f, props.limits.maxSamplerAnisotropy);
        }
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

    // Builds (or REBUILDS, for shader hot reload) the fill and wireframe pipelines from shaders/*.spv.
    // On failure it returns false with the reason in `error` and leaves any existing pipelines untouched,
    // so a broken shader edit can never take the renderer down.
    bool createGraphicsPipeline(std::string& error) {
        VkShaderModule vert = VK_NULL_HANDLE, frag = VK_NULL_HANDLE;
        try {
            vert = createShaderModule(readFile("shaders/mesh.vert.spv"));
            frag = createShaderModule(readFile("shaders/mesh.frag.spv"));
        } catch (const std::exception& ex) {
            if (vert != VK_NULL_HANDLE) vkDestroyShaderModule(device, vert, nullptr);
            error = ex.what();
            return false;
        }

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
        VkVertexInputAttributeDescription attrs[9] = {
            { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, (uint32_t)offsetof(MeshVertex, pos) },
            { 1, 0, VK_FORMAT_R32G32B32_SFLOAT, (uint32_t)offsetof(MeshVertex, color) },
            { 2, 0, VK_FORMAT_R32G32B32_SFLOAT, (uint32_t)offsetof(MeshVertex, normal) },
            { 3, 0, VK_FORMAT_R32G32_SFLOAT,    (uint32_t)offsetof(MeshVertex, u) },
            { 4, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 0 },
            { 5, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 16 },
            { 6, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 32 },
            { 7, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 48 },
            { 8, 1, VK_FORMAT_R32_UINT, (uint32_t)offsetof(DrawItem, materialId) },   // flags (selected, ...)
        };
        VkPipelineVertexInputStateCreateInfo vertexInput{};
        vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInput.vertexBindingDescriptionCount = 2;
        vertexInput.pVertexBindingDescriptions = bindings;
        vertexInput.vertexAttributeDescriptionCount = 9;
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
        VkDescriptorSetLayout setLayouts[2] = { descriptorSetLayout, materialSetLayout };   // set 0 = scene, set 1 = material
        layoutInfo.setLayoutCount = 2;
        layoutInfo.pSetLayouts = setLayouts;
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges = &push;
        if (pipelineLayout == VK_NULL_HANDLE &&           // created once; a rebuild reuses it
            vkCreatePipelineLayout(device, &layoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS)
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

        VkPipeline newFill = VK_NULL_HANDLE, newWire = VK_NULL_HANDLE, newNoCull = VK_NULL_HANDLE;
        VkResult result = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &newFill);

        // Same again without back-face culling, for double-sided materials
        if (result == VK_SUCCESS) {
            const VkCullModeFlags cull = raster.cullMode;
            raster.cullMode = VK_CULL_MODE_NONE;
            result = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &newNoCull);
            raster.cullMode = cull;
        }

        // Second pipeline: identical except it rasterizes triangle edges (debug toggle).
        // `info` still points at `raster`, so changing the mode there is all it takes.
        if (result == VK_SUCCESS && wireframeSupported) {
            raster.polygonMode = VK_POLYGON_MODE_LINE;
            result = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &newWire);
        }

        // Shader modules are only needed during pipeline creation
        vkDestroyShaderModule(device, frag, nullptr);
        vkDestroyShaderModule(device, vert, nullptr);

        if (result != VK_SUCCESS) {
            if (newFill != VK_NULL_HANDLE) vkDestroyPipeline(device, newFill, nullptr);
            if (newWire != VK_NULL_HANDLE) vkDestroyPipeline(device, newWire, nullptr);
            if (newNoCull != VK_NULL_HANDLE) vkDestroyPipeline(device, newNoCull, nullptr);
            error = "vkCreateGraphicsPipelines failed (" + std::to_string((int)result) + "); do the new shaders still match the vertex layout and descriptor sets?";
            return false;
        }

        // Swap in the new pipelines. The GPU may still be drawing with the old ones, so wait for it first.
        if (graphicsPipeline != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(device);
            vkDestroyPipeline(device, graphicsPipeline, nullptr);
            if (wireframePipeline != VK_NULL_HANDLE) vkDestroyPipeline(device, wireframePipeline, nullptr);
            vkDestroyPipeline(device, noCullPipeline, nullptr);
        }
        noCullPipeline = newNoCull;
        graphicsPipeline = newFill;
        wireframePipeline = newWire;
        return true;
    }

    void createGraphicsPipeline() {
        std::string error;
        if (!createGraphicsPipeline(error)) throw std::runtime_error("Failed to create graphics pipeline: " + error);
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
        // One range per mesh. Instanced: ONE draw call per mesh, the GPU repeats the mesh's indices for
        // instances first..first+count-1. Otherwise: one draw call per object (firstInstance picks its matrix).
        // A mesh with several materials is several draw calls (one per submesh), each with its material bound.
        uint32_t boundMaterial = ~0u;
        const bool wireActive = wireframe && wireframePipeline != VK_NULL_HANDLE;
        VkPipeline boundPipeline = wireActive ? wireframePipeline : graphicsPipeline;
        float boundCutoff = 0.0f;
        auto bindMaterial = [&](uint32_t mat) {
            if (mat == boundMaterial) return;
            const MaterialProps& mp = materialProps[mat];
            if (!wireActive) {      // double-sided materials switch to the no-culling pipeline
                VkPipeline want = mp.doubleSided ? noCullPipeline : graphicsPipeline;
                if (want != boundPipeline) { vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, want); boundPipeline = want; }
            }
            if (mp.alphaCutoff != boundCutoff) {   // params.z: the fragment shader discards pixels below this alpha
                vkCmdPushConstants(cmd, pipelineLayout, pushStages, (uint32_t)(offsetof(PushData, params) + 2 * sizeof(float)),
                                   sizeof(float), &mp.alphaCutoff);
                boundCutoff = mp.alphaCutoff;
            }
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 1, 1,
                                    &materialSets[mat], 0, nullptr);
            boundMaterial = mat;
        };
        for (const DrawRange& r : drawRanges) {
            const MeshInfo& m = meshes[r.mesh];
            for (const MeshInfo::Sub& sub : m.subs) {
                bindMaterial(sub.material);
                if (instanced) {
                    vkCmdDrawIndexed(cmd, sub.indexCount, r.count, sub.firstIndex, m.vertexOffset, r.first);
                    drawCallsLastFrame += 1;
                } else {
                    for (uint32_t k = 0; k < r.count; k++)
                        vkCmdDrawIndexed(cmd, sub.indexCount, 1, sub.firstIndex, m.vertexOffset, r.first + k);
                    drawCallsLastFrame += r.count;
                }
            }
        }
        bindMaterial(0);

        // Floor: static, checkered, less glossy. Its matrix is the last instance (index drawnCubes).
        PushData floorPush{};
        floorPush.model = Mat4::identity();
        floorPush.params[0] = 1.0f;  // checker on
        floorPush.params[1] = 16.0f;
        vkCmdPushConstants(cmd, pipelineLayout, pushStages, 0, sizeof(PushData), &floorPush);
        const MeshInfo& floorMesh = meshes[1];
        vkCmdDrawIndexed(cmd, floorMesh.indexCount, 1, floorMesh.firstIndex, floorMesh.vertexOffset, drawnCubes);
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
        MeshInfo cube;
        cube.name = "Cube";
        cube.indexCount = (uint32_t)meshIndices.size();   // 36; these built-in meshes use global indices, vertex offset 0
        cube.subs.push_back({ 0, cube.indexCount, 0 });
        meshes.push_back(cube);

        // Floor (same extents as Phase 1), wound counter-clockwise seen from above
        MeshInfo floorInfo;
        floorInfo.name = "Floor";
        floorInfo.firstIndex = (uint32_t)meshIndices.size();
        addQuad({-3,-0.5f, 1.0f}, { 3,-0.5f, 1.0f}, { 3,-0.5f,-7.0f}, {-3,-0.5f,-7.0f},
                { 0, 1, 0}, {0.6f, 0.6f, 0.6f});
        floorInfo.indexCount = 6;
        floorInfo.subs.push_back({ floorInfo.firstIndex, 6, 0 });
        floorInfo.bounds.min = { -3.0f, -0.5f, -7.0f };
        floorInfo.bounds.max = {  3.0f, -0.5f,  1.0f };
        meshes.push_back(floorInfo);

        uploadMeshBuffers();
    }

    // (Re)creates the GPU vertex and index buffers from the CPU arrays. Called at startup and whenever a mesh
    // is imported; the old buffers are only destroyed after the GPU has finished with them.
    void uploadMeshBuffers() {
        if (vertexBuffer != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(device);
            vkDestroyBuffer(device, indexBuffer, nullptr);
            vkFreeMemory(device, indexMemory, nullptr);
            vkDestroyBuffer(device, vertexBuffer, nullptr);
            vkFreeMemory(device, vertexMemory, nullptr);
            vertexBuffer = indexBuffer = VK_NULL_HANDLE;
        }
        createDeviceLocalBuffer(meshVertices.data(), sizeof(MeshVertex) * meshVertices.size(),
                                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertexBuffer, vertexMemory);
        createDeviceLocalBuffer(meshIndices.data(), sizeof(uint32_t) * meshIndices.size(),
                                VK_BUFFER_USAGE_INDEX_BUFFER_BIT, indexBuffer, indexMemory);
    }

    // Appends an imported mesh to the shared buffers and returns its id (an index into `meshes`)
    uint32_t addMesh(const MeshData& md) {
        MeshInfo info;
        info.name = md.name;
        info.vertexOffset = (int32_t)meshVertices.size();   // the mesh's indices stay local; this offset is added at draw time
        info.firstIndex = (uint32_t)meshIndices.size();
        info.indexCount = (uint32_t)md.indices.size();
        info.bounds = md.bounds;

        // Textures -> GPU, then one material per glTF material (-1 / untextured = the default material 0)
        std::vector<uint32_t> imageTex(md.images.size());
        for (size_t i = 0; i < md.images.size(); i++)
            imageTex[i] = createTexture(md.images[i].rgba.data(), (uint32_t)md.images[i].width, (uint32_t)md.images[i].height);
        std::vector<uint32_t> materialId(md.materials.size(), 0);
        for (size_t i = 0; i < md.materials.size(); i++) {
            const MaterialData& mat = md.materials[i];
            // A material only needs its own entry if it differs from the default white one
            if (mat.baseColorImage >= 0 || mat.doubleSided || mat.alphaCutoff > 0.0f) {
                uint32_t tex = mat.baseColorImage >= 0 ? imageTex[mat.baseColorImage] : 0;   // texture 0 = the white pixel
                materialId[i] = createMaterial(tex, MaterialProps{ mat.doubleSided, mat.alphaCutoff });
                if (mat.baseColorImage >= 0) info.textureCount++;
            }
        }
        for (const Submesh& sm : md.submeshes)
            info.subs.push_back({ info.firstIndex + sm.firstIndex, sm.indexCount,
                                  sm.material >= 0 ? materialId[sm.material] : 0u });
        if (info.subs.empty()) info.subs.push_back({ info.firstIndex, info.indexCount, 0 });

        meshVertices.insert(meshVertices.end(), md.vertices.begin(), md.vertices.end());
        meshIndices.insert(meshIndices.end(), md.indices.begin(), md.indices.end());
        meshes.push_back(info);
        uploadMeshBuffers();
        return (uint32_t)meshes.size() - 1;
    }

    // ---------- Camera input ----------

    // Right mouse button held = fly mode: the cursor is captured, the mouse turns the camera,
    // WASD/QE move it. (The keys only work in fly mode so they stay free for editor shortcuts later.)
    void updateCamera() {
        bool imguiWantsMouse = imguiEnabled && ImGui::GetIO().WantCaptureMouse;
        bool rmb = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;

        if (rmb && !looking && !imguiWantsMouse) {          // just pressed, and not over the overlay
            looking = true;
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
            if (glfwRawMouseMotionSupported()) glfwSetInputMode(window, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);
            glfwGetCursorPos(window, &lastMouseX, &lastMouseY);   // read AFTER the mode change
        } else if (!rmb && looking) {                        // released
            looking = false;
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        }
        if (!looking) return;

        double mx, my;
        glfwGetCursorPos(window, &mx, &my);
        cam.look((float)(mx - lastMouseX), (float)(my - lastMouseY));
        lastMouseX = mx; lastMouseY = my;

        auto down = [&](int key) { return glfwGetKey(window, key) == GLFW_PRESS ? 1.0f : 0.0f; };
        float fwd = down(GLFW_KEY_W) - down(GLFW_KEY_S);
        float rgt = down(GLFW_KEY_D) - down(GLFW_KEY_A);
        float up  = down(GLFW_KEY_E) - down(GLFW_KEY_Q);
        float fast = (down(GLFW_KEY_LEFT_SHIFT) > 0.0f) ? 3.0f : 1.0f;
        cam.move(fwd, rgt, up, fast, (float)std::min(frameDt, 0.1));
    }

    // ---------- Editing: add / duplicate / delete ----------

    bool selectionValid() const {
        return selected != NULL_ENTITY && world.alive(selected) && world.has<Transform>(selected);
    }

    // A new cube 3 units in front of the camera (never below the floor), named and selected
    void addCube() {
        if (world.size() >= (size_t)MAX_CUBES) return;
        Entity e = world.create();
        Vec3 f = cam.forward();
        Transform t;
        t.position = { cam.position.x + f.x * 3.0f, std::max(cam.position.y + f.y * 3.0f, -0.25f), cam.position.z + f.z * 3.0f };
        t.scale = 0.5f;
        world.add<Transform>(e, t);
        world.add<Spin>(e, Spin{ 0.8f });
        world.add<MeshRenderer>(e, MeshRenderer{ 0, 0 });
        world.add<Name>(e, Name{ "Cube " + std::to_string(++cubeSerial) });
        selected = e;
        cubeCount = (int)world.size();
    }

    void duplicateSelected() {
        if (!selectionValid() || world.size() >= (size_t)MAX_CUBES) return;
        // Copy by value first: adding components can reallocate the pools these references point into
        Transform t = world.get<Transform>(selected);
        bool hasSpin = world.has<Spin>(selected);
        Spin spin = hasSpin ? world.get<Spin>(selected) : Spin{};
        MeshRenderer mr = world.has<MeshRenderer>(selected) ? world.get<MeshRenderer>(selected) : MeshRenderer{};
        const bool hasAgent = world.has<AIAgent>(selected);
        AIAgent agent = hasAgent ? world.get<AIAgent>(selected) : AIAgent{};
        agent.stop();
        const bool hasNav = world.has<NavSource>(selected);
        const NavSource nav = hasNav ? world.get<NavSource>(selected) : NavSource{};
        std::string name = world.has<Name>(selected) ? world.get<Name>(selected).value : std::string("Cube");

        Entity e = world.create();
        t.position.x += 0.6f * t.scale;           // offset so the copy doesn't sit inside the original
        world.add<Transform>(e, t);
        if (hasSpin) world.add<Spin>(e, spin);
        world.add<MeshRenderer>(e, mr);
        world.add<Name>(e, Name{ name + " (copy)" });
        if (hasNav) world.add<NavSource>(e, nav);
        if (hasAgent) world.add<AIAgent>(e, agent);
        selected = e;
        cubeCount = (int)world.size();
    }

    void deleteSelected() {
        if (!selectionValid()) return;
        world.destroy(selected);                   // removes it from every component pool
        selected = NULL_ENTITY;
        cubeCount = (int)world.size();
    }

    // ---------- Hierarchy panel ----------

    // A second window listing every entity (one row each), sharing the selection with picking.
    // ImGuiListClipper submits only the rows that are on screen, so 100,000 entities cost the same as 20.
    void buildHierarchy() {
        int winW, winH;
        glfwGetWindowSize(window, &winW, &winH);
        ImGui::SetNextWindowPos(ImVec2((float)winW - 270.0f, 10.0f), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(260.0f, 380.0f), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowBgAlpha(0.88f);
        ImGui::Begin("Hierarchy");

        bool full = world.size() >= (size_t)MAX_CUBES;
        bool hasSel = selectionValid();
        ImGui::BeginDisabled(full);
        if (ImGui::Button("+ Cube")) addCube();
        ImGui::EndDisabled();
        ImGui::SameLine();
        ImGui::BeginDisabled(!hasSel || full);
        if (ImGui::Button("Duplicate")) duplicateSelected();
        ImGui::EndDisabled();
        ImGui::SameLine();
        ImGui::BeginDisabled(!hasSel);
        if (ImGui::Button("Delete")) deleteSelected();
        ImGui::EndDisabled();

        Pool<Transform>& pool = world.pool<Transform>();
        const std::vector<Entity>& ents = pool.entities();
        ImGui::Text("%zu entities", ents.size());
        ImGui::Separator();

        ImGui::BeginChild("##entity_list");
        const float rowHeight = ImGui::GetTextLineHeightWithSpacing();

        // Selection changed somewhere else (a click in the 3D view): bring its row into view
        if (selected != lastScrolledTo) {
            lastScrolledTo = selected;
            if (selectionValid())
                ImGui::SetScrollY(std::max(0.0f, (float)pool.indexOf(selected) * rowHeight - ImGui::GetWindowHeight() * 0.5f));
        }

        ImGuiListClipper clipper;
        clipper.Begin((int)ents.size(), rowHeight);
        while (clipper.Step()) {
            for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; i++) {
                Entity e = ents[i];
                char label[128];
                if (world.has<Name>(e)) snprintf(label, sizeof(label), "%s", world.get<Name>(e).value.c_str());
                else                    snprintf(label, sizeof(label), "Cube %u", e);
                ImGui::PushID((int)e);                       // entity id keeps rows with equal names distinct
                if (ImGui::Selectable(label, e == selected)) {
                    selected = e;
                    lastScrolledTo = e;                      // we clicked it: no need to scroll
                    selectedDistance = 0.0f;
                }
                ImGui::PopID();
            }
        }
        clipper.End();
        ImGui::EndChild();

        ImGui::End();
    }

    // ---------- Transform gizmo ----------

    // Draws the move / rotate / scale handles on the selected entity and applies a drag to its Transform.
    // ImGuizmo wants column-major float[16] matrices and an OpenGL-style view/projection, which is exactly
    // how Mat4 stores its data (m[col][row]) and what lastView / lastProj hold.
    void buildGizmo() {
        if (!selectionValid() || !havePickMatrix) { gizmoOver = gizmoActive = false; return; }

        ImGuiIO& io = ImGui::GetIO();
        ImGuizmo::SetOrthographic(false);
        ImGuizmo::SetDrawlist(ImGui::GetBackgroundDrawList());   // behind the panels
        ImGuizmo::SetRect(0.0f, 0.0f, io.DisplaySize.x, io.DisplaySize.y);

        Transform& t = world.get<Transform>(selected);
        Mat4 model = modelMatrix(t);

        const float* snap = nullptr;
        float snapTranslate[3] = { 0.1f, 0.1f, 0.1f };
        float snapRotate[3]    = { 15.0f, 15.0f, 15.0f };      // degrees
        float snapScale[3]     = { 0.1f, 0.1f, 0.1f };
        if (gizmoSnap) snap = gizmoOp == ImGuizmo::TRANSLATE ? snapTranslate
                            : gizmoOp == ImGuizmo::ROTATE_Y  ? snapRotate : snapScale;

        // Scale and yaw are about the cube's own axes; moving is along the world axes
        ImGuizmo::MODE mode = gizmoOp == ImGuizmo::TRANSLATE ? ImGuizmo::WORLD : ImGuizmo::LOCAL;
        bool changed = ImGuizmo::Manipulate(&lastView.m[0][0], &lastProj.m[0][0], gizmoOp, mode, &model.m[0][0],
                                            nullptr, snap);
        gizmoOver = ImGuizmo::IsOver();
        gizmoActive = ImGuizmo::IsUsingAny();

        if (changed) {
            float oldY = t.position.y;
            t = transformFromMatrix(model, t);
            t.scale = std::max(t.scale, 0.01f);
            // A bobbing cube's height is baseY + wobble, so moving it up or down moves baseY too
            if (world.has<Bob>(selected)) world.get<Bob>(selected).baseY += t.position.y - oldY;
        }
    }

    // ---------- Assets ----------

    void rescanAssets() {
        namespace fs = std::filesystem;
        assetFiles.clear();
        std::error_code ec;
        for (fs::recursive_directory_iterator it("assets", ec), end; !ec && it != end; it.increment(ec)) {
            if (!it->is_regular_file(ec)) continue;
            std::string ext = it->path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return (char)std::tolower(c); });
            if (ext == ".gltf" || ext == ".glb") assetFiles.push_back(it->path().generic_string());
        }
        std::sort(assetFiles.begin(), assetFiles.end());
        lastAssetScan = glfwGetTime();
    }

    // Loads a glTF file (once) and returns its mesh id, or -1 on failure (the reason goes into assetStatus)
    int importAsset(const std::string& path) {
        for (const auto& loaded : loadedAssets)
            if (loaded.first == path) return (int)loaded.second;

        MeshData md;
        std::string error;
        if (!loadGltf(path, md, error)) { assetStatus = "Import failed: " + error; return -1; }
        uint32_t id = addMesh(md);
        loadedAssets.push_back({ path, id });
        loadedAssetDeps.push_back(md.dependencies);
        for (const std::string& d : md.dependencies) assetWatch.watch(d);
        assetStatus = "Imported " + md.name + ": " + std::to_string(md.vertices.size()) + " vertices, " +
                      std::to_string(md.indices.size() / 3) + " triangles, " + std::to_string(md.images.size()) + " texture(s)";
        for (const std::string& w : md.warnings) assetStatus += "\nWarning: " + w;
        return (int)id;
    }

    // An entity using the given mesh, placed 3 units in front of the camera and resting on the floor
    void spawnMeshEntity(uint32_t meshId) {
        if (world.size() >= (size_t)MAX_CUBES || meshId >= meshes.size()) return;
        const MeshInfo& info = meshes[meshId];
        Entity e = world.create();
        Vec3 f = cam.forward();
        Transform t;
        t.scale = 1.0f;                                           // imported meshes are normalized to a 1-unit box
        t.position = { cam.position.x + f.x * 3.0f, cam.position.y + f.y * 3.0f, cam.position.z + f.z * 3.0f };
        t.position.y = std::max(t.position.y, -0.5f - info.bounds.min.y * t.scale);   // don't start below the floor
        world.add<Transform>(e, t);
        world.add<MeshRenderer>(e, MeshRenderer{ meshId, 0 });
        world.add<Name>(e, Name{ info.name + " " + std::to_string(++cubeSerial) });
        selected = e;
        cubeCount = (int)world.size();
    }

    void buildAssets() {
        int winW, winH;
        glfwGetWindowSize(window, &winW, &winH);
        ImGui::SetNextWindowPos(ImVec2(winW * 0.5f - 190.0f, winH - 250.0f), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(380.0f, 240.0f), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowBgAlpha(0.88f);
        ImGui::Begin("Assets");

        if (glfwGetTime() - lastAssetScan > 2.0) rescanAssets();   // new files show up without a restart
        ImGui::TextDisabled("assets/ (.gltf, .glb), rescanned every 2 s");
        ImGui::SameLine();
        if (ImGui::SmallButton("Rescan")) rescanAssets();
        ImGui::Separator();

        if (assetFiles.empty()) ImGui::TextDisabled("No glTF files found in assets/");
        for (size_t i = 0; i < assetFiles.size(); i++) {
            ImGui::PushID((int)i);
            if (ImGui::SmallButton("Import + spawn")) {
                int id = importAsset(assetFiles[i]);
                if (id >= 0) spawnMeshEntity((uint32_t)id);
            }
            ImGui::SameLine();
            ImGui::TextUnformatted(assetFiles[i].c_str());
            ImGui::PopID();
        }

        if (meshes.size() > 2) {
            ImGui::Separator();
            ImGui::Text("Imported meshes");
            for (uint32_t i = 2; i < meshes.size(); i++) {
                if (meshes[i].retired) continue;
                ImGui::PushID((int)i);
                if (ImGui::SmallButton("Spawn")) spawnMeshEntity(i);
                ImGui::SameLine();
                ImGui::Text("%s  (%u triangles, %zu submesh%s, %u texture%s)", meshes[i].name.c_str(), meshes[i].indexCount / 3,
                            meshes[i].subs.size(), meshes[i].subs.size() == 1 ? "" : "es",
                            meshes[i].textureCount, meshes[i].textureCount == 1 ? "" : "s");
                ImGui::PopID();
            }
        }
        if (!assetStatus.empty()) { ImGui::Separator(); ImGui::TextWrapped("%s", assetStatus.c_str()); }
        ImGui::End();
    }

    // ---------- Inspector panel ----------

    // Editable widgets for each component of the selected entity. The widgets get references straight
    // into the ECS arrays, so a change is live on the next frame. Structural changes (adding or
    // removing a component) are applied at the end, after every reference has been used, because
    // they can move the arrays' memory.
    void buildInspector() {
        int winW, winH;
        glfwGetWindowSize(window, &winW, &winH);
        ImGui::SetNextWindowPos(ImVec2((float)winW - 330.0f, 400.0f), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(320.0f, 310.0f), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowBgAlpha(0.88f);
        ImGui::Begin("Inspector");

        if (!selectionValid()) {
            nameEditing = false;
            ImGui::TextDisabled("Nothing selected");
        } else {
            const Entity e = selected;
            bool removeSpin = false, removeBob = false, addSpin = false, addBob = false, addName = false;
            bool addAgent = false, removeAgent = false, addBrain = false, removeBrain = false, addHealth = false, removeHealth = false, makePlayer = false, unmakePlayer = false;
            int navChange = -1;      // new navmesh role chosen in the combo (0 = ignore), applied with the other structural changes

            ImGui::TextDisabled("Entity %u", e);

            // --- Name ---
            if (world.has<Name>(e)) {
                Name& n = world.get<Name>(e);
                if (!nameEditing) snprintf(nameBuf, sizeof(nameBuf), "%s", n.value.c_str());
                if (ImGui::InputText("Name", nameBuf, sizeof(nameBuf))) n.value = nameBuf;
                nameEditing = ImGui::IsItemActive();
            } else {
                nameEditing = false;
                ImGui::TextDisabled("Name: (none)");
                ImGui::SameLine();
                if (ImGui::SmallButton("Add name")) addName = true;
            }

            // --- Transform ---
            Transform& t = world.get<Transform>(e);
            if (ImGui::CollapsingHeader("Transform", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::DragFloat3("Position", &t.position.x, 0.01f);
                float yawDeg = t.yaw * 57.2957795f;
                if (ImGui::DragFloat("Yaw (deg)", &yawDeg, 0.5f)) t.yaw = yawDeg * 0.0174532925f;
                ImGui::DragFloat("Scale", &t.scale, 0.01f, 0.01f, 10.0f, "%.2f");   // min > 0: picking divides by it
                if (world.has<Spin>(e)) ImGui::TextDisabled("(Spin is rotating this cube; set its speed to 0 to hold a yaw)");
                if (world.has<Bob>(e)) ImGui::TextDisabled("(Bob controls Position Y; edit Base Y below)");
            }

            // --- Mesh ---
            if (world.has<MeshRenderer>(e)) {
                MeshRenderer& mr = world.get<MeshRenderer>(e);
                if (ImGui::CollapsingHeader("Mesh Renderer", ImGuiTreeNodeFlags_DefaultOpen)) {
                    const char* current = mr.meshId < meshes.size() ? meshes[mr.meshId].name.c_str() : "(missing)";
                    if (ImGui::BeginCombo("Mesh", current)) {
                        for (uint32_t i = 0; i < meshes.size(); i++) {
                            if (meshes[i].retired && i != mr.meshId) continue;
                            if (i == 1) continue;                    // the floor isn't an entity mesh
                            if (ImGui::Selectable(meshes[i].name.c_str(), i == mr.meshId)) mr.meshId = i;
                        }
                        ImGui::EndCombo();
                    }
                    if (mr.meshId < meshes.size()) ImGui::TextDisabled("%u triangles", meshes[mr.meshId].indexCount / 3);
                }
            }

            // --- Spin ---
            if (world.has<Spin>(e)) {
                bool open = ImGui::CollapsingHeader("Spin", ImGuiTreeNodeFlags_DefaultOpen);
                if (open) {
                    Spin& s = world.get<Spin>(e);
                    ImGui::DragFloat("Speed (rad/s)", &s.speed, 0.05f);
                    if (ImGui::SmallButton("Remove Spin")) removeSpin = true;
                }
            }

            // --- Bob ---
            if (world.has<Bob>(e)) {
                if (ImGui::CollapsingHeader("Bob", ImGuiTreeNodeFlags_DefaultOpen)) {
                    Bob& b = world.get<Bob>(e);
                    ImGui::DragFloat("Amplitude", &b.amplitude, 0.01f, 0.0f, 5.0f);
                    ImGui::DragFloat("Frequency", &b.frequency, 0.05f, 0.0f, 20.0f);
                    ImGui::DragFloat("Base Y", &b.baseY, 0.01f);
                    ImGui::DragFloat("Phase", &b.phase, 0.05f);
                    if (ImGui::SmallButton("Remove Bob")) removeBob = true;
                }
            }

            // --- AI agent ---
            if (world.has<AIAgent>(e)) {
                if (ImGui::CollapsingHeader("AI Agent", ImGuiTreeNodeFlags_DefaultOpen)) {
                    AIAgent& ag = world.get<AIAgent>(e);
                    ImGui::DragFloat("Move speed", &ag.moveSpeed, 0.01f, 0.0f, 10.0f, "%.2f");
                    ImGui::DragFloat("Turn speed", &ag.turnSpeed, 0.1f, 0.0f, 50.0f, "%.1f");
                    ImGui::DragFloat("Foot offset", &ag.footOffset, 0.005f, 0.0f, 5.0f, "%.3f");
                    if (ag.walking()) ImGui::Text("Walking: corner %u of %zu", ag.waypoint + 1, ag.path.size());
                    else ImGui::TextDisabled("Idle");
                    if (ImGui::SmallButton("Stop")) ag.stop();
                    ImGui::SameLine();
                    if (ImGui::SmallButton("Send to...") && navMesh.valid()) pathTool = PathTool::SendAgent;
                    ImGui::SameLine();
                    if (ImGui::SmallButton("Remove agent")) removeAgent = true;
                    if (pathTool == PathTool::SendAgent) ImGui::TextDisabled("Left-click on the navmesh (Esc cancels)");
                }
            }

            // --- Health ---
            if (world.has<Health>(e) && ImGui::CollapsingHeader("Health", ImGuiTreeNodeFlags_DefaultOpen)) {
                Health& h = world.get<Health>(e);
                ImGui::DragFloat("Max##hp", &h.max, 1.0f, 1.0f, 10000.0f, "%.0f");
                ImGui::SliderFloat("Current##hp", &h.current, 0.0f, h.max, "%.0f");
                if (ImGui::SmallButton("Remove health")) removeHealth = true;
            }

            // --- Level markers ---
            if (world.has<Goal>(e) && ImGui::CollapsingHeader("Goal", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::DragFloat("Radius##goal", &world.get<Goal>(e).radius, 0.01f, 0.05f, 5.0f, "%.2f");
                ImGui::TextDisabled("The run is won when the player's feet are inside this ring.");
            }
            if (world.has<StartPad>(e) && ImGui::CollapsingHeader("Start pad", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::DragFloat("Radius##start", &world.get<StartPad>(e).radius, 0.01f, 0.05f, 5.0f, "%.2f");
                ImGui::TextDisabled("Where the player goes on Restart run.");
            }

            // --- Player tag ---
            if (world.has<Player>(e)) {
                ImGui::TextColored(ImVec4(0.4f, 0.9f, 0.5f, 1), "This is the player: agents with a brain react to it.");
                if (ImGui::SmallButton("Not the player")) unmakePlayer = true;
            }

            // --- AI Brain ---
            if (world.has<AIBrain>(e) && ImGui::CollapsingHeader("AI Brain", ImGuiTreeNodeFlags_DefaultOpen)) {
                AIBrain& b = world.get<AIBrain>(e);
                if (b.lastLeaf >= 0 && b.lastLeaf < combatTree.nodeCount())
                    ImGui::Text("Doing: %s%s", combatTree.node(b.lastLeaf).name.c_str(), b.fleeing ? "  (hurt)" : "");
                else ImGui::TextDisabled("Not running (needs the AI to be on, and an AI Agent)");
                ImGui::DragFloat("Detect range", &b.detectRange, 0.01f, 0.0f, 20.0f, "%.2f");
                ImGui::DragFloat("Contact range", &b.contactRange, 0.005f, 0.0f, 5.0f, "%.3f");
                ImGui::SliderFloat("Flee below", &b.fleeBelow, 0.0f, 1.0f, "%.2f");
                ImGui::SliderFloat("Recover at", &b.recoverAt, 0.0f, 1.0f, "%.2f");
                ImGui::DragFloat("Attack (player) /s", &b.attackDamage, 0.5f, 0.0f, 1000.0f, "%.1f");
                ImGui::DragFloat("Contact damage /s", &b.contactDamage, 0.5f, 0.0f, 1000.0f, "%.1f");
                ImGui::DragFloat("Regen /s", &b.regenPerSecond, 0.5f, 0.0f, 1000.0f, "%.1f");
                ImGui::DragFloat("Patrol wait", &b.patrolWait, 0.05f, 0.0f, 30.0f, "%.2f s");
                if (b.recoverAt < b.fleeBelow) b.recoverAt = b.fleeBelow;
                ImGui::Text("Patrol points: %zu", b.patrol.size());
                int removePoint = -1;
                for (size_t i = 0; i < b.patrol.size(); i++) {
                    ImGui::PushID((int)i);
                    float v[3] = { b.patrol[i].x, b.patrol[i].y, b.patrol[i].z };
                    ImGui::SetNextItemWidth(190.0f);
                    if (ImGui::DragFloat3("##pt", v, 0.01f, -50.0f, 50.0f, "%.2f")) b.patrol[i] = { v[0], v[1], v[2] };
                    ImGui::SameLine();
                    ImGui::Text("%zu", i + 1);
                    ImGui::SameLine();
                    if (ImGui::SmallButton("x")) removePoint = (int)i;
                    ImGui::PopID();
                }
                if (removePoint >= 0) { b.patrol.erase(b.patrol.begin() + removePoint); b.patrolStarted = false; b.patrolIndex = 0; }
                if (ImGui::SmallButton("Add point...") && navMesh.valid()) pathTool = PathTool::AddPatrol;
                ImGui::SameLine();
                if (ImGui::SmallButton("Add here") && world.has<Transform>(e)) {
                    const Vec3& p = world.get<Transform>(e).position;
                    const float foot = world.has<AIAgent>(e) ? world.get<AIAgent>(e).footOffset : 0.0f;
                    b.patrol.push_back({ p.x, p.y - foot, p.z });
                }
                ImGui::SameLine();
                if (ImGui::SmallButton("Clear points")) { b.patrol.clear(); b.patrolStarted = false; b.patrolIndex = 0; }
                if (pathTool == PathTool::AddPatrol) ImGui::TextDisabled("Left-click on the navmesh (Esc cancels)");
                if (ImGui::SmallButton("Remove brain")) removeBrain = true;
            }

            // --- Add component ---
            ImGui::Separator();
            ImGui::BeginDisabled(world.has<Spin>(e));
            if (ImGui::Button("+ Spin")) addSpin = true;
            ImGui::EndDisabled();
            ImGui::SameLine();
            ImGui::BeginDisabled(world.has<Bob>(e));
            if (ImGui::Button("+ Bob")) addBob = true;
            ImGui::EndDisabled();

            if (world.has<MeshRenderer>(e) && !world.has<AIAgent>(e)) {
                ImGui::SameLine();
                if (ImGui::Button("+ AI Agent")) addAgent = true;
            }

            if (world.has<AIAgent>(e) && !world.has<AIBrain>(e)) {
                ImGui::SameLine();
                if (ImGui::Button("+ AI Brain")) addBrain = true;
            }
            if (!world.has<Health>(e)) {
                ImGui::SameLine();
                if (ImGui::Button("+ Health")) addHealth = true;
            }
            if (!world.has<Goal>(e) && world.has<Transform>(e)) {
                ImGui::SameLine();
                if (ImGui::Button("Make goal")) world.add<Goal>(e, Goal{});
            }
            if (!world.has<StartPad>(e) && world.has<Transform>(e)) {
                ImGui::SameLine();
                if (ImGui::Button("Make start")) world.add<StartPad>(e, StartPad{});
            }
            if (!world.has<Player>(e) && world.has<Transform>(e)) {
                ImGui::SameLine();
                if (ImGui::Button("Make player")) makePlayer = true;
            }

            // --- Navigation role (only meaningful for entities with a mesh) ---
            if (world.has<MeshRenderer>(e)) {
                const char* roles[] = { "Ignore", "Obstacle (solid)", "Level geometry" };
                int role = !world.has<NavSource>(e) ? 0 : (world.get<NavSource>(e).mode == NAV_GEOMETRY ? 2 : 1);
                if (ImGui::Combo("Navmesh", &role, roles, 3)) navChange = role;
            }

            // Structural changes last
            if (addAgent) {
                AIAgent ag;
                const MeshRenderer& mr = world.get<MeshRenderer>(e);
                const float minY = mr.meshId < meshes.size() ? meshes[mr.meshId].bounds.min.y : -0.5f;
                ag.footOffset = -minY * world.get<Transform>(e).scale;     // origin-to-feet distance for this mesh and size
                world.add<AIAgent>(e, ag);
            }
            if (removeAgent) { world.remove<AIBrain>(e); world.remove<AIAgent>(e); }
            if (addBrain) {
                AIBrain br;
                if (world.has<Transform>(e)) {          // start with a patrol point where it stands
                    const Vec3& p = world.get<Transform>(e).position;
                    br.patrol.push_back({ p.x, p.y - world.get<AIAgent>(e).footOffset, p.z });
                }
                world.add<AIBrain>(e, br);
                if (!world.has<Health>(e)) world.add<Health>(e, Health{});
            }
            if (removeBrain) world.remove<AIBrain>(e);
            if (addHealth) world.add<Health>(e, Health{});
            if (removeHealth) world.remove<Health>(e);
            if (makePlayer) {                            // only one player at a time
                std::vector<Entity> old;
                world.each<Player>([&](Entity o, Player&) { old.push_back(o); });
                for (Entity o : old) world.remove<Player>(o);
                world.add<Player>(e, Player{});
            }
            if (unmakePlayer) world.remove<Player>(e);
            if (navChange == 0) world.remove<NavSource>(e);
            else if (navChange > 0) world.add<NavSource>(e, NavSource{ navChange == 2 ? (uint8_t)NAV_GEOMETRY : (uint8_t)NAV_OBSTACLE });
            if (addName)    world.add<Name>(e, Name{ "Cube " + std::to_string(e) });
            if (addSpin)    world.add<Spin>(e, Spin{ 1.0f });
            if (addBob)     world.add<Bob>(e, Bob{ 0.15f, 2.0f, world.get<Transform>(e).position.y, 0.0f });
            if (removeSpin) world.remove<Spin>(e);
            if (removeBob)  world.remove<Bob>(e);
        }

        // --- Lights ---
        ImGui::Separator();
        if (ImGui::CollapsingHeader("Lights")) {
            for (int i = 0; i < 3; i++) {
                ImGui::PushID(i);
                EditorLight& l = lightCfg[i];
                ImGui::Text("Light %d", i);
                ImGui::DragFloat3("Position", l.pos, 0.02f);
                ImGui::ColorEdit3("Color", l.color);
                ImGui::DragFloat("Intensity", &l.intensity, 0.05f, 0.0f, 50.0f);
                if (i == 2) ImGui::Checkbox("Orbit around position", &light2Orbit);
                ImGui::PopID();
                if (i < 2) ImGui::Separator();
            }
        }

        ImGui::End();
    }

    // ---------- Picking ----------

    // Left click (not over the overlay, not while flying the camera) selects the entity under the cursor.
    void handlePicking() {
        bool leftDown = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
        bool clicked = leftDown && !prevLeftDown;       // the frame the button went down
        prevLeftDown = leftDown;

        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) selected = NULL_ENTITY;
        bool saveDown = glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS &&
                        (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS);
        if (saveDown && !prevSaveDown && !looking && !(imguiEnabled && ImGui::GetIO().WantTextInput)) saveScene(scenePath);
        prevSaveDown = saveDown;
        bool deleteDown = glfwGetKey(window, GLFW_KEY_DELETE) == GLFW_PRESS;
        if (deleteDown && !prevDeleteDown && !looking && !(imguiEnabled && ImGui::GetIO().WantTextInput)) deleteSelected();
        prevDeleteDown = deleteDown;

        // W / E / R pick the gizmo mode (only when not flying the camera or typing in a text box)
        const int hotkeys[3] = { GLFW_KEY_W, GLFW_KEY_E, GLFW_KEY_R };
        const ImGuizmo::OPERATION ops[3] = { ImGuizmo::TRANSLATE, ImGuizmo::ROTATE_Y, ImGuizmo::SCALEU };
        for (int i = 0; i < 3; i++) {
            bool down = glfwGetKey(window, hotkeys[i]) == GLFW_PRESS;
            if (down && !prevHotkey[i] && !looking && !(imguiEnabled && ImGui::GetIO().WantTextInput)) gizmoOp = ops[i];
            prevHotkey[i] = down;
        }
        if (pathTool != PathTool::Off) { handlePathClick(clicked); return; }     // the click places a path point instead of selecting
        if (!clicked || looking || !havePickMatrix) return;
        if (imguiEnabled && ImGui::GetIO().WantCaptureMouse) return;
        if (gizmoOver || gizmoActive) return;           // the click grabbed a gizmo handle, not the scene

        double mx, my;
        glfwGetCursorPos(window, &mx, &my);
        int winW, winH;
        glfwGetWindowSize(window, &winW, &winH);        // same units as the cursor position (not framebuffer pixels)
        Ray ray = screenToRay(pickInvViewProj, (float)mx, (float)my, (float)winW, (float)winH);
        std::vector<Aabb> bounds;                         // each mesh's own bounding box, indexed by mesh id
        bounds.reserve(meshes.size());
        for (const MeshInfo& m : meshes) bounds.push_back(m.bounds);
        selected = pickEntity(world, ray, bounds, &selectedDistance);   // clicking empty space clears the selection
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
        // We switch the cursor to "disabled" while mouse-looking; stop ImGui resetting it every frame
        io.ConfigFlags |= ImGuiConfigFlags_NoMouseCursorChange;
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
        ImGuizmo::BeginFrame();
        drawNavmeshOverlay();
        drawAIOverlay();
        drawGameUI();

        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowBgAlpha(0.88f);
        ImGui::Begin("Toy Engine", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

        if (ImGui::CollapsingHeader("Stats", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Text("%.0f FPS   (%.2f ms/frame)", smoothFrameMs > 0 ? 1000.0 / smoothFrameMs : 0.0, smoothFrameMs);
            ImGui::PlotLines("##frames", frameHistory, HISTORY, historyPos, nullptr, 0.0f, 20.0f, ImVec2(300, 50));
            ImGui::Text("Entities: %zu cubes", world.size());
            ImGui::Text("Draw calls: %u   Triangles: %u", drawCallsLastFrame,
                        drawnTriangles + meshes[1].indexCount / 3);
            ImGui::Separator();
            ImGui::Text("CPU frame: %.3f ms   GPU: %.3f ms", smoothCpuMs, smoothGpuMs);
            ImGui::Text("  spin + bob:      %.3f ms", msSpin);
            ImGui::Text("  buildDrawList:   %.3f ms", msDrawList);
            ImGui::Text("  instance upload: %.3f ms", msUpload);
        }

        if (ImGui::CollapsingHeader("Scene", ImGuiTreeNodeFlags_DefaultOpen)) {
            int n = cubeCount;
            if (ImGui::SliderInt("Cubes", &n, 1, MAX_CUBES, "%d", ImGuiSliderFlags_Logarithmic) && n != cubeCount) {
                cubeCount = n;
                demoScene = false;
                spawnScene();
            }
            if (ImGui::Button("Demo scene (named cubes)")) { demoScene = true; spawnScene(); }
            ImGui::Checkbox("Instanced drawing", &instanced);
            ImGui::Checkbox("Pause simulation", &simPaused);

            ImGui::Separator();
            ImGui::InputText("File", scenePath, sizeof(scenePath));
            if (ImGui::Button("Save")) saveScene(scenePath);
            ImGui::SameLine();
            if (ImGui::Button("Load")) loadScene(scenePath);
            ImGui::SameLine();
            ImGui::TextDisabled("Ctrl+S saves");
            if (!sceneStatus.empty()) ImGui::TextWrapped("%s", sceneStatus.c_str());
        }

        if (ImGui::CollapsingHeader("Debug views", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (!wireframeSupported) ImGui::BeginDisabled();
            ImGui::Checkbox("Wireframe", &wireframe);
            if (!wireframeSupported) ImGui::EndDisabled();
            ImGui::Checkbox("Lighting", &lightingOn);
            {
                const char* modes[] = { "Nearest", "Bilinear (no mipmaps)", "Trilinear (mipmaps)" };
                bool changed = ImGui::Combo("Texture filter", &filterMode, modes, 3);
                const bool anisoOk = maxAnisotropy > 1.0f && filterMode != 0;
                if (!anisoOk) ImGui::BeginDisabled();
                int maxA = (int)maxAnisotropy;
                if (ImGui::SliderInt("Anisotropy", &anisotropy, 1, std::max(maxA, 1), "%dx")) changed = true;
                if (!anisoOk) ImGui::EndDisabled();
                if (changed) changeSampler();
            }
            ImGui::Checkbox("Auto-sway camera (demo)", &camAutoSway);
        }

        if (ImGui::CollapsingHeader("Hot reload", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Checkbox("Watch shaders and assets", &hotReload);
            ImGui::SameLine();
            if (ImGui::SmallButton("Reload shaders now")) reloadShaders({ "shaders/mesh.vert", "shaders/mesh.frag" });
            ImGui::TextWrapped("%s", shaderStatus.c_str());
        }

        if (ImGui::CollapsingHeader("Navmesh", ImGuiTreeNodeFlags_DefaultOpen)) buildNavmeshPanel();
        if (ImGui::CollapsingHeader("AI Debug", ImGuiTreeNodeFlags_DefaultOpen)) buildAIDebugPanel();

        if (ImGui::CollapsingHeader("Gizmo", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (ImGui::RadioButton("Move (W)", gizmoOp == ImGuizmo::TRANSLATE)) gizmoOp = ImGuizmo::TRANSLATE;
            ImGui::SameLine();
            if (ImGui::RadioButton("Rotate (E)", gizmoOp == ImGuizmo::ROTATE_Y)) gizmoOp = ImGuizmo::ROTATE_Y;
            ImGui::SameLine();
            if (ImGui::RadioButton("Scale (R)", gizmoOp == ImGuizmo::SCALEU)) gizmoOp = ImGuizmo::SCALEU;
            ImGui::Checkbox("Snap (0.1 units / 15 deg / 0.1 scale)", &gizmoSnap);
            ImGui::TextDisabled("Rotation is yaw only: the Transform has no pitch or roll");
        }

        if (ImGui::CollapsingHeader("Selection", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (selected != NULL_ENTITY && world.alive(selected) && world.has<Transform>(selected)) {
                const char* name = world.has<Name>(selected) ? world.get<Name>(selected).value.c_str() : "(unnamed)";
                ImGui::Text("%s   (entity %u)", name, selected);
                if (selectedDistance > 0.0f) ImGui::Text("Picked at distance %.2f", selectedDistance);
                if (ImGui::Button("Clear selection")) selected = NULL_ENTITY;
            } else {
                ImGui::TextDisabled("Click a cube, or a row in the Hierarchy (Esc clears, Del deletes)");
            }
        }

        if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Text("Hold right mouse: look, then WASD move, Q/E down/up, Shift = fast");
            ImGui::Text("Position: %.2f  %.2f  %.2f", cam.position.x, cam.position.y, cam.position.z);
            ImGui::SliderFloat("Move speed", &cam.moveSpeed, 0.2f, 20.0f, "%.1f", ImGuiSliderFlags_Logarithmic);
            if (ImGui::Button("Reset camera")) cam = FlyCamera();
        }

        ImGui::End();
        buildHierarchy();
        buildInspector();
        buildAssets();
        buildGizmo();
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
        selected = NULL_ENTITY;
        game = GameState{};
        if (demoScene) { spawnDemoScene(); return; }
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

    // ---------- Navigation mesh (Step 7.1) ----------

    // Appends a mesh's triangles, placed by `model`, to the navmesh input. If `points` is given, the mesh's
    // world-space vertices are collected there too (to build the solid footprint).
    void addMeshToNavInput(NavInput& in, const MeshInfo& mesh, const Mat4& model, std::vector<float>* points) {
        auto world3 = [&](uint32_t index, float* out) {
            const Vec3& p = meshVertices[(size_t)((int64_t)index + mesh.vertexOffset)].pos;
            Vec3 w = (model * Vec4(p.x, p.y, p.z, 1.0f)).xyz();
            out[0] = w.x; out[1] = w.y; out[2] = w.z;
        };
        const uint32_t end = mesh.firstIndex + mesh.indexCount;
        for (uint32_t i = mesh.firstIndex; i + 2 < end; i += 3) {
            float a[3], b[3], c[3];
            world3(meshIndices[i], a);
            world3(meshIndices[i + 1], b);
            world3(meshIndices[i + 2], c);
            in.addTriangle(a, b, c);
            if (points) for (const float* p : { a, b, c }) points->insert(points->end(), p, p + 3);
        }
    }

    // The floor (always) plus every entity with a NavSource, at its current position
    void gatherNavInput(NavInput& in) {
        addMeshToNavInput(in, meshes[1], Mat4::identity(), nullptr);
        world.each<NavSource, MeshRenderer, Transform>([&](Entity, NavSource& nav, MeshRenderer& mr, Transform& t) {
            if (mr.meshId >= meshes.size()) return;
            std::vector<float> points;
            addMeshToNavInput(in, meshes[mr.meshId], modelMatrix(t), nav.mode == NAV_OBSTACLE ? &points : nullptr);
            if (nav.mode == NAV_OBSTACLE) in.addSolid(points.data(), (int)(points.size() / 3));
        });
    }

    void bakeNavmesh() {
        NavInput in;
        gatherNavInput(in);
        std::string error;
        if (navMesh.build(in, navParams, error)) {
            const NavStats& s = navMesh.stats();
            computePath();      // the old route may no longer be valid on the new navmesh
            navStatus = "Built from " + std::to_string(in.triangleCount()) + " triangles: " + std::to_string(s.polygons) +
                        " polygons, " + std::to_string(s.triangles) + " triangles, " +
                        std::to_string((int)(s.buildMs + 0.5)) + " ms";
        } else {
            navStatus = "Build failed" + std::string(navMesh.valid() ? " (keeping the previous navmesh)" : "") + ": " + error;
        }
    }

    void buildNavmeshPanel() {
        ImGui::Checkbox("Show overlay", &showNavmesh);
        ImGui::SameLine();
        if (ImGui::Button("Build")) bakeNavmesh();
        ImGui::SameLine();
        if (ImGui::Button("Clear")) { navMesh.clear(); navStatus = "cleared"; navPath = NavPath{}; pathTool = PathTool::Off; pathStatus.clear(); }

        if (ImGui::Button("Mark all entities as obstacles")) {
            std::vector<Entity> all = world.pool<MeshRenderer>().entities();      // copy: adding NavSource must not disturb the loop
            for (Entity e : all) if (!world.has<NavSource>(e)) world.add<NavSource>(e, NavSource{ NAV_OBSTACLE });
        }
        ImGui::SameLine();
        if (ImGui::Button("Unmark all")) {
            std::vector<Entity> all = world.pool<NavSource>().entities();
            for (Entity e : all) world.remove<NavSource>(e);
        }
        ImGui::TextDisabled("The floor is always included. Pick an entity to set its role in the Inspector.");

        if (ImGui::TreeNode("Agent and build settings")) {
            ImGui::DragFloat("Agent radius", &navParams.agentRadius, 0.005f, 0.01f, 1.0f, "%.3f");
            ImGui::DragFloat("Agent height", &navParams.agentHeight, 0.01f, 0.05f, 3.0f, "%.2f");
            ImGui::DragFloat("Max climb", &navParams.agentMaxClimb, 0.005f, 0.0f, 1.0f, "%.3f");
            ImGui::DragFloat("Max slope (deg)", &navParams.agentMaxSlope, 0.5f, 0.0f, 89.0f, "%.1f");
            ImGui::DragFloat("Cell size", &navParams.cellSize, 0.002f, 0.01f, 0.5f, "%.3f");
            ImGui::DragFloat("Cell height", &navParams.cellHeight, 0.001f, 0.005f, 0.5f, "%.3f");
            if (ImGui::Button("Reset to defaults")) navParams = NavParams{};
            ImGui::TreePop();
        }
        ImGui::TextWrapped("%s", navStatus.c_str());

        ImGui::Separator();
        ImGui::Text("Path query");
        if (!navMesh.valid()) ImGui::BeginDisabled();
        if (ImGui::Button(pathTool == PathTool::SetStart ? "Click the navmesh..." : "Set start")) pathTool = PathTool::SetStart;
        ImGui::SameLine();
        if (ImGui::Button(pathTool == PathTool::SetEnd ? "Click the navmesh..." : "Set end")) pathTool = PathTool::SetEnd;
        ImGui::SameLine();
        if (ImGui::Button("Clear path")) {
            haveStart = haveEnd = false;
            navPath = NavPath{};
            pathTool = PathTool::Off;
            pathStatus.clear();
        }
        if (ImGui::Button("Spawn agent at start")) spawnAgentAtStart();
        if (!navMesh.valid()) ImGui::EndDisabled();
        ImGui::Checkbox("Show agent paths", &showAgentPaths);
        ImGui::Checkbox("Run AI", &runAI);
        ImGui::Separator();
        ImGui::TextUnformatted("Place by clicking the navmesh:");
        {
            const char* kinds[] = { "Plain agent", "Guard (patrols, chases, flees)", "Player" };
            int k = (int)placeKind;
            if (ImGui::Combo("What##place", &k, kinds, 3)) placeKind = (ActorKind)k;
            if (placeMesh >= meshes.size() || placeMesh == 1) placeMesh = 0;
            if (ImGui::BeginCombo("Model##place", meshes[placeMesh].name.c_str())) {
                for (uint32_t i = 0; i < meshes.size(); i++) {
                    if (i == 1 || meshes[i].retired) continue;
                    if (ImGui::Selectable((meshes[i].name + "##pm" + std::to_string(i)).c_str(), i == placeMesh)) placeMesh = i;
                }
                ImGui::EndCombo();
            }
            if (ImGui::Button(pathTool == PathTool::Place ? "Click the navmesh... (Esc to stop)" : "Place...")) pathTool = PathTool::Place;
        }
        ImGui::Separator();
        if (ImGui::Button("Build demo level")) buildAIDemo();
        ImGui::SameLine();
        ImGui::TextDisabled("(replaces the scene)");
        if (game.active) {
            if (ImGui::Button("Restart run")) restartRun();
            ImGui::SameLine();
            if (ImGui::Button(pathTool == PathTool::MovePlayer ? "Click the floor..." : "Click to move player")) pathTool = PathTool::MovePlayer;
        }
        ImGui::Separator();
        if (ImGui::Button("Spawn player at start")) spawnPlayerAtStart();
        ImGui::SameLine();
        if (ImGui::Button("Spawn guard at start")) spawnGuardAtStart();
        if (pathTool != PathTool::Off) ImGui::TextDisabled("Left-click on the navmesh to place it (Esc cancels)");
        if (!pathStatus.empty()) ImGui::TextWrapped("%s", pathStatus.c_str());
    }

    void computePath() {
        navPath = NavPath{};
        if (!haveStart || !haveEnd || !navMesh.valid()) return;
        std::string error;
        if (navMesh.findPath(pathStart, pathEnd, navPath, error)) {
            char buf[160];
            std::snprintf(buf, sizeof buf, "Path: %zu waypoints, length %.2f%s", navPath.count(), navPath.length,
                          navPath.partial ? " (the end is unreachable: stops at the closest point)" : "");
            pathStatus = buf;
        } else {
            pathStatus = "No path: " + error;
        }
    }

    // ---------- The full-loop demo level (Step 8.3) ----------

    // Replaces the scene with the demo level (see demo_level.h): a zig-zag course with guards, a start pad and a goal.
    // It is ordinary entities with ordinary components, so it can be edited, saved and loaded like anything else.
    void buildAIDemo() {
        world = Registry();
        selected = NULL_ENTITY;
        game = GameState{};
        haveStart = haveEnd = false;
        navPath = NavPath{};
        cubeSerial = 0;
        const DemoLevelInfo info = buildDemoLevel(world);
        agentSerial = info.guards;
        selected = info.player;
        cubeCount = (int)world.size();
        bakeNavmesh();
        cam.position = { 0.0f, 7.0f, 1.0f };
        cam.yaw = 0.0f;
        cam.pitch = -1.12f;
        if (navMesh.valid()) {
            pathTool = PathTool::MovePlayer;
            pathStatus = "Demo level ready: arrow keys (or clicking the floor) walk the player to the glowing GOAL. Avoid the guards.";
        } else {
            pathTool = PathTool::Off;
            pathStatus = "Demo level: the navmesh failed to build: " + navStatus;
        }
    }

    // Arrow keys walk the player directly: Up/Down along the camera's heading, Left/Right sideways. Movement goes
    // through the navmesh (moveAlong), so walls and edges stop it and it slides along them. Pressing a key cancels
    // any click-to-move path. The camera keeps WASD and the mouse, so the arrows are free.
    void playerKeyboardMove(float dt) {
        if (dt <= 0.0f || !navMesh.valid() || !imguiEnabled || ImGui::GetIO().WantTextInput) return;
        auto down = [&](int key) { return glfwGetKey(window, key) == GLFW_PRESS ? 1.0f : 0.0f; };
        const float fwd = down(GLFW_KEY_UP) - down(GLFW_KEY_DOWN);
        const float side = down(GLFW_KEY_RIGHT) - down(GLFW_KEY_LEFT);
        if (fwd == 0.0f && side == 0.0f) return;
        const Entity player = findPlayer(world);
        if (player == NULL_ENTITY || !world.has<AIAgent>(player)) return;
        AIAgent& ag = world.get<AIAgent>(player);
        Transform& t = world.get<Transform>(player);
        float dx = std::sin(cam.yaw) * fwd + std::cos(cam.yaw) * side;      // camera heading on the ground
        float dz = -std::cos(cam.yaw) * fwd + std::sin(cam.yaw) * side;
        const float len = std::sqrt(dx * dx + dz * dz);
        dx /= len; dz /= len;
        ag.stop();
        const float feet[3] = { t.position.x, t.position.y - ag.footOffset, t.position.z };
        const float want[3] = { feet[0] + dx * ag.moveSpeed * dt, feet[1], feet[2] + dz * ag.moveSpeed * dt };
        float out[3];
        if (!navMesh.moveAlong(feet, want, out)) return;
        t.position = { out[0], out[1] + ag.footOffset, out[2] };
        t.yaw = turnToward(t.yaw, std::atan2(dx, dz), std::max(ag.turnSpeed, 1.0f) * dt);
    }

    void restartRun() {
        resetRun(world, game);
        if (navMesh.valid() && findPlayer(world) != NULL_ENTITY) pathTool = PathTool::MovePlayer;
    }

    // Projects a world point to screen pixels; false if it is behind the camera
    bool worldToScreen(const Vec3& p, ImVec2& out) const {
        const ImVec2 disp = ImGui::GetIO().DisplaySize;
        const Vec4 c = (lastProj * lastView) * Vec4(p.x, p.y, p.z, 1.0f);
        if (c.w < 0.05f) return false;
        out = ImVec2((c.x / c.w * 0.5f + 0.5f) * disp.x, (1.0f - (c.y / c.w * 0.5f + 0.5f)) * disp.y);
        return true;
    }

    // Rings on the floor for the start pad and the goal, and the run's HUD (health, time, win/lose banner)
    void drawGameUI() {
        ImDrawList* dl = ImGui::GetBackgroundDrawList();
        auto ring = [&](Entity e, float radius, ImU32 color, const char* label) {
            const Vec3 c = world.get<Transform>(e).position;
            const float pulse = 1.0f + 0.08f * std::sin((float)simTime * 3.0f);
            ImVec2 prev;
            bool havePrev = false;
            for (int i = 0; i <= 40; i++) {
                const float a = (float)i / 40.0f * 6.2831853f;
                ImVec2 p;
                const bool ok = worldToScreen({ c.x + std::cos(a) * radius * pulse, c.y + 0.012f, c.z + std::sin(a) * radius * pulse }, p);
                if (ok && havePrev) dl->AddLine(prev, p, color, 3.0f);
                prev = p; havePrev = ok;
            }
            ImVec2 mid;
            if (worldToScreen({ c.x, c.y + 0.012f, c.z }, mid)) {
                const ImVec2 sz = ImGui::CalcTextSize(label);
                dl->AddText(ImVec2(mid.x - sz.x * 0.5f, mid.y - sz.y * 0.5f), color, label);
            }
        };
        world.each<StartPad, Transform>([&](Entity e, StartPad& s, Transform&) { ring(e, s.radius, IM_COL32(90, 200, 255, 255), "START"); });
        world.each<Goal, Transform>([&](Entity e, Goal& g, Transform&) { ring(e, g.radius, IM_COL32(255, 220, 70, 255), "GOAL"); });

        if (!game.active || !runAI) return;
        const ImVec2 disp = ImGui::GetIO().DisplaySize;
        const Entity player = findPlayer(world);
        ImGui::SetNextWindowPos(ImVec2(disp.x * 0.5f, 10.0f), ImGuiCond_Always, ImVec2(0.5f, 0.0f));
        ImGui::SetNextWindowBgAlpha(0.7f);
        ImGui::Begin("##hud", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove |
                                       ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav);
        ImGui::Text("Reach the GOAL     Time %.1f s", game.time);
        if (player != NULL_ENTITY && world.has<Health>(player)) {
            const Health& h = world.get<Health>(player);
            ImGui::ProgressBar(h.fraction(), ImVec2(220, 14), "");
            ImGui::SameLine();
            ImGui::Text("HP %.0f", h.current);
        }
        if (pathTool == PathTool::MovePlayer) ImGui::TextDisabled("Arrow keys or click the floor to move. Esc stops click-to-move.");
        else if (game.phase == GameState::Playing) { if (ImGui::SmallButton("Click to move")) pathTool = PathTool::MovePlayer; }
        ImGui::End();

        if (game.phase != GameState::Playing) {
            const bool won = game.phase == GameState::Won;
            ImGui::SetNextWindowPos(ImVec2(disp.x * 0.5f, disp.y * 0.4f), ImGuiCond_Always, ImVec2(0.5f, 0.5f));
            ImGui::SetNextWindowBgAlpha(0.85f);
            ImGui::Begin("##result", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove |
                                              ImGuiWindowFlags_NoSavedSettings);
            ImGui::SetWindowFontScale(2.0f);
            ImGui::TextColored(won ? ImVec4(0.4f, 1.0f, 0.5f, 1) : ImVec4(1.0f, 0.4f, 0.35f, 1), won ? "GOAL REACHED!" : "CAUGHT!");
            ImGui::SetWindowFontScale(1.0f);
            ImGui::Text("%s in %.1f seconds", won ? "Finished" : "Lasted", game.time);
            if (ImGui::Button("Play again")) restartRun();
            ImGui::End();
        }
    }

    // ---------- Placing actors (Step 8.1) ----------

    enum class ActorKind { Agent, Guard, Player };

    // Creates a walking cube (or imported model) standing with its feet at `feet`. Returns the entity, or NULL_ENTITY.
    Entity spawnActor(ActorKind kind, const float feet[3], uint32_t meshId) {
        if (world.size() >= (size_t)MAX_CUBES) return NULL_ENTITY;
        if (meshId >= meshes.size() || meshId == 1) meshId = 0;                // not the floor
        if (kind == ActorKind::Player) {                                       // only one player at a time
            std::vector<Entity> old;
            world.each<Player>([&](Entity o, Player&) { old.push_back(o); });
            for (Entity o : old) world.remove<Player>(o);
        }
        const Aabb& bb = meshes[meshId].bounds;
        const float height = std::max(bb.max.y - bb.min.y, 0.001f);
        Transform t;
        t.scale = (meshId == 0 ? 0.15f : 0.25f / height) * (kind == ActorKind::Player ? 4.0f / 3.0f : 1.0f);
        AIAgent ag;
        ag.footOffset = -bb.min.y * t.scale;
        if (kind == ActorKind::Player) ag.moveSpeed = 0.8f;
        t.position = { feet[0], feet[1] + ag.footOffset, feet[2] };
        Entity e = world.create();
        world.add<Transform>(e, t);
        world.add<MeshRenderer>(e, MeshRenderer{ meshId, 0 });
        world.add<AIAgent>(e, ag);
        if (kind == ActorKind::Player) {
            world.add<Name>(e, Name{ "Player" });
            world.add<Player>(e, Player{});
        } else {
            world.add<Name>(e, Name{ (kind == ActorKind::Guard ? "Guard " : "Agent ") + std::to_string(++agentSerial) });
            if (kind == ActorKind::Guard) {
                AIBrain br;
                br.patrol.push_back({ feet[0], feet[1], feet[2] });
                world.add<AIBrain>(e, br);
                world.add<Health>(e, Health{});
            }
        }
        selected = e;
        cubeCount = (int)world.size();
        return e;
    }

    uint32_t placeMesh = 0;                                  // which mesh the Place tool uses
    ActorKind placeKind = ActorKind::Guard;

    void spawnAgentAtStart() {
        if (!haveStart) { pathStatus = "Set a start point first (Set start, then click the navmesh)."; return; }
        if (spawnActor(ActorKind::Agent, pathStart, 0) != NULL_ENTITY)
            pathStatus = "Spawned an agent. Use Send to... on it, or click Set end to see a route.";
    }
    // ---------- AI debug view (Step 7.5) ----------

    static ImU32 statusColor(int s) {
        switch (s) {
            case (int)BTStatus::Success: return IM_COL32(90, 220, 110, 255);
            case (int)BTStatus::Failure: return IM_COL32(235, 90, 80, 255);
            case (int)BTStatus::Running: return IM_COL32(255, 210, 60, 255);
            default: return IM_COL32(130, 130, 130, 255);        // not visited this tick
        }
    }
    static const char* statusText(int s) {
        return s < 0 ? "-" : toString((BTStatus)s);
    }

    // Draws one node and its children, coloured by what each returned on the last tick
    void drawTreeNode(const BTNode& n, const std::vector<int8_t>& trace, int depth) {
        const int st = (size_t)n.id < trace.size() ? trace[(size_t)n.id] : -1;
        const char* kind = n.kind == BTNode::Kind::Selector ? "Selector" : (n.kind == BTNode::Kind::Sequence ? "Sequence" : "Leaf");
        ImGui::Indent(depth > 0 ? 14.0f : 0.0f);
        ImGui::PushStyleColor(ImGuiCol_Text, statusColor(st));
        ImGui::Text("%s %s  [%s]", kind, n.name.c_str(), statusText(st));
        ImGui::PopStyleColor();
        for (const auto& c : n.children) drawTreeNode(*c, trace, depth + 1);
        ImGui::Unindent(depth > 0 ? 14.0f : 0.0f);
    }

    void buildAIDebugPanel() {
        ImGui::Checkbox("Labels above agents", &showAILabels);
        if (!runAI) ImGui::TextDisabled("AI is off (Navmesh > Run AI)");
        const bool haveSel = selectionValid() && world.has<AIBrain>(selected);
        if (haveSel) {
            ImGui::TextDisabled("Tree for the selected agent (green ok, red failed, yellow running, grey not reached):");
            drawTreeNode(combatTree.root(), world.get<AIBrain>(selected).trace, 0);
        } else {
            ImGui::TextDisabled("Select an agent with an AI Brain to see its tree.");
        }
        ImGui::Separator();
        int count = 0;
        if (ImGui::BeginTable("aiagents", 4, ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_RowBg)) {
            ImGui::TableSetupColumn("Agent");
            ImGui::TableSetupColumn("Doing");
            ImGui::TableSetupColumn("HP");
            ImGui::TableSetupColumn("State");
            ImGui::TableHeadersRow();
            world.each<AIBrain>([&](Entity e, AIBrain& b) {
                if (++count > 40) return;                         // keep the table short
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                const std::string label = (world.has<Name>(e) ? world.get<Name>(e).value : "Entity " + std::to_string(e)) + "##ai" + std::to_string(e);
                if (ImGui::Selectable(label.c_str(), e == selected)) selected = e;
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(b.lastLeaf >= 0 && b.lastLeaf < combatTree.nodeCount() ? combatTree.node(b.lastLeaf).name.c_str() : "-");
                ImGui::TableNextColumn();
                if (world.has<Health>(e)) ImGui::Text("%.0f", world.get<Health>(e).current); else ImGui::TextUnformatted("-");
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(b.fleeing ? "hurt" : (b.chasing ? "sees player" : "calm"));
            });
            ImGui::EndTable();
        }
        if (count == 0) ImGui::TextDisabled("No agents have a brain yet.");
    }

    // Text and a health bar floating above every agent that has a brain, via ImGui's background draw list
    void drawAIOverlay() {
        if (!showAILabels || !runAI) return;
        ImDrawList* dl = ImGui::GetBackgroundDrawList();
        const ImVec2 disp = ImGui::GetIO().DisplaySize;
        const Mat4 vp = lastProj * lastView;
        world.each<AIBrain, Transform>([&](Entity e, AIBrain& b, Transform& t) {
            if (b.lastLeaf < 0 || b.lastLeaf >= combatTree.nodeCount()) return;
            float top = t.scale * 0.5f;                                   // fallback: unit-ish mesh
            if (world.has<MeshRenderer>(e)) {
                const uint32_t m = world.get<MeshRenderer>(e).meshId;
                if (m < meshes.size()) top = meshes[m].bounds.max.y * t.scale;
            }
            Vec4 c = vp * Vec4(t.position.x, t.position.y + top + 0.06f, t.position.z, 1.0f);
            if (c.w < 0.05f) return;                                      // behind the camera
            const ImVec2 p((c.x / c.w * 0.5f + 0.5f) * disp.x, (1.0f - (c.y / c.w * 0.5f + 0.5f)) * disp.y);
            if (p.x < -200 || p.y < -50 || p.x > disp.x + 200 || p.y > disp.y + 50) return;

            const char* leafName = combatTree.node(b.lastLeaf).name.c_str();
            const ImU32 col = b.fleeing ? IM_COL32(255, 120, 90, 255)
                            : (b.pathOwner == AIBrain::OWNER_CHASE ? IM_COL32(255, 210, 60, 255) : IM_COL32(150, 230, 255, 255));
            const ImVec2 sz = ImGui::CalcTextSize(leafName);
            const ImVec2 tp(p.x - sz.x * 0.5f, p.y - sz.y - 8.0f);
            dl->AddRectFilled(ImVec2(tp.x - 3, tp.y - 1), ImVec2(tp.x + sz.x + 3, tp.y + sz.y + 1), IM_COL32(0, 0, 0, 150), 3.0f);
            dl->AddText(tp, col, leafName);
            if (world.has<Health>(e)) {                                   // bar under the text
                const float w = 40.0f, h = 4.0f, f = world.get<Health>(e).fraction();
                const ImVec2 a(p.x - w * 0.5f, p.y - 6.0f);
                dl->AddRectFilled(a, ImVec2(a.x + w, a.y + h), IM_COL32(0, 0, 0, 180));
                dl->AddRectFilled(a, ImVec2(a.x + w * f, a.y + h), f < b.fleeBelow ? IM_COL32(235, 80, 70, 255) : IM_COL32(90, 220, 110, 255));
            }
        });
    }

    void spawnPlayerAtStart() {
        if (!haveStart) { pathStatus = "Set a start point first (Set start, then click the navmesh)."; return; }
        if (spawnActor(ActorKind::Player, pathStart, 0) != NULL_ENTITY)
            pathStatus = "Spawned the player. Use Send to... to walk it, or drag it with the gizmo.";
    }
    // A guard that patrols between the start and end markers (whichever are set)
    void spawnGuardAtStart() {
        if (!haveStart) { pathStatus = "Set a start point first (Set start, then click the navmesh)."; return; }
        Entity e = spawnActor(ActorKind::Guard, pathStart, 0);
        if (e == NULL_ENTITY) return;
        if (haveEnd) world.get<AIBrain>(e).patrol.push_back({ pathEnd[0], pathEnd[1], pathEnd[2] });
        pathStatus = "Spawned a guard. It patrols the start/end markers, chases the player when close, and flees when hurt.";
    }
    // Sends the selected agent to a point: finds a path from where its feet are now and hands it over
    void sendAgentTo(const float target[3], Entity who) {
        if (who == NULL_ENTITY || !world.has<AIAgent>(who) || !world.has<Transform>(who)) {
            pathStatus = "Select an AI agent first.";
            return;
        }
        AIAgent& ag = world.get<AIAgent>(who);
        const Vec3& p = world.get<Transform>(who).position;
        const float feet[3] = { p.x, p.y - ag.footOffset, p.z };
        NavPath path;
        std::string error;
        if (!navMesh.findPath(feet, target, path, error)) { pathStatus = "No path for the agent: " + error; return; }
        std::vector<Vec3> corners;
        for (size_t i = 0; i < path.count(); i++) corners.push_back({ path.point(i)[0], path.point(i)[1], path.point(i)[2] });
        ag.setPath(std::move(corners));
        pathStatus = std::string("Agent is walking ") + std::to_string(path.count()) + " waypoints" + (path.partial ? " (goal unreachable: stopping at the closest point)" : "");
    }

    // The path tool is armed: a left click places a point on the navmesh under the cursor
    void handlePathClick(bool clicked) {
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) pathTool = PathTool::Off;
        if (!clicked || looking || !havePickMatrix) return;
        if (imguiEnabled && ImGui::GetIO().WantCaptureMouse) return;
        if (!navMesh.valid()) { pathStatus = "Build the navmesh first."; pathTool = PathTool::Off; return; }

        double mx, my;
        glfwGetCursorPos(window, &mx, &my);
        int winW, winH;
        glfwGetWindowSize(window, &winW, &winH);
        Ray ray = screenToRay(pickInvViewProj, (float)mx, (float)my, (float)winW, (float)winH);
        const float o[3] = { ray.origin.x, ray.origin.y, ray.origin.z }, d[3] = { ray.dir.x, ray.dir.y, ray.dir.z };
        float t = 0.0f;
        if (!navMesh.raycast(o, d, t)) { pathStatus = "That click did not hit the navmesh."; return; }
        const float hit[3] = { o[0] + d[0] * t, o[1] + d[1] * t, o[2] + d[2] * t };
        if (pathTool == PathTool::SendAgent) { sendAgentTo(hit, selectionValid() ? selected : NULL_ENTITY); pathTool = PathTool::Off; return; }
        if (pathTool == PathTool::MovePlayer) {          // stays armed: click again to redirect the player
            if (game.phase == GameState::Playing) sendAgentTo(hit, findPlayer(world));
            return;
        }
        if (pathTool == PathTool::Place) {               // stays armed (except for the player) so several can be placed
            spawnActor(placeKind, hit, placeMesh);
            if (placeKind == ActorKind::Player) pathTool = PathTool::Off;
            return;
        }
        if (pathTool == PathTool::AddPatrol) {          // stay armed so several points can be placed in a row
            if (selectionValid() && world.has<AIBrain>(selected)) world.get<AIBrain>(selected).patrol.push_back({ hit[0], hit[1], hit[2] });
            else pathTool = PathTool::Off;
            return;
        }
        float* target = pathTool == PathTool::SetStart ? pathStart : pathEnd;
        target[0] = hit[0]; target[1] = hit[1]; target[2] = hit[2];
        if (pathTool == PathTool::SetStart) { haveStart = true; pathTool = haveEnd ? PathTool::Off : PathTool::SetEnd; }
        else { haveEnd = true; pathTool = PathTool::Off; }
        computePath();
    }

    // Draws the navmesh over the scene with ImGui's background draw list (so it reuses the debug overlay and
    // needs no extra Vulkan pipeline). It has no depth test: the mesh shows through objects, which is what you
    // want for checking it, but it does mean it can't be hidden behind a wall.
    void drawNavmeshOverlay() {
        if (!navMesh.valid()) return;
        ImDrawList* dl = ImGui::GetBackgroundDrawList();
        const ImVec2 disp = ImGui::GetIO().DisplaySize;
        const Mat4 vp = lastProj * lastView;
        const float lift = 0.01f;                    // draw slightly above the surface so it doesn't sink into the floor

        struct Clip { float x, y, z, w; };
        auto toClip = [&](const float* p) {
            Vec4 c = vp * Vec4(p[0], p[1] + lift, p[2], 1.0f);
            return Clip{ c.x, c.y, c.z, c.w };
        };
        auto toScreen = [&](const Clip& c) {
            return ImVec2((c.x / c.w * 0.5f + 0.5f) * disp.x, (1.0f - (c.y / c.w * 0.5f + 0.5f)) * disp.y);
        };
        const float nearW = 0.05f;                   // clip against this w: anything closer is behind or beside the camera
        auto lerp = [](const Clip& a, const Clip& b, float t) {
            return Clip{ a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t, a.w + (b.w - a.w) * t };
        };

        if (showNavmesh) {
        const ImDrawListFlags oldFlags = dl->Flags;
        dl->Flags &= ~ImDrawListFlags_AntiAliasedFill;      // anti-aliased edges would show seams between triangles

        for (const NavDebugTri& t : navMesh.debugTriangles()) {
            Clip in[3] = { toClip(t.a), toClip(t.b), toClip(t.c) };
            Clip poly[4];
            int n = 0;
            for (int i = 0; i < 3; i++) {            // Sutherland-Hodgman against the near plane
                const Clip& cur = in[i];
                const Clip& nxt = in[(i + 1) % 3];
                const bool curIn = cur.w >= nearW, nxtIn = nxt.w >= nearW;
                if (curIn) poly[n++] = cur;
                if (curIn != nxtIn) poly[n++] = lerp(cur, nxt, (nearW - cur.w) / (nxt.w - cur.w));
            }
            if (n < 3) continue;
            ImVec2 pts[4];
            for (int i = 0; i < n; i++) pts[i] = toScreen(poly[i]);
            // A stable colour per polygon, so the polygon structure is visible
            float hue = (float)((t.poly * 2654435761u) >> 8 & 0xFFFF) / 65535.0f;
            float r, g, b;
            ImGui::ColorConvertHSVtoRGB(hue, 0.55f, 0.95f, r, g, b);
            dl->AddConvexPolyFilled(pts, n, ImGui::ColorConvertFloat4ToU32(ImVec4(r, g, b, 0.30f)));
        }
        dl->Flags = oldFlags;

        for (const NavDebugEdge& e : navMesh.debugEdges()) {
            Clip a = toClip(e.a), b = toClip(e.b);
            if (a.w < nearW && b.w < nearW) continue;
            if (a.w < nearW) a = lerp(a, b, (nearW - a.w) / (b.w - a.w));
            else if (b.w < nearW) b = lerp(b, a, (nearW - b.w) / (a.w - b.w));
            dl->AddLine(toScreen(a), toScreen(b),
                        e.border ? IM_COL32(60, 255, 120, 235) : IM_COL32(255, 255, 255, 70), e.border ? 2.0f : 1.0f);
        }
        }   // showNavmesh

        // The path: a thick line strip through the waypoints, a dot at each corner, and the two end markers
        auto drawLine = [&](const float* p, const float* q, ImU32 color, float thickness) {
            Clip a = toClip(p), b = toClip(q);
            if (a.w < nearW && b.w < nearW) return;
            if (a.w < nearW) a = lerp(a, b, (nearW - a.w) / (b.w - a.w));
            else if (b.w < nearW) b = lerp(b, a, (nearW - b.w) / (a.w - b.w));
            dl->AddLine(toScreen(a), toScreen(b), color, thickness);
        };
        auto marker = [&](const float* p, ImU32 color, float radius) {
            Clip c = toClip(p);
            if (c.w < nearW) return;
            dl->AddCircleFilled(toScreen(c), radius, color);
            dl->AddCircle(toScreen(c), radius, IM_COL32(0, 0, 0, 255), 0, 2.0f);
        };
        for (size_t i = 0; i + 1 < navPath.count(); i++)
            drawLine(navPath.point(i), navPath.point(i + 1), IM_COL32(255, 215, 40, 255), 3.5f);
        for (size_t i = 1; i + 1 < navPath.count(); i++) marker(navPath.point(i), IM_COL32(255, 255, 255, 255), 3.5f);
        if (showAgentPaths) {
            world.each<AIAgent, Transform>([&](Entity e, AIAgent& ag, Transform& t) {
                if (!ag.walking()) return;
                const ImU32 color = e == selected ? IM_COL32(80, 220, 255, 255) : IM_COL32(80, 160, 220, 190);
                float from[3] = { t.position.x, t.position.y - ag.footOffset, t.position.z };
                for (size_t i = ag.waypoint; i < ag.path.size(); i++) {
                    const float to[3] = { ag.path[i].x, ag.path[i].y, ag.path[i].z };
                    drawLine(from, to, color, 2.0f);
                    from[0] = to[0]; from[1] = to[1]; from[2] = to[2];
                }
                const Vec3& goal = ag.path.back();
                const float g[3] = { goal.x, goal.y, goal.z };
                marker(g, color, 4.5f);
            });
        }
        if (selectionValid() && world.has<AIBrain>(selected)) {      // the selected brain's patrol loop, numbered
            const auto& pts = world.get<AIBrain>(selected).patrol;
            for (size_t i = 0; i < pts.size(); i++) {
                const float a[3] = { pts[i].x, pts[i].y, pts[i].z };
                const Vec3& nx = pts[(i + 1) % pts.size()];
                const float b2[3] = { nx.x, nx.y, nx.z };
                if (pts.size() > 1) drawLine(a, b2, IM_COL32(255, 160, 40, 170), 1.5f);
                marker(a, IM_COL32(255, 160, 40, 255), 5.0f);
            }
        }
        if (haveStart) marker(pathStart, IM_COL32(60, 220, 90, 255), 8.0f);
        if (haveEnd) marker(pathEnd, IM_COL32(235, 70, 60, 255), 8.0f);
    }

    // ---------- Hot reload (Step 6.4) ----------

    // Where glslc lives: $VULKAN_SDK/Bin (or bin) if that variable is set, otherwise whatever is on the PATH
    static std::string findGlslc() {
        std::string sdk;
#ifdef _WIN32
        char* buf = nullptr;
        size_t len = 0;
        if (_dupenv_s(&buf, &len, "VULKAN_SDK") == 0 && buf) { sdk = buf; free(buf); }
#else
        if (const char* v = std::getenv("VULKAN_SDK")) sdk = v;
#endif
        if (!sdk.empty()) {
            for (const char* dir : { "Bin", "bin" })
                for (const char* exe : { "glslc.exe", "glslc" }) {
                    std::filesystem::path p = std::filesystem::path(sdk) / dir / exe;
                    if (std::filesystem::exists(p)) return p.string();
                }
        }
        return "glslc";
    }

    // Runs glslc on one shader source. On failure `log` holds the compiler's messages.
    bool compileShader(const std::string& src, std::string& log) {
        const std::string out = src + ".spv";
        const std::string logFile = (std::filesystem::temp_directory_path() / "toyengine_glslc.log").string();
        std::string cmd = "\"" + findGlslc() + "\" \"" + src + "\" -o \"" + out + "\" > \"" + logFile + "\" 2>&1";
#ifdef _WIN32
        cmd = "\"" + cmd + "\"";       // cmd.exe strips the first and last quote of the line unless the whole thing is wrapped
#endif
        int rc = std::system(cmd.c_str());
        std::ifstream f(logFile);
        log.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
        if (log.size() > 1500) log.resize(1500), log += "...";
        return rc == 0;
    }

    // A shader source and/or its .spv changed on disk. Sources are recompiled with glslc (a change to just the
    // .spv means someone else, e.g. the CMake build, already compiled it); then the pipelines are rebuilt.
    void reloadShaders(const std::vector<std::string>& changed) {
        for (const std::string& p : changed) {
            if (p.size() > 4 && p.compare(p.size() - 4, 4, ".spv") == 0) continue;
            std::string log;
            if (!compileShader(p, log)) {
                shaderStatus = "Compile failed (" + p + "), keeping the old shaders:\n" + log;
                return;
            }
            shaderWatch.acknowledge(p + ".spv");       // we just wrote it; no second reload for our own output
        }
        std::string error;
        if (createGraphicsPipeline(error)) {
            shaderReloads++;
            shaderStatus = "Shaders reloaded (#" + std::to_string(shaderReloads) + ")";
        } else {
            shaderStatus = "Reload failed, keeping the old shaders: " + error;
        }
    }

    // A model file or one of its textures changed: import it again and swap the new mesh in under the same
    // entities. The old mesh stays in the GPU buffers (marked retired) until the app restarts; for an editor
    // session that is a few MB per reload.
    void reloadAsset(size_t k) {
        const std::string path = loadedAssets[k].first;
        MeshData md;
        std::string error;
        if (!loadGltf(path, md, error)) { assetStatus = "Reload of " + path + " failed, keeping the old one: " + error; return; }

        const uint32_t oldId = loadedAssets[k].second;
        const uint32_t newId = addMesh(md);
        meshes[oldId].retired = true;
        loadedAssets[k].second = newId;
        loadedAssetDeps[k] = md.dependencies;
        for (const std::string& d : md.dependencies) assetWatch.watch(d);
        world.each<MeshRenderer>([&](Entity, MeshRenderer& mr) { if (mr.meshId == oldId) mr.meshId = newId; });
        assetStatus = "Reloaded " + path + ": " + std::to_string(md.vertices.size()) + " vertices, " +
                      std::to_string(md.images.size()) + " texture(s)";
        for (const std::string& w : md.warnings) assetStatus += "\nWarning: " + w;
    }

    void pollHotReload() {
        if (!imguiEnabled || !hotReload) return;
        double now = glfwGetTime();
        if (now - lastHotPoll < 0.5) return;
        lastHotPoll = now;

        std::vector<std::string> shaderChanges = shaderWatch.poll();
        if (!shaderChanges.empty()) reloadShaders(shaderChanges);

        std::vector<std::string> assetChanges = assetWatch.poll();
        if (assetChanges.empty()) return;
        std::vector<bool> dirty(loadedAssets.size(), false);
        for (const std::string& p : assetChanges)
            for (size_t k = 0; k < loadedAssetDeps.size(); k++)
                for (const std::string& d : loadedAssetDeps[k]) if (d == p) dirty[k] = true;
        for (size_t k = 0; k < dirty.size(); k++) if (dirty[k]) reloadAsset(k);
    }

    // ---------- Scene save / load (Step 5.6) ----------

    // Mesh ids depend on import order, so scenes store names: "builtin:cube" or the asset's file path
    std::string meshKey(uint32_t id) const {
        if (id == 1) return "builtin:floor";
        for (const auto& l : loadedAssets) if (l.second == id) return l.first;
        return "builtin:cube";
    }

    int meshResolve(const std::string& key) {
        if (key == "builtin:cube") return 0;
        if (key == "builtin:floor") return 1;
        return importAsset(key);       // imports the file if this run hasn't loaded it yet; -1 if it can't
    }

    bool saveScene(const std::string& path) {
        sceneio::json doc = sceneio::newSceneDoc();
        doc["camera"] = { { "position", sceneio::toJson(cam.position) }, { "yaw", sceneio::f(cam.yaw) }, { "pitch", sceneio::f(cam.pitch) } };
        sceneio::json lights = sceneio::json::array();
        for (const EditorLight& l : lightCfg)
            lights.push_back({ { "position", sceneio::toJson({ l.pos[0], l.pos[1], l.pos[2] }) },
                               { "color", sceneio::toJson({ l.color[0], l.color[1], l.color[2] }) },
                               { "intensity", sceneio::f(l.intensity) } });
        doc["lights"] = { { "list", lights }, { "orbitLight2", light2Orbit } };
        doc["entities"] = sceneio::entitiesToJson(world, [this](uint32_t id) { return meshKey(id); });

        std::string err;
        if (!sceneio::writeSceneFile(path, doc, err)) { sceneStatus = "Save failed: " + err; return false; }
        sceneStatus = "Saved " + std::to_string(world.size()) + " entities to " + path;
        return true;
    }

    // All-or-nothing: the file is parsed into temporaries and only applied if every part was valid,
    // so a bad file leaves the current scene untouched.
    bool loadScene(const std::string& path) {
        sceneio::json doc;
        std::string err;
        if (!sceneio::readSceneFile(path, doc, err)) { sceneStatus = "Load failed: " + err; return false; }

        Registry fresh;
        std::vector<std::string> warnings;
        FlyCamera newCam = cam;
        EditorLight newLights[3];
        for (int i = 0; i < 3; i++) newLights[i] = lightCfg[i];
        bool newOrbit = light2Orbit;
        try {
            if (!doc.contains("entities")) throw std::runtime_error("no 'entities' section");
            sceneio::entitiesFromJson(doc["entities"], fresh, [this](const std::string& k) { return meshResolve(k); },
                                      (size_t)MAX_CUBES, warnings);
            if (auto c = doc.find("camera"); c != doc.end()) {
                if (auto p = c->find("position"); p != c->end()) newCam.position = sceneio::toVec3(*p);
                newCam.yaw = sceneio::numOr(*c, "yaw", newCam.yaw);
                newCam.pitch = sceneio::numOr(*c, "pitch", newCam.pitch);
            }
            if (auto l = doc.find("lights"); l != doc.end()) {
                if (auto list = l->find("list"); list != l->end()) {
                    if (!list->is_array()) throw std::runtime_error("'lights.list' must be an array");
                    for (size_t i = 0; i < list->size() && i < 3; i++) {
                        const sceneio::json& lj = (*list)[i];
                        if (lj.contains("position")) { Vec3 v = sceneio::toVec3(lj["position"]); newLights[i].pos[0] = v.x; newLights[i].pos[1] = v.y; newLights[i].pos[2] = v.z; }
                        if (lj.contains("color")) { Vec3 v = sceneio::toVec3(lj["color"]); newLights[i].color[0] = v.x; newLights[i].color[1] = v.y; newLights[i].color[2] = v.z; }
                        newLights[i].intensity = sceneio::numOr(lj, "intensity", newLights[i].intensity);
                    }
                }
                newOrbit = l->value("orbitLight2", newOrbit);
            }
        } catch (const std::exception& ex) {
            sceneStatus = std::string("Load failed: ") + ex.what();
            return false;
        }

        world = std::move(fresh);
        selected = NULL_ENTITY;
        game = GameState{};
        gizmoActive = false;
        demoScene = false;
        cam = newCam;
        for (int i = 0; i < 3; i++) lightCfg[i] = newLights[i];
        light2Orbit = newOrbit;
        cubeCount = (int)world.size();
        cubeSerial = (int)world.size();
        lastSimTime = glfwGetTime();
        sceneStatus = "Loaded " + std::to_string(world.size()) + " entities from " + path;
        for (const std::string& w : warnings) sceneStatus += "\nWarning: " + w;
        return true;
    }

    // A handful of named cubes of different sizes and spin speeds: something to click on.
    void spawnDemoScene() {
        struct Demo { const char* name; Vec3 pos; float scale; float spin; };
        const Demo demo[] = {
            { "Spinner",     {  0.00f,  0.00f, -1.5f }, 1.0f,  1.0f },   // the original cube
            { "Small Cube",  { -0.70f, -0.35f, -1.0f }, 0.3f,  2.5f },
            { "Mini",        {  0.75f, -0.40f, -0.8f }, 0.2f,  3.0f },
            { "Big Block",   {  0.70f, -0.15f, -3.0f }, 0.7f,  0.4f },
            { "Far Cube",    { -0.50f, -0.30f, -3.6f }, 0.4f, -1.2f },
        };
        for (const Demo& d : demo) {
            Entity e = world.create();
            Transform t;
            t.position = d.pos;
            t.scale = d.scale;
            world.add<Transform>(e, t);
            world.add<Spin>(e, Spin{ d.spin });
            world.add<MeshRenderer>(e, MeshRenderer{ 0, 0 });
            world.add<Name>(e, Name{ d.name });
        }
        cubeCount = (int)world.size();
        lastSimTime = glfwGetTime();
    }

    // The CPU side of a frame: run the systems, build the draw list, upload it.
    void simulateAndUpload(uint32_t frame) {
        double now = glfwGetTime();
        float dt = (float)std::min(now - lastSimTime, 0.1);   // clamp so a stall can't make cubes jump
        lastSimTime = now;
        if (simPaused || gizmoActive) dt = 0.0f;   // the scene holds still while a gizmo is being dragged
        simTime += dt;

        double t0 = glfwGetTime();
        spinSystem(world, dt);
        bobSystem(world, simTime);
        // Once a run is won or lost, the agents freeze (the decorations keep spinning)
        const float aiDt = (runAI && game.phase != GameState::Playing) ? 0.0f : dt;
        if (runAI) { healthSystem(world, aiDt); behaviorSystem(world, aiDt, combatTree, navPathFn); }
        pathFollowSystem(world, aiDt);
        if (runAI) playerKeyboardMove(aiDt);
        if (runAI) updateGame(world, game, dt);
        double t1 = glfwGetTime();
        buildDrawList(world, drawList, selected);
        drawnCubes = (uint32_t)drawList.size();

        // Group by mesh so each mesh is one instanced draw. When everything is the same mesh (the common
        // benchmark case) the list is already sorted and this is just one pass over it.
        for (DrawItem& d : drawList) if (d.meshId >= meshes.size()) d.meshId = 0;     // unknown mesh id -> cube
        auto byMesh = [](const DrawItem& a, const DrawItem& b) { return a.meshId < b.meshId; };
        if (!std::is_sorted(drawList.begin(), drawList.end(), byMesh))
            std::stable_sort(drawList.begin(), drawList.end(), byMesh);
        drawRanges.clear();
        drawnTriangles = 0;
        for (uint32_t i = 0; i < drawnCubes; i++) {
            if (drawRanges.empty() || drawRanges.back().mesh != drawList[i].meshId)
                drawRanges.push_back({ drawList[i].meshId, i, 1 });
            else
                drawRanges.back().count++;
        }
        for (const DrawRange& r : drawRanges) drawnTriangles += r.count * (meshes[r.mesh].indexCount / 3);

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

        // Set 1: the material (base color texture), bound per submesh
        VkDescriptorSetLayoutBinding mb{};
        mb.binding = 0;
        mb.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        mb.descriptorCount = 1;
        mb.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        VkDescriptorSetLayoutCreateInfo minfo{};
        minfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        minfo.bindingCount = 1;
        minfo.pBindings = &mb;
        if (vkCreateDescriptorSetLayout(device, &minfo, nullptr, &materialSetLayout) != VK_SUCCESS)
            throw std::runtime_error("Failed to create material set layout");
    }

    // ---------- Textures & materials ----------

    VkCommandBuffer beginOneTime() {
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
        return cmd;
    }

    void endOneTime(VkCommandBuffer cmd) {
        vkEndCommandBuffer(cmd);
        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &cmd;
        vkQueueSubmit(graphicsQueue, 1, &submit, VK_NULL_HANDLE);
        vkQueueWaitIdle(graphicsQueue);
        vkFreeCommandBuffers(device, commandPool, 1, &cmd);
    }

    // Uploads RGBA8 pixels as a sampled sRGB texture (the GPU converts to linear when sampling) and returns its index
    uint32_t createTexture(const uint8_t* rgba, uint32_t w, uint32_t h) {
        VkDeviceSize size = (VkDeviceSize)w * h * 4;
        VkBuffer staging;
        VkDeviceMemory stagingMem;
        createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging, stagingMem);
        void* mapped;
        vkMapMemory(device, stagingMem, 0, size, 0, &mapped);
        memcpy(mapped, rgba, (size_t)size);
        vkUnmapMemory(device, stagingMem);

        // Mip chain: each level is half the size of the one above, down to 1x1. Built on the GPU with
        // blits, which needs linear-blit support for the format (virtually every GPU has it).
        VkFormatProperties fp;
        vkGetPhysicalDeviceFormatProperties(physicalDevice, VK_FORMAT_R8G8B8A8_SRGB, &fp);
        const VkFormatFeatureFlags needed = VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT |
                                            VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT;
        const bool canBlit = (fp.optimalTilingFeatures & needed) == needed;

        GpuTexture tex;
        tex.mipLevels = 1;
        if (canBlit) {
            uint32_t largest = std::max(w, h);
            while (largest > 1) { largest >>= 1; tex.mipLevels++; }
        }
        const uint32_t mips = tex.mipLevels;
        VkImageCreateInfo ii{};
        ii.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ii.imageType = VK_IMAGE_TYPE_2D;
        ii.extent = { w, h, 1 };
        ii.mipLevels = mips;
        ii.arrayLayers = 1;
        ii.format = VK_FORMAT_R8G8B8A8_SRGB;
        ii.tiling = VK_IMAGE_TILING_OPTIMAL;
        ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        ii.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        ii.samples = VK_SAMPLE_COUNT_1_BIT;
        ii.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateImage(device, &ii, nullptr, &tex.image) != VK_SUCCESS)
            throw std::runtime_error("Failed to create texture image");
        VkMemoryRequirements req;
        vkGetImageMemoryRequirements(device, tex.image, &req);
        VkMemoryAllocateInfo alloc{};
        alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        alloc.allocationSize = req.size;
        alloc.memoryTypeIndex = findMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (vkAllocateMemory(device, &alloc, nullptr, &tex.memory) != VK_SUCCESS)
            throw std::runtime_error("Failed to allocate texture memory");
        vkBindImageMemory(device, tex.image, tex.memory, 0);

        // UNDEFINED -> TRANSFER_DST, copy the pixels, TRANSFER_DST -> SHADER_READ_ONLY
        VkCommandBuffer cmd = beginOneTime();
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = tex.image;
        barrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, mips, 0, 1 };   // every level: ready to be written
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                             0, nullptr, 0, nullptr, 1, &barrier);
        VkBufferImageCopy copy{};
        copy.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        copy.imageExtent = { w, h, 1 };
        vkCmdCopyBufferToImage(cmd, staging, tex.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
        // Level 0 now holds the pixels. For each next level: finish the previous one (DST -> SRC), blit it down to
        // half size, then hand the previous level over to the shader.
        int32_t mw = (int32_t)w, mh = (int32_t)h;
        for (uint32_t i = 1; i < mips; i++) {
            barrier.subresourceRange.baseMipLevel = i - 1;
            barrier.subresourceRange.levelCount = 1;
            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                                 0, nullptr, 0, nullptr, 1, &barrier);

            int32_t nw = std::max(mw / 2, 1), nh = std::max(mh / 2, 1);
            VkImageBlit blit{};
            blit.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, i - 1, 0, 1 };
            blit.srcOffsets[1] = { mw, mh, 1 };
            blit.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, i, 0, 1 };
            blit.dstOffsets[1] = { nw, nh, 1 };
            vkCmdBlitImage(cmd, tex.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           tex.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);

            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                                 0, nullptr, 0, nullptr, 1, &barrier);
            mw = nw;
            mh = nh;
        }
        // The last level (which is level 0 for a 1x1 texture) was only ever written to
        barrier.subresourceRange.baseMipLevel = mips - 1;
        barrier.subresourceRange.levelCount = 1;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                             0, nullptr, 0, nullptr, 1, &barrier);
        endOneTime(cmd);
        vkDestroyBuffer(device, staging, nullptr);
        vkFreeMemory(device, stagingMem, nullptr);

        VkImageViewCreateInfo vi{};
        vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vi.image = tex.image;
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = ii.format;
        vi.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, mips, 0, 1 };
        if (vkCreateImageView(device, &vi, nullptr, &tex.view) != VK_SUCCESS)
            throw std::runtime_error("Failed to create texture view");
        textures.push_back(tex);
        return (uint32_t)textures.size() - 1;
    }

    // A material = a descriptor set pointing at one texture. Returns its index (0, the default, if the pool is full).
    uint32_t createMaterial(uint32_t textureIndex) { return createMaterial(textureIndex, MaterialProps{}); }

    uint32_t createMaterial(uint32_t textureIndex, const MaterialProps& props) {
        if (materialSets.size() >= MAX_MATERIALS) return 0;
        VkDescriptorSetAllocateInfo alloc{};
        alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc.descriptorPool = materialPool;
        alloc.descriptorSetCount = 1;
        alloc.pSetLayouts = &materialSetLayout;
        VkDescriptorSet set;
        if (vkAllocateDescriptorSets(device, &alloc, &set) != VK_SUCCESS)
            throw std::runtime_error("Failed to allocate material descriptor set");
        VkDescriptorImageInfo img{ textureSampler, textures[textureIndex].view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkWriteDescriptorSet w{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = set;
        w.dstBinding = 0;
        w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w.descriptorCount = 1;
        w.pImageInfo = &img;
        vkUpdateDescriptorSets(device, 1, &w, 0, nullptr);
        materialSets.push_back(set);
        materialTexture.push_back(textureIndex);
        materialProps.push_back(props);
        return (uint32_t)materialSets.size() - 1;
    }

    // Sampler, descriptor pool, and material 0 (white). Needs the command pool (texture uploads).
    // (Re)creates the shared sampler from filterMode / anisotropy. All three modes are a sampler setting, not a
    // texture one, so switching is instant and the same texture data shows the difference.
    void createSampler() {
        VkSamplerCreateInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        const bool nearest = filterMode == 0;
        si.magFilter = si.minFilter = nearest ? VK_FILTER_NEAREST : VK_FILTER_LINEAR;
        si.mipmapMode = nearest ? VK_SAMPLER_MIPMAP_MODE_NEAREST : VK_SAMPLER_MIPMAP_MODE_LINEAR;
        si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        si.minLod = 0.0f;
        si.maxLod = filterMode == 2 ? VK_LOD_CLAMP_NONE : 0.0f;    // maxLod 0 = never leave the full-size level
        si.anisotropyEnable = (maxAnisotropy > 1.0f && filterMode != 0 && anisotropy > 1) ? VK_TRUE : VK_FALSE;
        si.maxAnisotropy = std::min((float)anisotropy, maxAnisotropy);
        if (vkCreateSampler(device, &si, nullptr, &textureSampler) != VK_SUCCESS)
            throw std::runtime_error("Failed to create sampler");
    }

    // Swap the sampler while the app runs: wait for the GPU, then point every material at the new one
    void changeSampler() {
        vkDeviceWaitIdle(device);
        vkDestroySampler(device, textureSampler, nullptr);
        createSampler();
        for (size_t i = 0; i < materialSets.size(); i++) {
            VkDescriptorImageInfo img{ textureSampler, textures[materialTexture[i]].view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
            VkWriteDescriptorSet w{};
            w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet = materialSets[i];
            w.dstBinding = 0;
            w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w.descriptorCount = 1;
            w.pImageInfo = &img;
            vkUpdateDescriptorSets(device, 1, &w, 0, nullptr);
        }
    }

    void createTextureResources() {
        createSampler();

        VkDescriptorPoolSize ps{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, MAX_MATERIALS };
        VkDescriptorPoolCreateInfo pi{};
        pi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pi.poolSizeCount = 1;
        pi.pPoolSizes = &ps;
        pi.maxSets = MAX_MATERIALS;
        if (vkCreateDescriptorPool(device, &pi, nullptr, &materialPool) != VK_SUCCESS)
            throw std::runtime_error("Failed to create material descriptor pool");

        const uint8_t white[4] = { 255, 255, 255, 255 };
        createMaterial(createTexture(white, 1, 1));    // material 0
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
        // Benchmarks use the same fixed camera as always. Otherwise: the free-fly camera,
        // or (optional demo mode) the gentle automatic sway around the scene.
        const Vec3 target = { 0, -0.3f, -3.0f };
        Vec3 eye = { 0, 0.8f, 1.0f };
        Mat4 view;
        if (benchMode) {
            view = Mat4::lookAt(eye, target, { 0, 1, 0 });
        } else if (camAutoSway) {
            camTime += (float)frameDt;
            float angle = 0.35f * std::sin(camTime * 0.5f);
            float dx = eye.x - target.x, dz = eye.z - target.z;
            eye.x = target.x + dx * std::cos(angle) + dz * std::sin(angle);
            eye.z = target.z - dx * std::sin(angle) + dz * std::cos(angle);
            view = Mat4::lookAt(eye, target, { 0, 1, 0 });
        } else {
            eye = cam.position;
            view = cam.view();
        }
        Mat4 proj = Mat4::perspective(3.14159265f / 4.0f,
                                      (float)swapchainExtent.width / swapchainExtent.height, 0.1f, 100.0f);

        // OpenGL-style clip space -> Vulkan clip space:
        //   Y is flipped (Vulkan's Y points down), and depth is remapped from [-1,1] to [0,1]
        Mat4 fix = Mat4::identity();
        fix.m[1][1] = -1.0f;
        fix.m[2][2] = 0.5f;
        fix.m[3][2] = 0.5f;

        lastView = view;
        lastProj = proj;
        pickInvViewProj = (proj * view).inverse();   // for turning mouse clicks into world-space rays
        havePickMatrix = true;

        SceneUBO ubo{};
        ubo.viewProj = fix * proj * view;
        ubo.cameraPos[0] = eye.x; ubo.cameraPos[1] = eye.y; ubo.cameraPos[2] = eye.z;

        // Lights come from lightCfg (editable in the Inspector). Light 2 can orbit its position,
        // so you can watch the lighting respond to movement.
        for (int i = 0; i < 3; i++) {
            const EditorLight& l = lightCfg[i];
            float px = l.pos[0], py = l.pos[1], pz = l.pos[2];
            if (i == 2 && light2Orbit) { px += std::sin(t * 1.2f) * 1.3f; pz += std::cos(t * 1.2f) * 1.3f; }
            ubo.lights[i] = { { px, py, pz, 1.0f }, { l.color[0], l.color[1], l.color[2], l.intensity } };
        }
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
        if (!benchMode) { updateCamera(); handlePicking(); }

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
