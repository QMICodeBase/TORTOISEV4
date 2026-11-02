/* Minimal Vulkan device probe - no vulkan headers needed.
 * Confirms which physical devices Dawn's Vulkan backend would see, and whether
 * VK_ICD_FILENAMES successfully hides the AMD integrated GPU.
 * Reads only the leading fields of VkPhysicalDeviceProperties, whose layout is
 * fixed by the spec: apiVersion, driverVersion, vendorID, deviceID, deviceType,
 * deviceName[256].
 */
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef void *VkInstance;
typedef void *VkPhysicalDevice;
typedef void *(*PFN_vkGetInstanceProcAddr)(VkInstance, const char *);
typedef int (*PFN_vkCreateInstance)(const void *, const void *, VkInstance *);
typedef int (*PFN_vkEnumeratePhysicalDevices)(VkInstance, uint32_t *, VkPhysicalDevice *);
typedef void (*PFN_vkGetPhysicalDeviceProperties)(VkPhysicalDevice, void *);

struct AppInfo {
    uint32_t sType; const void *pNext;
    const char *pApplicationName; uint32_t applicationVersion;
    const char *pEngineName; uint32_t engineVersion; uint32_t apiVersion;
};
struct InstInfo {
    uint32_t sType; const void *pNext; uint32_t flags;
    const struct AppInfo *pApplicationInfo;
    uint32_t enabledLayerCount; const char *const *ppEnabledLayerNames;
    uint32_t enabledExtensionCount; const char *const *ppEnabledExtensionNames;
};

struct PropsHead {
    uint32_t apiVersion, driverVersion, vendorID, deviceID, deviceType;
    char deviceName[256];
};

static const char *TypeName(uint32_t t)
{
    switch (t) {
    case 0: return "other";
    case 1: return "integrated";
    case 2: return "DISCRETE";
    case 3: return "virtual";
    case 4: return "cpu";
    default: return "?";
    }
}

int main(void)
{
    void *lib = dlopen("libvulkan.so.1", RTLD_NOW);
    if (!lib) { printf("no libvulkan.so.1: %s\n", dlerror()); return 2; }

    PFN_vkGetInstanceProcAddr gipa = (PFN_vkGetInstanceProcAddr)dlsym(lib, "vkGetInstanceProcAddr");
    PFN_vkCreateInstance createInstance = (PFN_vkCreateInstance)gipa(NULL, "vkCreateInstance");
    if (!createInstance) { printf("no vkCreateInstance\n"); return 2; }

    struct AppInfo app = {0};
    app.sType = 0;                    /* VK_STRUCTURE_TYPE_APPLICATION_INFO */
    app.pApplicationName = "tortoise-vkprobe";
    app.apiVersion = (1u << 22) | (1u << 12);  /* 1.1 */
    struct InstInfo ii = {0};
    ii.sType = 1;                     /* VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO */
    ii.pApplicationInfo = &app;

    VkInstance inst = NULL;
    int rc = createInstance(&ii, NULL, &inst);
    if (rc != 0) { printf("vkCreateInstance failed: %d\n", rc); return 2; }

    PFN_vkEnumeratePhysicalDevices enumerate =
        (PFN_vkEnumeratePhysicalDevices)gipa(inst, "vkEnumeratePhysicalDevices");
    PFN_vkGetPhysicalDeviceProperties getProps =
        (PFN_vkGetPhysicalDeviceProperties)gipa(inst, "vkGetPhysicalDeviceProperties");

    uint32_t n = 0;
    enumerate(inst, &n, NULL);
    if (n == 0) { printf("no Vulkan physical devices visible\n"); return 1; }
    VkPhysicalDevice devs[16];
    if (n > 16) n = 16;
    enumerate(inst, &n, devs);

    printf("Vulkan physical devices visible: %u\n", n);
    int nvidia_discrete = 0;
    for (uint32_t i = 0; i < n; i++) {
        char buf[4096];
        memset(buf, 0, sizeof(buf));
        getProps(devs[i], buf);
        struct PropsHead *p = (struct PropsHead *)buf;
        printf("  [%u] vendor=0x%04X type=%-10s  %s\n", i, p->vendorID, TypeName(p->deviceType),
               p->deviceName);
        if (p->vendorID == 0x10DE && p->deviceType == 2) nvidia_discrete++;
    }
    printf("\nNVIDIA discrete adapters: %d\n", nvidia_discrete);
    printf("selection policy would %s\n",
           nvidia_discrete == 1 ? "SUCCEED (exactly one NVIDIA discrete adapter)"
                                : (nvidia_discrete == 0 ? "ABORT (no NVIDIA discrete adapter)"
                                                        : "need a tie-break (several)"));
    return nvidia_discrete == 1 ? 0 : 1;
}
