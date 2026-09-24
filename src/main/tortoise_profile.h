#ifndef _TORTOISE_PROFILE_H
#define _TORTOISE_PROFILE_H

// Coarse wall-clock stage timers. Each scope prints one line to stderr on exit:
//     [PROFILE] <name> <seconds>
// Summing them per run attributes wall time to named stages (plan_optimize.md M0).
// Measurement only: no arithmetic, no allocation, no synchronisation.

#include <chrono>
#include <cstdio>
#include <string>
#include <atomic>
#include <sys/resource.h>
#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/task_info.h>
#endif

// Apple's phys_footprint - the number Activity Monitor calls "Memory". Unlike RSS it
// INCLUDES IOKit/IOAccelerator allocations, i.e. GPU buffers, which is exactly what a
// GPU backend comparison needs: RSS can under-count a backend that keeps buffers in
// device-private storage and over-count one that keeps them host-visible, so two
// backends can look different purely from accounting. Returns 0 where unavailable.
// High-water of phys_footprint across every sample taken so far. The raw value is
// instantaneous, and ProfileScope samples at entry and exit - after a stage has already
// released its buffers - so reporting the raw exit value as a "peak" understates any
// backend that peaks mid-stage, which would invert a cross-backend comparison.
inline std::atomic<size_t> &TortoisePhysFootprintPeak()
{
    static std::atomic<size_t> peak{0};
    return peak;
}

inline size_t TortoisePhysFootprintBytes()
{
#ifdef __APPLE__
    task_vm_info_data_t info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if(task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &count) == KERN_SUCCESS)
    {
        std::atomic<size_t> &pk = TortoisePhysFootprintPeak();
        const size_t cur = (size_t)info.phys_footprint;
        size_t prev = pk.load();
        while(cur > prev && !pk.compare_exchange_weak(prev, cur)) {}
        return pk.load();
    }
#endif
    return 0;
}

// Peak resident set for the process so far. ru_maxrss is a HIGH-WATER mark and never
// decreases, so the growth across a scope tells you whether that stage raised the
// peak - which is exactly the question, and needs no sampling thread.
// Units differ by platform: bytes on macOS, kilobytes on Linux.
inline size_t TortoisePeakRSSBytes()
{
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
#ifdef __APPLE__
    return (size_t)ru.ru_maxrss;
#else
    return (size_t)ru.ru_maxrss * 1024;
#endif
}

struct ProfileScope
{
    std::string name;
    std::chrono::steady_clock::time_point t0;
    size_t rss0;

    explicit ProfileScope(const std::string &n)
        : name(n), t0(std::chrono::steady_clock::now()), rss0(TortoisePeakRSSBytes())
    { TortoisePhysFootprintBytes(); }

    ~ProfileScope()
    {
        double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const size_t rss1 = TortoisePeakRSSBytes();
        const size_t fp1  = TortoisePhysFootprintBytes();
        // Fields are APPENDED, never reordered: existing parsers read $2 as the name
        // and $3 as the seconds. peak_MiB is the process high-water at scope exit;
        // grew_MiB is how much THIS stage raised it, which is what localises a
        // regression to a stage instead of only to a run.
        fprintf(stderr, "[PROFILE] %s %.3f peak_MiB %.1f grew_MiB %.1f foot_MiB %.1f\n",
                name.c_str(), s, rss1 / 1048576.0,
                (rss1 > rss0 ? rss1 - rss0 : 0) / 1048576.0,
                fp1 / 1048576.0);
        fflush(stderr);
    }
};

#define TORTOISE_PROFILE(nm) ProfileScope profile_scope_(nm)

#endif
