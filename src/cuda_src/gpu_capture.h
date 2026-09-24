#ifndef _GPU_CAPTURE_H
#define _GPU_CAPTURE_H

// Opt-in golden-vector capture for the GPU wrappers (CLAUDE.md 4.1).
//
// Off unless TORTOISE_GPU_CAPTURE names an output directory, in which case each
// wrapper writes its inputs, parameters and outputs to <dir>/<op>.<seq>/ so the
// operation can be replayed in isolation by the gpu_replay tool.
//   TORTOISE_GPU_CAPTURE=<dir>   enable, write records here
//   TORTOISE_GPU_CAPTURE_N=<n>   keep the first n records per op (default 2)
//
// When disabled every call below is a predictable-branch no-op.

#include "cuda_image.h"
#include <string>
#include <vector>
#include <memory>

namespace gpucap
{

bool Enabled();

// Record schema version. Bump on any incompatible change to record.json;
// gpu_replay refuses records it does not understand.
const int SCHEMA_VERSION = 2;

class Rec
{
public:
    // `variant` gives an op its own quota per shape class (e.g. scalar vs
    // 3-component). Without it, "first N calls per op" silently biases capture
    // toward whichever pipeline phase runs first: DIFFPREP smooths scalar
    // images, DRBUDDI smooths vector fields, and only the former was ever
    // captured.
    explicit Rec(const std::string &op, int variant = -1);
    ~Rec();

    bool on() const { return active; }

    Rec &param(const std::string &key, double value);
    Rec &param(const std::string &key, const std::string &value);
    Rec &param(const std::string &key, const std::vector<double> &value);
    Rec &param(const std::string &key, float3 v);

    Rec &in (const std::string &name, const CUDAIMAGE::Pointer img);
    // Geometry-only argument: an image used solely as a sampling grid (its
    // buffer may never be allocated). Records metadata, no data blob.
    Rec &geom(const std::string &name, const CUDAIMAGE::Pointer img);
    Rec &out(const std::string &name, const CUDAIMAGE::Pointer img);
    Rec &scalar(const std::string &name, double value);   // e.g. returned metric

    void save();

private:
    Rec &addTensor(const char *role, const std::string &name, const CUDAIMAGE::Pointer img);

    struct Impl;
    std::shared_ptr<Impl> d;
    bool active{false};
};

} // namespace gpucap

#endif
