#ifndef _GPU_RECORD_H
#define _GPU_RECORD_H

// Backend-agnostic golden-vector record I/O, shared by gpu_replay (CUDA) and
// webgpu_replay. Keeping one loader means both backends agree on what a record
// means and on which records are rejected.

#include "defines.h"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

// Canonical record format version. Must equal gpucap::SCHEMA_VERSION in
// cuda_src/gpu_capture.h; gpu_replay_main.cxx static_asserts that it does (it is
// the one translation unit where both are visible). Declared here so the WebGPU
// replay tool can read records without pulling in any CUDA header.
static const int RECORD_SCHEMA_VERSION = 2;

// ---------------------------------------------------------------- record I/O

struct Tensor
{
    std::string name, role, file, digest;
    std::vector<int> dims;
    int ncomp{1};
    std::vector<double> spacing, origin, direction;
    std::vector<float> data;
};

struct Record
{
    std::string dir, op;
    // "exact" when the expected output was computed analytically rather than
    // captured from CUDA - such records get the tight tolerance for every op,
    // because CUDA's own sampler inaccuracy is not in play.
    std::string reference;
    // Optional: this record is KNOWN to differ from the captured reference because
    // the REFERENCE is the incorrect implementation. Declared in record.json as
    //   "expected_divergence": {"reason": "...", "rel": 0.001651, "rel_tol": 0.05}
    // The replay tool then requires the divergence to be present AND of the stated
    // size: agreeing with the reference means the backend reproduced the reference's
    // bug, which is a failure. See CLAUDE.md 0.0b.
    bool expected_divergence{false};
    double expected_rel{0.0};
    double expected_rel_tol{0.05};
    std::string expected_reason;
    int seq{0};
    json params, scalars;
    std::vector<Tensor> tensors;

    const Tensor *find(const std::string &role, const std::string &name) const
    {
        for(size_t i = 0; i < tensors.size(); i++)
            if(tensors[i].role == role && tensors[i].name == name)
                return &tensors[i];
        return nullptr;
    }
};

inline std::string Digest(const std::vector<float> &v)
{
    uint64_t h = 1469598103934665603ULL;
    const unsigned char *p = (const unsigned char *)v.data();
    const size_t n = v.size() * sizeof(float);
    for(size_t i = 0; i < n; i++) { h ^= p[i]; h *= 1099511628211ULL; }
    char buf[32];
    snprintf(buf, sizeof(buf), "%016lx", (unsigned long)h);
    return std::string(buf);
}

// Throws std::runtime_error with a specific reason on any unusable record.
inline Record LoadRecord(const std::string &dir)
{
    Record r;
    r.dir = dir;

    std::ifstream jf((dir + "/record.json").c_str());
    if(!jf.good())
        throw std::runtime_error("no record.json in " + dir);

    json j;
    try { jf >> j; }
    catch(std::exception &e) { throw std::runtime_error("record.json is not valid JSON: " + std::string(e.what())); }

    if(!j.contains("schema_version"))
        throw std::runtime_error("record.json has no schema_version");
    const int v = j["schema_version"];
    if(v != RECORD_SCHEMA_VERSION)
        throw std::runtime_error("record schema_version " + std::to_string(v) +
                                 " but this build understands " + std::to_string(RECORD_SCHEMA_VERSION));

    r.op        = j.value("op", "");
    r.reference = j.value("reference", "cuda");
    if(j.contains("expected_divergence"))
    {
        const json &e = j["expected_divergence"];
        if(!e.contains("rel"))
            throw std::runtime_error("expected_divergence has no 'rel' - refusing to "
                                     "treat an unquantified divergence as expected");
        r.expected_divergence = true;
        r.expected_rel        = e["rel"].get<double>();
        r.expected_rel_tol    = e.value("rel_tol", 0.05);
        r.expected_reason     = e.value("reason", std::string("(no reason given)"));
    }
    r.seq     = j.value("seq", 0);
    r.params  = j.value("params",  json::object());
    r.scalars = j.value("scalars", json::object());
    if(r.op.empty())
        throw std::runtime_error("record.json has no op");

    for(auto &t : j["tensors"])
    {
        if(t.value("unreadable", false))
            throw std::runtime_error("record contains an unreadable tensor '" +
                                     t.value("name", "?") + "' - capture was incomplete");
        Tensor tt;
        tt.role      = t["role"];
        tt.name      = t["name"];
        tt.file      = t.value("file", "");
        tt.dims      = t["dims"].get<std::vector<int> >();
        tt.ncomp     = t["ncomp"];
        tt.spacing   = t["spacing"].get<std::vector<double> >();
        tt.origin    = t["origin"].get<std::vector<double> >();
        tt.direction = t["direction"].get<std::vector<double> >();
        tt.digest    = t.value("digest", "");

        if(tt.role == "geom")     // sampling grid only: metadata, no data
        {
            r.tensors.push_back(tt);
            continue;
        }

        const size_t expect = (size_t)tt.dims[0] * tt.dims[1] * tt.dims[2] * tt.ncomp;
        std::ifstream bf((dir + "/" + tt.file).c_str(), std::ios::binary | std::ios::ate);
        if(!bf.good())
            throw std::runtime_error("missing blob " + tt.file);
        const size_t have = (size_t)bf.tellg() / sizeof(float);
        if(have != expect)
            throw std::runtime_error("blob " + tt.file + " is truncated: " +
                                     std::to_string(have) + " floats, expected " + std::to_string(expect));
        bf.seekg(0);
        tt.data.resize(expect);
        bf.read((char *)tt.data.data(), expect * sizeof(float));
        if(!tt.digest.empty() && Digest(tt.data) != tt.digest)
            throw std::runtime_error("blob " + tt.file + " fails its digest - corrupted");
        r.tensors.push_back(tt);
    }
    return r;
}


// ---------------------------------------------------------------- comparison

struct Diff
{
    size_t n{0}, ndiff{0}, nexceed{0}, nnonfinite{0};
    double maxabs{0}, maxref{0}, rel{0};
    double exceed_frac{0};
    bool   exceed_valid{false};   // did the exceedance pass actually run?

    // PER-ELEMENT statistics. `rel` above is max|a-b| / max|b| - a single global
    // allowance set by the largest voxel in the volume, NOT a per-element relative
    // error. On a metric update field whose max is 1.9e7, tol=1e-5 permits an
    // absolute error of 194 at EVERY element, and ~52% of elements (34% of them
    // nonzero) sit below that. A port returning zero for all of them would pass.
    // These fields measure the port against a scale-aware criterion instead:
    //   |a-b| <= tol * (|b| + rms(ref))
    //
    // HONEST LIMITS - corrected 2026-08-20 after an audit measured them.
    // An earlier comment called rms "a robust scale that a single outlier cannot
    // inflate". That is FALSE: rms is dominated by outliers (one value M among n
    // elements contributes M/sqrt(n), so the metric field's single 1.8e7 peak sets
    // rms ~ 5.4e5 on its own). The measured effect over all 150 output tensors of
    // the fast suite is a median band shrink of only 1.08x versus the
    // max-normalised gate, and on 50 of 150 tensors the per-element band is LOOSER.
    // On ComputeMetric_MSJac.0 the fraction of nonzero elements an all-zero return
    // could hide moves 34.0 % -> 33.7 %.
    //
    // So this materially helps the small/elementwise ops and barely helps the metric
    // kernels. It is a real tightening, not the 3.6x improvement previously claimed
    // (that figure assumed an attacker aiming at the OLD bound). Stated plainly:
    // the metric kernels still have no strong per-element gate, and their evidence
    // is the expression-by-expression source review, not this number.
    double rms_ref{0};
    size_t nexceed_elem{0};
    double exceed_elem_frac{0};
    double worst_elem_rel{0};     // max |a-b| / (|b| + rms)
};

// `tol` is used only to count how many elements exceed it; the caller decides
// what to do with that. Two passes because the scale is not known until the
// reference has been swept.
inline Diff Compare(const std::vector<float> &got, const std::vector<float> &ref,
                    double tol = 0.0)
{
    // A wrongly-shaped result must fail outright. Comparing the common prefix
    // would let a wrapper that returns the wrong dimensions report PASS - the one
    // thing this harness must never do.
    if(got.size() != ref.size())
        throw std::runtime_error("output size mismatch: got " + std::to_string(got.size()) +
                                 " floats, expected " + std::to_string(ref.size()));
    Diff d;
    d.n = ref.size();
    for(size_t i = 0; i < ref.size(); i++)
    {
        const double a = got[i], b = ref[i];
        // NaN needs explicit handling only because std::max(x, NaN) returns x, so a
        // NaN difference would never raise maxabs and the comparison would PASS.
        // This is about the COMPARISON being correct, not about NaN being bad: if
        // both backends produce NaN they agree, and that is a match. The reference
        // kernels contain no NaN checks and neither do the ported ones.
        if(!std::isfinite(a) || !std::isfinite(b))
        {
            const bool same = (std::isnan(a) && std::isnan(b)) || (a == b);   // ±Inf compares equal
            if(!same)
            {
                d.nnonfinite++;
                d.ndiff++;
            }
            continue;
        }
        if(a != b) d.ndiff++;
        d.maxabs = std::max(d.maxabs, std::fabs(a - b));
        d.maxref = std::max(d.maxref, std::fabs(b));
    }
    d.rel = d.maxref > 0 ? d.maxabs / d.maxref : d.maxabs;

    // Per-element pass. Reported always; gating on it is a separate decision.
    {
        double sq = 0.0;
        for(size_t i = 0; i < ref.size(); i++)
            if(std::isfinite(ref[i])) sq += (double)ref[i] * (double)ref[i];
        d.rms_ref = ref.empty() ? 0.0 : std::sqrt(sq / (double)ref.size());
        for(size_t i = 0; i < ref.size(); i++)
        {
            const double a = got[i], b = ref[i];
            if(!std::isfinite(a) || !std::isfinite(b)) continue;
            const double denom = std::fabs(b) + d.rms_ref;
            if(denom <= 0.0) continue;
            const double r = std::fabs(a - b) / denom;
            if(r > d.worst_elem_rel) d.worst_elem_rel = r;
            if(tol > 0 && r > tol) d.nexceed_elem++;
        }
        d.exceed_elem_frac = d.n ? (double)d.nexceed_elem / (double)d.n : 0.0;
    }

    // Runs whenever a tolerance is in play. It must NOT be skipped when maxref==0
    // (an all-zero reference): exceed_frac would stay 0 and the outlier rule in
    // check() would then clear any failure, however large. With maxref==0 the
    // limit is 0, so any nonzero difference counts - which is correct.
    if(tol > 0)
    {
        d.exceed_valid = true;
        const double limit = tol * d.maxref;
        for(size_t i = 0; i < ref.size(); i++)
        {
            const double a = got[i], b = ref[i];
            // Matching non-finite values agreed in the first pass; keep that
            // verdict here rather than counting them as exceedances too.
            if(!std::isfinite(a) || !std::isfinite(b))
            {
                if(!((std::isnan(a) && std::isnan(b)) || a == b))
                    d.nexceed++;
                continue;
            }
            if(!(std::fabs(a - b) <= limit))   // negated so NaN counts as exceeding
                d.nexceed++;
        }
        d.exceed_frac = (double)d.nexceed / (double)d.n;
    }
    return d;
}

// tol == 0 means bitwise (used by the CUDA self-replay); otherwise the §6.2
// relative tolerance for the backend under test.
struct Outcome
{
    bool ok{true};
    double tol{0.0};
    // Ops whose kernel contains a domain-guard branch are discontinuous in the
    // sampling coordinate: a 1-ulp difference flips the branch and produces a
    // jump of order the local field variation, no matter how good the port is.
    // CUDA disagrees with ITSELF this way when only float contraction changes
    // (measured on ComposeFields), so max-relative-error alone is the wrong
    // statistic. Such ops additionally pass if the exceedances are confined to a
    // negligible fraction of voxels - a genuine bug moves far more than that.
    double max_outlier_frac{0.0};
    // Absolute allowance for scalar comparisons, when the statistic is computed
    // from a discretised intermediate (a histogram) whose bin assignment is a
    // truncation and therefore discontinuous. Set by the caller; 0 disables it.
    // See webgpu_replay_main.cxx's ComputeJointEntropy handler for what it is and,
    // importantly, what it is NOT - it is a heuristic flip allowance, the scalar
    // analogue of max_outlier_frac, not a derived quantum.
    double scalar_abs_floor{0.0};
    double worst_elem_rel{0.0};   // per-element counterpart of worst_rel
    // Per-element gate. `tol` alone is max-normalised - one global allowance set by
    // the volume's largest voxel - so on a field with max 1.9e7 it permits an
    // absolute error of 194 EVERYWHERE, and a port returning zero for the ~52% of
    // elements below that would pass. This closes that. 0 disables.
    double elem_tol{0.0};
    // Worst relative residual seen, recorded whether or not the gate passed. A PASS
    // that prints no number is indistinguishable from a PASS against a loose gate -
    // that ambiguity caused a texture-sampling op to be reported as "bit-exact"
    // when its residual was 57 % of its budget.
    double worst_rel{0.0};
    std::string detail;

    void check(const std::string &what, const Diff &d)
    {
        if(d.rel > worst_rel) worst_rel = d.rel;
        if(d.worst_elem_rel > worst_elem_rel) worst_elem_rel = d.worst_elem_rel;
        bool bad = (tol == 0.0) ? (d.ndiff != 0) : !(d.rel <= tol);
        // An all-zero reference has no scale to be relative to, so `rel` above
        // degrades to a raw ABSOLUTE difference: a port returning values of
        // magnitude <= tol everywhere the reference is exactly 0 would pass while
        // differing at every element. There is no such thing as a small relative
        // error against zero, so any difference at all is a divergence.
        if(tol > 0.0 && d.maxref == 0.0 && d.ndiff != 0) bad = true;
        // Only relax on the outlier rule if the exceedance count is meaningful.
        if(bad && max_outlier_frac > 0 && d.exceed_valid && d.exceed_frac <= max_outlier_frac)
            bad = false;
        // Per-element criterion, applied in ADDITION to the max-normalised one and
        // not relaxable by the outlier rule above (that rule exists for guarded ops'
        // branch discontinuities, which are already reflected in elem_tol's value).
        if(elem_tol > 0.0 && d.worst_elem_rel > elem_tol)
        {
            bad = true;
            char eb[160];
            snprintf(eb, sizeof(eb), "%s: per-element %.3g exceeds %.3g "
                     "(max-normalised was %.3g); ", what.c_str(),
                     d.worst_elem_rel, elem_tol, d.rel);
            detail += eb;
        }
        // Only a DISAGREEMENT involving a non-finite value fails; matching NaNs
        // above were not counted here.
        if(d.nnonfinite)
        {
            ok = false;
            char nb[128];
            snprintf(nb, sizeof(nb), "%s: %zu non-finite mismatch(es) (NaN/Inf in one backend only); ",
                     what.c_str(), d.nnonfinite);
            detail += nb;
            return;
        }
        if(bad)
        {
            ok = false;
            char buf[256];
            snprintf(buf, sizeof(buf),
                     "%s: %zu/%zu differ, %zu over tol (%.3g%%), max|d|=%.6g rel=%.3g (tol %.3g); ",
                     what.c_str(), d.ndiff, d.n, d.nexceed, 100.0 * d.exceed_frac,
                     d.maxabs, d.rel, tol);
            detail += buf;
        }
    }
    void checkScalar(const std::string &what, double got, double ref)
    {
        if(std::isfinite(got) && std::isfinite(ref))
        {
            const double denom = std::fabs(ref) > 0 ? std::fabs(ref) : 1.0;
            const double r = std::fabs(got - ref) / denom;
            if(r > worst_rel) worst_rel = r;
        }
        // Matching non-finite values agree; only a divergence is a failure.
        if(!std::isfinite(got) || !std::isfinite(ref))
        {
            const bool same = (std::isnan(got) && std::isnan(ref)) || (got == ref);
            if(!same)
            {
                ok = false;
                detail += what + ": non-finite mismatch (one backend only); ";
            }
            return;
        }
        const double rel = (ref != 0.0) ? std::fabs(got - ref) / std::fabs(ref) : std::fabs(got - ref);
        // Negated comparisons so a NaN rel counts as a failure, not a pass.
        bool bad = (tol == 0.0) ? (got != ref) : !(rel <= tol);
        // A difference below the statistic's own quantum carries no information.
        // Applies only when the caller declared one, and never in bitwise mode.
        if(bad && tol > 0.0 && scalar_abs_floor > 0.0 &&
           std::fabs(got - ref) <= scalar_abs_floor)
            bad = false;
        if(bad)
        {
            ok = false;
            char buf[256];
            snprintf(buf, sizeof(buf), "%s: got %.9g expected %.9g (rel %.3g); ",
                     what.c_str(), got, ref, rel);
            detail += buf;
        }
    }
};

#endif
