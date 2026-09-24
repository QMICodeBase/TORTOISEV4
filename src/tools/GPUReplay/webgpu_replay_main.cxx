// Replay captured CUDA golden vectors against a GPU backend port.
//
// ONE source, two executables: `webgpu_replay` and `metal_replay`. The backend is
// selected by src/gpu_src/gpu_backend.h at compile time, so the record loader, the
// comparator, the tolerance selection and the driver below are written once. Only
// the binding differs - which is the whole reason the two backends can be graded
// against the same records and the same tolerances.
//
//   webgpu_replay --all  <capture_dir> [--tol 1e-6]
//   webgpu_replay <record_dir>...
//   webgpu_replay --list <capture_dir>
//
// Same records, same loader and same comparison code as gpu_replay; only the
// registry differs. Ops with no WebGPU implementation are reported as "unported"
// and DO count toward a non-zero exit (all 20 reachable ops are ported, so an
// unported record now means a registry or record-naming fault, not work in
// progress). The earlier "unported is not a failure" rule expired at M3.
//
// Two gates, both applied:
//   * max-normalised   max|a-b| / max|b|            1e-5 elementwise/reduction,
//                                                   5e-3 texture-sampling
//   * per-element      |a-b| <= elem_tol*(|b|+rms)  1e-4 elementwise/exact,
//                                                   1e-3 guarded, 2e-2 texture
//
// 1e-5 rather than 1e-6 because 1e-6 sits BELOW the reference's own noise floor:
// the same CUDA source built with -fmad=false disagrees with itself by 2.67e-6.
//
// Texture-sampling kernels get the LOOSER 5e-3 gate against CUDA because CUDA's
// sampler is itself lossy - it misses exact trilinear by 3.3e-5 on a linear ramp,
// which the port reproduces to well under 1e-5. Their real correctness gate is the
// synthetic `reference == "exact"` records at 1e-5, not this one. (An earlier
// comment here claimed CUDA's filter matched exact fp32 weights and that sampling
// kernels therefore took the same gate as everything else; measurement reversed
// that - see CLAUDE.md §5.2.)
//
// TORTOISE_WEBGPU_CUDA_TEXFILTER=1 switches the port to emulate CUDA's quantised
// weights. It is a DIAGNOSTIC only - it feeds no tolerance logic here, and
// measurement showed it is worse than exact trilinear.

#include "gpu_record.h"
#include "gpu_backend.h"   // selects the backend; see src/gpu_src/

#include <dirent.h>
#include <functional>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <map>
#include <sys/stat.h>

// ------------------------------------------------------------ device helpers

static CUDAIMAGE::Pointer ToDevice(const Tensor &t)
{
    CUDAIMAGE::Pointer img = CUDAIMAGE::New();
    img->sz   = make_int3(t.dims[0], t.dims[1], t.dims[2]);
    img->spc  = make_float3(t.spacing[0], t.spacing[1], t.spacing[2]);
    img->orig = make_float3(t.origin[0], t.origin[1], t.origin[2]);
    CUDAIMAGE::ImageType3D::DirectionType dir;
    for(int r = 0; r < 3; r++)
        for(int c = 0; c < 3; c++)
            dir(r, c) = t.direction[3 * r + c];
    img->dir = dir;
    img->components_per_voxel = t.ncomp;
    img->Allocate();
    if(!t.data.empty())
        gpuctx::Upload(img->getFloatdata().buf, t.data.data(), t.data.size() * sizeof(float));
    return img;
}

static std::vector<float> FromDevice(CUDAIMAGE::Pointer img)
{
    std::vector<float> out(img->NumFloats());
    gpuctx::Download(img->getFloatdata().buf, out.data(), out.size() * sizeof(float));
    return out;
}

// ---------------------------------------------------------- operation registry

typedef std::function<void(const Record &, Outcome &)> OpFn;

#define NEED_IN(var, nm)                                                        \
    const Tensor *t_##var = rec.find("in", nm);                                 \
    if(!t_##var) throw std::runtime_error("record is missing input '" nm "'");   \
    CUDAIMAGE::Pointer var = ToDevice(*t_##var);

#define NEED_GEOM(var, nm)                                                      \
    const Tensor *t_##var = rec.find("geom", nm);                               \
    if(!t_##var) throw std::runtime_error("record is missing geometry '" nm "'"); \
    CUDAIMAGE::Pointer var = ToDevice(*t_##var);

#define CHECK_OUT(img, nm)                                                       \
    {                                                                            \
        const Tensor *t_o = rec.find("out", nm);                                 \
        if(!t_o) throw std::runtime_error("record is missing output '" nm "'");   \
        out.check(nm, Compare(FromDevice(img), t_o->data, out.tol));                       \
    }

static float3 F3(const json &j)
{
    std::vector<double> v = j.get<std::vector<double> >();
    float3 r; r.x = v[0]; r.y = v[1]; r.z = v[2];
    return r;
}

static std::vector<float> TapsFrom(const Record &rec)
{
    std::vector<double> k = rec.params["kernel"].get<std::vector<double> >();
    return std::vector<float>(k.begin(), k.end());
}

static std::map<std::string, OpFn> BuildRegistry()
{
    std::map<std::string, OpFn> R;

    R["ResampleImage"] = [](const Record &rec, Outcome &out) {
        NEED_IN(main_field, "main_field");
        NEED_GEOM(virtual_img, "virtual_img");
        CUDAIMAGE::Pointer o = ResampleImage(main_field, virtual_img);
        CHECK_OUT(o, "output");
    };

    R["WarpImage"] = [](const Record &rec, Outcome &out) {
        NEED_IN(main_image, "main_image");
        NEED_IN(field_image, "field_image");
        CUDAIMAGE::Pointer o = WarpImage(main_image, field_image);
        CHECK_OUT(o, "output");
    };

    R["QuadraticTransformImageC"] = [](const Record &rec, Outcome &out) {
        NEED_IN(main_image, "main_image");
        NEED_GEOM(target_img, "target_img");
        std::vector<double> pr = rec.params["params"].get<std::vector<double> >();
        TransformType::Pointer tp = TransformType::New();
        tp->SetPhase((short)(double)rec.params["phase"]);
        TransformType::ParametersType par(TransformType::NQUADPARAMS);
        for(int i = 0; i < TransformType::NQUADPARAMS; i++) par[i] = pr[i];
        tp->SetParameters(par);
        std::vector<double> mref = rec.params["matrix"].get<std::vector<double> >();
        TransformType::MatrixType m = tp->GetMatrix();
        for(int r = 0; r < 3; r++)
            for(int c = 0; c < 3; c++)
                if((float)m(r, c) != (float)mref[3 * r + c])
                    throw std::runtime_error("rebuilt transform matrix differs from the captured one");
        CUDAIMAGE::Pointer o = QuadraticTransformImageC(main_image, tp, target_img);
        CHECK_OUT(o, "output");
    };

    R["GaussianSmoothImage"] = [](const Record &rec, Outcome &out) {
        NEED_IN(main_image, "main_image");
        CUDAIMAGE::Pointer o;
        if(rec.params.contains("kernel"))
        {
            // Synthetic record: taps supplied so the reference is exact.
            std::vector<double> kd = rec.params["kernel"].get<std::vector<double> >();
            std::vector<float> taps(kd.begin(), kd.end());
            o = GaussianSmoothImageWithTaps(main_image, taps, (float)rec.params["std"]);
        }
        else
            o = GaussianSmoothImage(main_image, (float)rec.params["std"]);
        CHECK_OUT(o, "output");
    };

    R["AddImages"] = [](const Record &rec, Outcome &out) {
        NEED_IN(im1, "im1"); NEED_IN(im2, "im2");
        CHECK_OUT(AddImages(im1, im2), "output");
    };
    R["MultiplyImages"] = [](const Record &rec, Outcome &out) {
        NEED_IN(im1, "im1"); NEED_IN(im2, "im2");
        CHECK_OUT(MultiplyImages(im1, im2), "output");
    };
    R["MultiplyImage"] = [](const Record &rec, Outcome &out) {
        NEED_IN(im1, "im1");
        CHECK_OUT(MultiplyImage(im1, (float)rec.params["factor"]), "output");
    };
    R["PreprocessImage"] = [](const Record &rec, Outcome &out) {
        NEED_IN(img, "img");
        CHECK_OUT(PreprocessImage(img, (float)rec.params["low_val"],
                                  (float)rec.params["up_val"]), "output");
    };
    R["RestrictPhase"] = [](const Record &rec, Outcome &out) {
        NEED_IN(field, "field");
        RestrictPhase(field, F3(rec.params["phase"]));
        CHECK_OUT(field, "field");
    };
    R["ContrainDefFields"] = [](const Record &rec, Outcome &out) {
        NEED_IN(ufield, "ufield"); NEED_IN(dfield, "dfield");
        ContrainDefFields(ufield, dfield);
        CHECK_OUT(ufield, "ufield"); CHECK_OUT(dfield, "dfield");
    };

    R["ScaleUpdateField"] = [](const Record &rec, Outcome &out) {
        NEED_IN(field, "field");
        ScaleUpdateField(field, (float)rec.params["scale_factor"]);
        CHECK_OUT(field, "field");
    };
    R["AddToUpdateField"] = [](const Record &rec, Outcome &out) {
        NEED_IN(updateField, "updateField"); NEED_IN(updateField_temp, "updateField_temp");
        AddToUpdateField(updateField, updateField_temp, (float)rec.params["weight"],
                         rec.params["normalize"] != 0);
        CHECK_OUT(updateField, "updateField");
    };
    R["SumImage"] = [](const Record &rec, Outcome &out) {
        NEED_IN(im1, "im1");
        out.checkScalar("sum", SumImage(im1), rec.scalars["sum"]);
    };

    R["ComposeFields"] = [](const Record &rec, Outcome &out) {
        NEED_IN(main_field, "main_field"); NEED_IN(update_field, "update_field");
        CHECK_OUT(ComposeFields(main_field, update_field), "output");
    };

    R["InvertField"] = [](const Record &rec, Outcome &out) {
        NEED_IN(field, "field");
        CUDAIMAGE::Pointer init = nullptr;
        const Tensor *ti = rec.find("in", "initial_estimate");
        if(ti) init = ToDevice(*ti);
        CHECK_OUT(InvertField(field, init), "output");
    };

    R["ComputeJointEntropy"] = [](const Record &rec, Outcome &out) {
        NEED_IN(img1, "img1"); NEED_IN(img2, "img2");
        std::vector<double> L = rec.params["lims"].get<std::vector<double> >();
        float ej, e1, e2;
        ComputeJointEntropy(img1, L[0], L[1], img2, L[2], L[3],
                            (int)(double)rec.params["Nbins"], ej, e1, e2);

        // Allowance for a few histogram bin-index flips. The bin index is a
        // truncation, so a 1-ulp difference in the index expression moves a sample
        // between bins - a discontinuous change no arithmetic fidelity can prevent.
        // This is the SCALAR analogue of the max_outlier_frac rule used for guarded
        // array ops, not a bound derived from first principles.
        //
        // HONEST STATEMENT OF WHAT IT IS NOT. An earlier version of this comment
        // called 2*ln2/N "the quantum" of the statistic and claimed nothing smaller
        // could be measured. That was WRONG. Writing S = (1/N)*sum g(n_i) - ln N
        // with g(n) = n*ln n, moving one sample from a bin of count a to one of
        // count b changes S by [g(a-1)+g(b+1)-g(a)-g(b)]/N. That equals 2*ln2/N
        // only when a == b == 1. It is exactly 0 when b == a-1, of order 1/(nN)
        // for a = n+1, b = n-1, and grows like (ln n + 1)/N when a sample enters a
        // well-populated bin - i.e. UNBOUNDED above, not a lattice spacing. The
        // expression below is a heuristic scale, and K was chosen, not measured.
        //
        // Cost: on `fast` this is ~15-25x the elementwise gate, so it does reduce
        // detection power - notably for defects touching only the >=low/<=high
        // admission boundary, which move a handful of samples. The exact-reference
        // record below is what preserves a tight check.
        //
        // NEVER applied to an exact-reference record. Those exist precisely to
        // escape this allowance - every value sits at a Parzen bin centre so the
        // histogram is identical by construction and the only thing left to measure
        // is the p*log(p) arithmetic (measured residual 5.3e-8, ~200x under the
        // gate). Applying the floor there would have loosened the one record built
        // to be immune to it by 8-14x, defeating its entire purpose.
        if(rec.reference != "exact")
        {
            const Tensor *ti = rec.find("in", "img1");
            const double N = ti ? (double)ti->dims[0] * ti->dims[1] * ti->dims[2] : 0.0;
            const int    K = 4;
            if(N > 0)
                out.scalar_abs_floor = K * 2.0 * std::log(2.0) / N;
        }
        out.checkScalar("entropy_joint", ej, rec.scalars["entropy_joint"]);
        out.checkScalar("entropy_img1",  e1, rec.scalars["entropy_img1"]);
        out.checkScalar("entropy_img2",  e2, rec.scalars["entropy_img2"]);
    };

    R["ComputeMetric_CC"] = [](const Record &rec, Outcome &out) {
        NEED_IN(up_img, "up_img"); NEED_IN(down_img, "down_img");
        CUDAIMAGE::Pointer uF, uM;
        const float v = ComputeMetric_CC(up_img, down_img, uF, uM);
        out.checkScalar("metric_value", v, rec.scalars["metric_value"]);
        CHECK_OUT(uF, "updateFieldF"); CHECK_OUT(uM, "updateFieldM");
    };

    R["ComputeMetric_CCSK"] = [](const Record &rec, Outcome &out) {
        NEED_IN(up_img, "up_img"); NEED_IN(down_img, "down_img"); NEED_IN(str_img, "str_img");
        CUDAIMAGE::Pointer uF, uM;
        const float v = ComputeMetric_CCSK(up_img, down_img, str_img, uF, uM,
                                           (float)rec.params["t"]);
        out.checkScalar("metric_value", v, rec.scalars["metric_value"]);
        CHECK_OUT(uF, "updateFieldF"); CHECK_OUT(uM, "updateFieldM");
    };

    R["ComputeMetric_MSJac"] = [](const Record &rec, Outcome &out) {
        NEED_IN(up_img, "up_img"); NEED_IN(down_img, "down_img");
        NEED_IN(def_FINV, "def_FINV"); NEED_IN(def_MINV, "def_MINV");
        CUDAIMAGE::Pointer uF, uM;
        const float v = ComputeMetric_MSJacWithTaps(up_img, down_img, def_FINV, def_MINV, uF, uM,
                                                    F3(rec.params["phase_vector"]), TapsFrom(rec));
        out.checkScalar("metric_value", v, rec.scalars["metric_value"]);
        CHECK_OUT(uF, "updateFieldF"); CHECK_OUT(uM, "updateFieldM");
    };

    R["ComputeMetric_CCJacS"] = [](const Record &rec, Outcome &out) {
        NEED_IN(up_img, "up_img"); NEED_IN(down_img, "down_img"); NEED_IN(str_img, "str_img");
        NEED_IN(def_FINV, "def_FINV"); NEED_IN(def_MINV, "def_MINV");
        CUDAIMAGE::Pointer uF, uM;
        const float v = ComputeMetric_CCJacSWithTaps(up_img, down_img, str_img, def_FINV, def_MINV,
                                                     uF, uM, F3(rec.params["phase_vector"]),
                                                     TapsFrom(rec));
        out.checkScalar("metric_value", v, rec.scalars["metric_value"]);
        CHECK_OUT(uF, "updateFieldF"); CHECK_OUT(uM, "updateFieldM");
    };

    return R;
}

// Kernels containing a domain-guard branch, where a 1-ulp coordinate difference
// can flip the branch for a handful of voxels (see Outcome::max_outlier_frac).
static bool IsGuarded(const std::string &op)
{
    // InvertField is a fixed-point iteration built on ComposeFields - it calls the
    // guarded kernel ~40 times, so it inherits the branch discontinuity.
    return op == "ComposeFields" || op == "ResampleImage" ||
           op == "QuadraticTransformImageC" || op == "InvertField";
}

// Ops whose CUDA implementation samples through a hardware texture. NOTE: the
// documented 1/256 filter-weight quantisation is NOT the operative model - plan
// §5.2 records that emulating it was ~1000x FURTHER from CUDA than exact fp32
// weights. These ops get a loose gate against CUDA because CUDA's sampler is
// measurably the lossy side, not because the port quantises.
static bool IsTextureSampling(const std::string &op)
{
    return op == "WarpImage" || op == "QuadraticTransformImageC";
}

// ------------------------------------------------------------------- driver

static std::vector<std::string> ListRecordDirs(const std::string &root)
{
    std::vector<std::string> dirs;
    DIR *d = opendir(root.c_str());
    if(!d) return dirs;
    struct dirent *e;
    while((e = readdir(d)))
    {
        if(e->d_name[0] == '.') continue;
        const std::string p = root + "/" + e->d_name;
        struct stat st, js;
        if(stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode) &&
           stat((p + "/record.json").c_str(), &js) == 0)
            dirs.push_back(p);
    }
    closedir(d);
    std::sort(dirs.begin(), dirs.end());
    return dirs;
}

// --- benchmark mode -------------------------------------------------------
//
// Times each op on the captured records, which are real pipeline shapes. This
// exists because CLAUDE.md 7.1's four suspected causes of the WebGPU slowdown are
// hypotheses: DRBUDDI's ITERATION_TIME localises the cost to a stage, not to an
// op, and TORTOISE_GPU_PROFILE was never implemented.
//
// CAVEAT, do not forget when reading the output: each timed call includes the
// harness's own host-side Compare() over the whole output buffer. That term is
// identical in gpu_replay, so a WebGPU-vs-CUDA ratio is CONSERVATIVE (biased
// toward 1) and an op that still looks slow really is. It is not a clean kernel
// timing - it measures the wrapper as the pipeline calls it, which is the thing
// being optimised.
struct BenchRow
{
    std::string op;
    double      best_ms{0.0};
    double      total_ms{0.0};
    int         calls{0};
};

int RunBench(const std::vector<std::string> &dirs,
             std::map<std::string, OpFn> &registry, int reps)
{
    std::cout << "per-op timing over " << reps << " reps (best of), including the\n"
                 "harness's host-side comparison - see the caveat in the source.\n\n";
    std::map<std::string, BenchRow> rows;
    for(size_t i = 0; i < dirs.size(); i++)
    {
        Record rec;
        try { rec = LoadRecord(dirs[i]); } catch(std::exception &) { continue; }
        std::map<std::string, OpFn>::iterator it = registry.find(rec.op);
        if(it == registry.end()) continue;

        // Warm-up: the first call builds and caches the pipeline, which would
        // otherwise be charged entirely to whichever record ran first.
        try { Outcome warm; it->second(rec, warm); }
        catch(std::exception &) { continue; }

        double best = 1e300, total = 0.0;
        for(int r = 0; r < reps; r++)
        {
            // tol MUST stay 0 here. Compare()'s exceedance sweep is gated on
            // `tol > 0`, and gpu_replay's CHECK_OUT passes no tol - so setting one
            // made WebGPU run two passes over every output buffer where CUDA ran
            // one, biasing the ratio against WebGPU in proportion to output size.
            Outcome o;
            const std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
            try { it->second(rec, o); } catch(std::exception &) { best = -1.0; break; }
            const double ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - t0).count();
            if(ms < best) best = ms;
            total += ms;
        }
        if(best < 0) continue;
        BenchRow &row = rows[rec.op];
        row.op = rec.op;
        row.best_ms += best;
        row.total_ms += total;
        row.calls++;
    }

    std::vector<BenchRow> v;
    for(std::map<std::string, BenchRow>::iterator it = rows.begin(); it != rows.end(); ++it)
        v.push_back(it->second);
    // Slowest first: that is the optimisation work list.
    for(size_t a = 0; a + 1 < v.size(); a++)
        for(size_t b = a + 1; b < v.size(); b++)
            if(v[b].best_ms > v[a].best_ms) std::swap(v[a], v[b]);

    double grand = 0.0;
    for(size_t i = 0; i < v.size(); i++) grand += v[i].best_ms;
    printf("  %-28s %10s %8s %10s\n", "op", "best_ms", "records", "share");
    for(size_t i = 0; i < v.size(); i++)
        printf("  %-28s %10.2f %8d %9.1f%%\n", v[i].op.c_str(), v[i].best_ms,
               v[i].calls, grand > 0 ? 100.0 * v[i].best_ms / grand : 0.0);
    printf("  %-28s %10.2f\n", "TOTAL", grand);
    return 0;
}

int main(int argc, char *argv[])
{
    double tol_elementwise = 1e-5;   // measured contraction noise floor is 2.7e-6
    // 500x looser than tol_elementwise, deliberately: CUDA's sampler is lossy, so
    // this gate bounds CUDA's OWN error and catches gross faults only. The primary
    // correctness gate for these ops is the synthetic exact-reference records.
    // (Observed max vs CUDA: 3.6e-3.) See CLAUDE.md §5.2.
    double tol_texture     = 5e-3;
    int    reps            = 5;

    std::vector<std::string> args;
    for(int i = 1; i < argc; i++)
    {
        const std::string a = argv[i];
        // Sets BOTH gates. Previously it set only the elementwise one, so
        // `--tol 1e-9` was silently ignored for WarpImage and
        // QuadraticTransformImageC - which are graded at the loose 5e-3 texture
        // gate - and a PASS there was misread as bit-exactness.
        if(a == "--tol" && i + 1 < argc)
        { tol_elementwise = tol_texture = atof(argv[++i]); continue; }
        if(a == "--tol-texture" && i + 1 < argc) { tol_texture = atof(argv[++i]); continue; }
        if(a == "--reps" && i + 1 < argc) { reps = atoi(argv[++i]); continue; }
        args.push_back(a);
    }
    if(args.empty())
    {
        std::cerr << "usage: webgpu_replay [--all|--list|--bench] <capture_dir>"
                     " [--tol T] [--tol-texture T] [--reps N]\n";
        return 1;
    }

    std::map<std::string, OpFn> registry = BuildRegistry();

    std::vector<std::string> dirs;
    const std::string mode = args[0];
    if(mode == "--all" || mode == "--list" || mode == "--bench")
    {
        if(args.size() < 2) { std::cerr << "need a capture dir\n"; return 1; }
        dirs = ListRecordDirs(args[1]);
        if(dirs.empty()) { std::cerr << "no records under " << args[1] << "\n"; return 1; }
    }
    else
        dirs = args;

    if(mode == "--list")
    {
        std::map<std::string, int> per_op;
        for(size_t i = 0; i < dirs.size(); i++)
            try { per_op[LoadRecord(dirs[i]).op]++; } catch(std::exception &) {}
        std::cout << "op coverage (" GPU_BACKEND_NAME "):\n";
        for(std::map<std::string, int>::iterator it = per_op.begin(); it != per_op.end(); ++it)
            std::cout << "  " << (registry.count(it->first) ? "[ported  ] " : "[unported] ")
                      << it->first << "  x" << it->second << "\n";
        return 0;
    }

    if(mode == "--bench")
    {
        gpuctx::Init();
        std::cout << GPU_BACKEND_NAME " bench on " << gpuctx::Info().Describe() << "\n\n";
        return RunBench(dirs, registry, reps);
    }

    gpuctx::Init();
    std::cout << GPU_BACKEND_NAME " replay on " << gpuctx::Info().Describe() << "\n"
              << "tolerances: elementwise " << tol_elementwise
              << ", texture-sampling " << tol_texture
              << (gpuctx::TexFilterEmulation() ? " (cuda-filter emulation ON)" : "")
              << "\n\n";

    int pass = 0, fail = 0, bad = 0, unported = 0;
    int xdiverge = 0;   // records with a declared, verified expected divergence
    for(size_t i = 0; i < dirs.size(); i++)
    {
        Record rec;
        try { rec = LoadRecord(dirs[i]); }
        catch(std::exception &e)
        { std::cout << "BAD      " << dirs[i] << ": " << e.what() << "\n"; bad++; continue; }

        std::map<std::string, OpFn>::iterator it = registry.find(rec.op);
        if(it == registry.end())
        {
            // Print it. Silently skipping meant a suite could lose whole ops and
            // still exit 0 - the count in revalidate.sh was the only thing noticing,
            // and that is a TOTAL, so swapping records between ops hid it entirely.
            std::cout << "UNPORTED " << rec.op << " (" << dirs[i] << ")\n";
            unported++;
            continue;
        }

        const uint64_t errors_before = gpuctx::ErrorCount();
        Outcome out;
        // Against an exact analytic reference every op gets the tight gate; only
        // comparisons against CUDA absorb CUDA's own sampler loss.
        const bool exact_ref = (rec.reference == "exact");
        out.tol = (IsTextureSampling(rec.op) && !exact_ref) ? tol_texture : tol_elementwise;
        if(IsGuarded(rec.op))
            out.max_outlier_frac = 1e-3;      // 0.1 % of voxels may flip the guard

        // PER-ELEMENT gate, derived from measurement (2026-08-20). Worst observed
        // per-element error, by class:
        //   11 ops (AddImages, GaussianSmoothImage, ComputeJointEntropy, ...)  0
        //   metric kernels                                       3.0e-6 - 2.8e-5
        //   guarded (ComposeFields, InvertField)                 1.8e-4 - 4.0e-4
        //   texture-sampling vs CUDA                             2.2e-3 - 9.1e-3
        // Measured effect (audit, 2026-08-20): median band shrink 1.08x versus the
        // max-normalised gate; on 50 of 150 tensors the per-element band is looser.
        // It helps the elementwise ops materially and the metric kernels barely.
        // An earlier comment claimed "3.6x margin"; that assumed an attacker aiming
        // at the old bound and is withdrawn.
        // Guarded and texture ops get looser values reflecting their measured
        // floors (branch discontinuity and CUDA's own sampler loss respectively),
        // each still well under what a defect produces.
        out.elem_tol = exact_ref            ? 1e-4
                     : IsTextureSampling(rec.op) ? 2e-2
                     : IsGuarded(rec.op)         ? 1e-3
                                                 : 1e-4;
        try { it->second(rec, out); }
        catch(std::exception &e)
        { std::cout << "BAD      " << rec.op << ": " << e.what() << "\n"; bad++; continue; }

        std::string label = dirs[i].substr(dirs[i].find_last_of('/') + 1);
        if(gpuctx::ErrorCount() != errors_before)
        {
            std::cout << "BAD      " << label << ": "
                      << (gpuctx::ErrorCount() - errors_before)
                      << " " GPU_BACKEND_NAME " validation error(s) - dispatch was rejected, results are"
                         " meaningless\n";
            bad++;
            continue;
        }
        // EXPECTED DIVERGENCE. A record may declare that the port is KNOWN to differ
        // from CUDA here, because CUDA is the incorrect implementation - currently
        // only ScaleUpdateField on a field width where CUDA's pitched/flat indexing
        // bug fires (CLAUDE.md 0.0b). Such a record must NOT be silently excluded:
        //   * diverging by the predicted amount  -> XDIVERGE, correct
        //   * agreeing with CUDA                 -> FAIL, the backend reproduced the bug
        //   * diverging by some OTHER amount     -> FAIL, unexplained
        // So this converts a permanent unexplained failure into a real two-sided
        // check, rather than muting it.
        if(rec.expected_divergence)
        {
            const double want = rec.expected_rel, got = out.worst_rel;
            const double lo = want * (1.0 - rec.expected_rel_tol);
            const double hi = want * (1.0 + rec.expected_rel_tol);
            if(got >= lo && got <= hi)
            {
                printf("XDIVERGE %-34s rel %.4g (expected %.4g +/- %.0f%%) - %s\n",
                       label.c_str(), got, want, rec.expected_rel_tol * 100.0,
                       rec.expected_reason.c_str());
                xdiverge++;
                continue;
            }
            std::cout << "FAIL     " << label << ": expected a KNOWN divergence of rel "
                      << want << " +/- " << (rec.expected_rel_tol * 100.0) << "% ("
                      << rec.expected_reason << ") but measured rel " << got
                      << (got < lo ? " - the backend appears to REPRODUCE CUDA's bug"
                                   : " - divergence is larger than predicted")
                      << "\n";
            fail++;
            continue;
        }
        if(out.ok)
        {
            // Always report the residual and the gate it was judged against.
            // Report BOTH statistics. `rel` is max-normalised (a global allowance);
            // `elem` is per-element against |ref| + rms. The second is the honest
            // one - see the Diff comment in gpu_record.h.
            printf("PASS     %-34s rel %.3g elem %.3g (gate %.3g)\n", label.c_str(),
                   out.worst_rel, out.worst_elem_rel, out.tol);
            pass++;
        }
        else       { std::cout << "FAIL     " << label << ": " << out.detail << "\n"; fail++; }
    }

    std::cout << "\n" << pass << " passed, " << fail << " failed, " << bad
              << " unusable, " << unported << " not yet ported";
    if(xdiverge) std::cout << ", " << xdiverge << " expected divergence(s)";
    std::cout << "\n";
    // `unported` is a failure now. The "usable while incomplete" rationale in this
    // file's header expired at M3; all 20 reachable ops have been ported since.
    return (fail || bad || unported) ? 1 : 0;
}
