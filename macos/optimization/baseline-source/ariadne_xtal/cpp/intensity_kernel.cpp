// Optional Ariadne intensity/analytic-derivative kernel.
//
// CPython's checked buffer protocol is the only binding dependency.  All
// external arrays contain native double or signed int64; no NumPy headers,
// std::complex layout, BLAS implementation, or OpenMP runtime is assumed.
// Statistical models and crystallographic parameterizations stay in Python.

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

namespace {

constexpr double PI = 3.141592653589793238462643383279502884;
constexpr double TWO_PI = 2.0 * PI;
constexpr double MINUS_TWO_PI_SQUARED = -2.0 * (PI * PI);

struct Buffer {
    Py_buffer value{};
    bool acquired = false;
    Py_ssize_t elements = 0;

    Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    ~Buffer() { if (acquired) PyBuffer_Release(&value); }

    bool get(PyObject* object, const char* name, bool integer, int ndim,
             bool writable = false) {
        const int flags = PyBUF_FORMAT | PyBUF_STRIDES | (writable ? PyBUF_WRITABLE : 0);
        if (PyObject_GetBuffer(object, &value, flags) != 0) return false;
        acquired = true;
        if (value.ndim != ndim || value.shape == nullptr) {
            PyErr_Format(PyExc_ValueError, "%s must have %d dimensions", name, ndim);
            return false;
        }
        const char* format = value.format == nullptr ? "" : value.format;
        bool native_format = false;
        if (!integer) {
            native_format = std::strcmp(format, "d") == 0 || std::strcmp(format, "@d") == 0
                         || std::strcmp(format, "=d") == 0;
        } else {
            native_format = std::strcmp(format, "q") == 0 || std::strcmp(format, "@q") == 0
                         || std::strcmp(format, "=q") == 0
                         || (sizeof(long) == 8 && (std::strcmp(format, "l") == 0
                                                   || std::strcmp(format, "@l") == 0))
                         || (sizeof(Py_ssize_t) == 8 && (std::strcmp(format, "n") == 0
                                                         || std::strcmp(format, "@n") == 0));
        }
        if (value.itemsize != 8 || !native_format) {
            PyErr_Format(PyExc_TypeError, "%s requires native-endian %s", name,
                         integer ? "signed int64" : "float64");
            return false;
        }
        if (!PyBuffer_IsContiguous(&value, 'C')) {
            PyErr_Format(PyExc_ValueError, "%s must be C-contiguous", name);
            return false;
        }
        elements = 1;
        for (int i = 0; i < ndim; ++i) {
            const Py_ssize_t dimension = value.shape[i];
            if (dimension < 0 || (dimension != 0 && elements > PY_SSIZE_T_MAX / dimension)) {
                PyErr_Format(PyExc_OverflowError, "%s shape exceeds addressable memory", name);
                return false;
            }
            elements *= dimension;
        }
        if (elements > PY_SSIZE_T_MAX / 8 || value.len != elements * 8) {
            PyErr_Format(PyExc_ValueError, "%s shape and buffer length disagree", name);
            return false;
        }
        if (value.len && (value.buf == nullptr
                || reinterpret_cast<std::uintptr_t>(value.buf) % alignof(double) != 0)) {
            PyErr_Format(PyExc_ValueError, "%s must be aligned to native double storage", name);
            return false;
        }
        const auto address = reinterpret_cast<std::uintptr_t>(value.buf);
        if (static_cast<std::uintmax_t>(value.len)
                > std::numeric_limits<std::uintptr_t>::max() - address) {
            PyErr_Format(PyExc_OverflowError, "%s address range overflows", name);
            return false;
        }
        return true;
    }

    const double* doubles() const { return static_cast<const double*>(value.buf); }
    double* writable_doubles() const { return static_cast<double*>(value.buf); }
    std::int64_t integer(Py_ssize_t index) const {
        // memcpy avoids type-aliasing assumptions about a platform's long vs
        // long long representation of a signed, native-endian 64-bit buffer.
        std::int64_t answer;
        std::memcpy(&answer, static_cast<const char*>(value.buf) + index * 8, 8);
        return answer;
    }
};

bool overlap(const Buffer& a, const Buffer& b) {
    if (!a.value.len || !b.value.len) return false;
    const auto first = reinterpret_cast<std::uintptr_t>(a.value.buf);
    const auto second = reinterpret_cast<std::uintptr_t>(b.value.buf);
    return first < second + b.value.len && second < first + a.value.len;
}

bool finite(const Buffer& buffer, const char* name) {
    const double* values = buffer.doubles();
    for (Py_ssize_t i = 0; i < buffer.elements; ++i) {
        if (!std::isfinite(values[i])) {
            PyErr_Format(PyExc_ValueError, "%s contains a nonfinite value", name);
            return false;
        }
    }
    return true;
}

inline void compensated_add(double value, double& sum, double& correction) {
    const double total = sum + value;
    correction += std::abs(sum) >= std::abs(value) ? (sum - total) + value
                                                  : (value - total) + sum;
    sum = total;
}

PyObject* intensity_block(PyObject*, PyObject* args) {
    PyObject *objects[10];
    if (!PyArg_ParseTuple(args, "OOOOOOOOOO:intensity_block", &objects[0], &objects[1],
            &objects[2], &objects[3], &objects[4], &objects[5], &objects[6],
            &objects[7], &objects[8], &objects[9])) return nullptr;
    Buffer b[10];
    const char* names[] = {"hkl", "q", "form_factors", "sites", "site_elements",
                          "tangent_offsets", "tangent_columns", "tangents", "intensity", "jacobian"};
    const int dimensions[] = {2, 2, 2, 2, 1, 1, 1, 2, 1, 2};
    for (int i = 0; i < 10; ++i) {
        const bool is_integer = i == 0 || i == 4 || i == 5 || i == 6;
        if (!b[i].get(objects[i], names[i], is_integer, dimensions[i], i >= 8)) return nullptr;
    }
    const Py_ssize_t n = b[0].value.shape[0];
    const Py_ssize_t elements = b[2].value.shape[0];
    const Py_ssize_t sites = b[3].value.shape[0];
    const Py_ssize_t tangents = b[6].value.shape[0];
    const Py_ssize_t parameters = b[9].value.shape[1];
    if (b[0].value.shape[1] != 3 || b[1].value.shape[0] != n || b[1].value.shape[1] != 3
            || b[2].value.shape[1] != n || b[3].value.shape[1] != 12
            || b[4].value.shape[0] != sites || b[5].value.shape[0] != sites + 1
            || b[7].value.shape[0] != tangents || b[7].value.shape[1] != 10
            || b[8].value.shape[0] != n || b[9].value.shape[0] != n) {
        PyErr_SetString(PyExc_ValueError, "inconsistent intensity kernel dimensions");
        return nullptr;
    }
    for (int output = 8; output < 10; ++output) {
        for (int input = 0; input < output; ++input) {
            if (overlap(b[output], b[input])) {
                PyErr_SetString(PyExc_ValueError, "kernel outputs must not alias an input or another output");
                return nullptr;
            }
        }
    }
    for (int i : {1, 2, 3, 7}) if (!finite(b[i], names[i])) return nullptr;
    for (Py_ssize_t i = 0; i < b[0].elements; ++i) {
        if (b[0].integer(i) < std::numeric_limits<std::int32_t>::min()
                || b[0].integer(i) > std::numeric_limits<std::int32_t>::max()) {
            PyErr_SetString(PyExc_ValueError, "kernel Miller index exceeds supported int32 range");
            return nullptr;
        }
    }
    const double* site_data = b[3].doubles();
    for (Py_ssize_t i = 0; i < sites; ++i) {
        if (b[4].integer(i) < 0 || b[4].integer(i) >= elements) {
            PyErr_SetString(PyExc_ValueError, "site element index is out of range");
            return nullptr;
        }
        if (site_data[i * 12 + 9] < 0 || site_data[i * 12 + 9] > 1) {
            PyErr_SetString(PyExc_ValueError, "site occupancy must lie in [0,1]");
            return nullptr;
        }
    }
    if (b[5].integer(0) != 0 || b[5].integer(sites) != tangents) {
        PyErr_SetString(PyExc_ValueError, "tangent offsets must span the tangent arrays");
        return nullptr;
    }
    for (Py_ssize_t i = 0; i < sites; ++i) {
        if (b[5].integer(i) < 0 || b[5].integer(i) > b[5].integer(i + 1)
                || b[5].integer(i + 1) > tangents) {
            PyErr_SetString(PyExc_ValueError, "tangent offsets must be monotone and in range");
            return nullptr;
        }
    }
    for (Py_ssize_t i = 0; i < tangents; ++i) {
        if (b[6].integer(i) < 0 || b[6].integer(i) >= parameters) {
            PyErr_SetString(PyExc_ValueError, "tangent parameter column is out of range");
            return nullptr;
        }
    }
    if (!n) Py_RETURN_NONE;
    const double* q = b[1].doubles();
    const double* ff = b[2].doubles();
    const double* tangent_data = b[7].doubles();
    double* intensity = b[8].writable_doubles();
    double* jacobian = b[9].writable_doubles();
    bool numeric_error = false;
    try {
        // Only one reflection's contributions and compensation are retained.
        // This replaces a block-by-parameter complex derivative array.
        std::vector<double> contribution_real(tangents ? sites : 0);
        std::vector<double> contribution_imag(tangents ? sites : 0);
        std::vector<double> gradient_correction(parameters);
        Py_BEGIN_ALLOW_THREADS
        for (Py_ssize_t h = 0; h < n && !numeric_error; ++h) {
            const double hx = static_cast<double>(b[0].integer(h * 3));
            const double hy = static_cast<double>(b[0].integer(h * 3 + 1));
            const double hz = static_cast<double>(b[0].integer(h * 3 + 2));
            const double qx = q[h * 3], qy = q[h * 3 + 1], qz = q[h * 3 + 2];
            const double metric[6] = {qx*qx, qy*qy, qz*qz, 2*qx*qy, 2*qx*qz, 2*qy*qz};
            double real = 0, imag = 0, correction_real = 0, correction_imag = 0;
            for (Py_ssize_t site = 0; site < sites; ++site) {
                const double* a = site_data + site * 12;
                double quadratic = 0;
                for (int j = 0; j < 6; ++j) quadratic += metric[j] * a[3 + j];
                const double decay = std::exp(MINUS_TWO_PI_SQUARED * quadratic);
                const double phase = TWO_PI * (hx * a[0] + hy * a[1] + hz * a[2]);
                const double cosine = std::cos(phase), sine = std::sin(phase);
                const double fr = ff[b[4].integer(site) * n + h] + a[10];
                const double fi = a[11];
                const double cr = decay * (fr * cosine - fi * sine);
                const double ci = decay * (fr * sine + fi * cosine);
                if (!std::isfinite(cr) || !std::isfinite(ci)) { numeric_error = true; break; }
                compensated_add(a[9] * cr, real, correction_real);
                compensated_add(a[9] * ci, imag, correction_imag);
                if (tangents) {
                    contribution_real[site] = cr;
                    contribution_imag[site] = ci;
                }
            }
            if (numeric_error) break;
            real += correction_real;
            imag += correction_imag;
            intensity[h] = real * real + imag * imag;
            if (!std::isfinite(intensity[h])) { numeric_error = true; break; }
            if (!parameters) continue;
            double* row = jacobian + h * parameters;
            std::fill(row, row + parameters, 0.0);
            std::fill(gradient_correction.begin(), gradient_correction.end(), 0.0);
            if (tangents) for (Py_ssize_t site = 0; site < sites; ++site) {
                const double occupancy = site_data[site * 12 + 9];
                const double cr = contribution_real[site], ci = contribution_imag[site];
                const double real_weight = 2 * (real * cr + imag * ci);
                const double imag_weight = 2 * (imag * cr - real * ci);
                for (std::int64_t t = b[5].integer(site); t < b[5].integer(site + 1); ++t) {
                    const double* direction = tangent_data + t * 10;
                    double quadratic = 0;
                    for (int j = 0; j < 6; ++j) quadratic += metric[j] * direction[3 + j];
                    const double ddecay = MINUS_TWO_PI_SQUARED * quadratic;
                    const double dphase = TWO_PI * (hx * direction[0] + hy * direction[1] + hz * direction[2]);
                    const double value = real_weight * (direction[9] + occupancy * ddecay)
                                       + imag_weight * occupancy * dphase;
                    const auto column = b[6].integer(t);
                    compensated_add(value, row[column], gradient_correction[column]);
                }
            }
            for (Py_ssize_t column = 0; column < parameters; ++column) {
                row[column] += gradient_correction[column];
                if (!std::isfinite(row[column])) { numeric_error = true; break; }
            }
        }
        Py_END_ALLOW_THREADS
    } catch (const std::bad_alloc&) {
        return PyErr_NoMemory();
    } catch (const std::length_error&) {
        PyErr_SetString(PyExc_OverflowError, "kernel scratch arrays exceed addressable memory");
        return nullptr;
    }
    if (numeric_error) {
        PyErr_SetString(PyExc_ValueError, "intensity kernel exceeded finite floating-point range");
        return nullptr;
    }
    Py_RETURN_NONE;
}

PyObject* build_info(PyObject*, PyObject*) {
#if defined(__x86_64__) || defined(_M_X64)
    const char* architecture = "x86_64";
#elif defined(__aarch64__) || defined(_M_ARM64)
    const char* architecture = "aarch64";
#else
    const char* architecture = "other";
#endif
#ifdef _MSC_VER
    const char* compiler = "MSVC";
#else
    const char* compiler = __VERSION__;
#endif
    return Py_BuildValue("{s:i,s:s,s:s,s:s,s:i,s:O,s:O,s:O}",
        "kernel_version", 1, "binding", "CPython checked native buffers",
        "compiler", compiler, "architecture", architecture, "internal_threads", 1,
        "releases_gil", Py_True, "openmp", Py_False, "fast_math", Py_False);
}

PyMethodDef methods[] = {
    {"intensity_block", intensity_block, METH_VARARGS,
     "Fill a real intensity vector and analytic Jacobian from checked packed site buffers."},
    {"build_info", build_info, METH_NOARGS, "Return compiler, architecture and thread policy."},
    {nullptr, nullptr, 0, nullptr}
};
PyModuleDef module = {PyModuleDef_HEAD_INIT, "_intensity_cpp",
                     "Single-thread compensated native intensity/derivative accumulation.", -1, methods};

}  // namespace

PyMODINIT_FUNC PyInit__intensity_cpp() { return PyModule_Create(&module); }
