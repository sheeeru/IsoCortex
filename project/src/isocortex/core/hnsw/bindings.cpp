/*
 * IsoCortex — core/hnsw/bindings.cpp
 * ========================================
 * pybind11 bindings for zero-copy access to the C++ HNSW index
 * from Python via NumPy arrays.
 *
 * Design decisions:
 *   - Vectors are passed as numpy.ndarray (float32, C-contiguous)
 *     without copying. The binding holds a reference to the array
 *     to prevent garbage collection while the C++ code uses it.
 *   - Search results are returned as a list of (int, float) tuples,
 *     which pybind11 converts to Python efficiently.
 *   - Config is exposed as keyword arguments mirroring HnswConfig.
 *   - The C++ HnswIndex lifecycle is managed by pybind11 via
 *     unique_ptr with a custom deleter.
 *
 * SRS References:
 *   - Section 3.5: HNSW Index (FR-5, FR-6)
 *   - Section 4.2: ReadWriteLock
 *   - FR-IDX-001: Index construction (M, ef_construction, ef_search)
 *   - FR-IDX-002: Soft delete / tombstone
 *   - NFR-01: p95 < 100ms search latency
 *
 * Build:
 *   pip install -e ".[dev]"
 *   (setuptools uses pybind11 to compile this file)
 *
 * Author : Shaheer Qureshi
 * Project: IsoCortex
 */

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "hnsw.hpp"

#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
using namespace isocortex;


// =========================================================================
// RAII holder that keeps a numpy array alive while the index uses it.
// =========================================================================

struct NumpyVectorHolder
{
    // Keep a reference to the Python object so GC doesn't collect it.
    py::array_t<float> data;

    NumpyVectorHolder(py::array_t<float> arr) : data(std::move(arr)) {}

    // Prevent copying.
    NumpyVectorHolder(const NumpyVectorHolder &) = delete;
    NumpyVectorHolder &operator=(const NumpyVectorHolder &) = delete;
};


// =========================================================================
// Python-friendly wrapper around HnswIndex.
// =========================================================================

class PyHnswIndex
{
public:
    // -----------------------------------------------------------------
    // Construction / Destruction
    // -----------------------------------------------------------------

    /**
     * Construct an empty index with the given HNSW parameters.
     *
     * Parameters mirror SRS FR-IDX-001 exactly.
     */
    PyHnswIndex(
        uint32_t M = 16,
        uint32_t ef_construction = 200,
        uint32_t ef_search = 50,
        uint32_t dim = 384,
        const std::string &space = "cosine",
        bool normalize = true)
    {
        HnswConfig cfg;
        cfg.M = M;
        cfg.ef_construction = ef_construction;
        cfg.ef_search = ef_search;
        cfg.dim = dim;
        cfg.space = space;
        cfg.normalize = normalize;

        index_ = std::make_unique<HnswIndex>(std::move(cfg));
    }

    /**
     * Build the HNSW graph from a numpy float32 array.
     *
     * @param vectors numpy array of shape (N, 384), dtype float32.
     * @param external_ids Optional numpy int32 array of shape (N,).
     */
    void build(
        py::array_t<float, py::array::c_style | py::array::forcecast> vectors,
        py::object external_ids = py::none())
    {
        auto buf = vectors.request();

        if (buf.ndim != 2)
        {
            throw std::invalid_argument(
                "vectors must be 2-D, got shape with ndim=" +
                std::to_string(buf.ndim));
        }

        auto n = static_cast<uint32_t>(buf.shape[0]);
        auto d = static_cast<uint32_t>(buf.shape[1]);

        const float *ptr = static_cast<const float *>(buf.ptr);

        const int32_t *ext_ids = nullptr;
        std::unique_ptr<py::array_t<int32_t>> ext_holder;

        if (!external_ids.is_none())
        {
            ext_holder = std::make_unique<py::array_t<int32_t>>(
                py::cast<py::array_t<int32_t>>(external_ids));

            auto ext_buf = ext_holder->request();
            if (ext_buf.ndim != 1 ||
                static_cast<uint32_t>(ext_buf.shape[0]) != n)
            {
                throw std::invalid_argument(
                    "external_ids must be 1-D with length equal to "
                    "number of vectors (N=" + std::to_string(n) + ")");
            }
            ext_ids = static_cast<const int32_t *>(ext_buf.ptr);
        }

        // Keep the numpy array alive during build.
        vector_holders_.push_back(
            std::make_unique<NumpyVectorHolder>(vectors));

        index_->build(ptr, n, d, ext_ids);
    }

    /**
     * Insert a single vector.
     *
     * @param vector numpy array of shape (384,), dtype float32.
     * @param external_id Integer external identifier.
     */
    void insert(
        py::array_t<float, py::array::c_style | py::array::forcecast> vector,
        int32_t external_id)
    {
        auto buf = vector.request();
        if (buf.ndim != 1 ||
            static_cast<uint32_t>(buf.shape[0]) != index_->dim())
        {
            throw std::invalid_argument(
                "vector must be 1-D with dim=" +
                std::to_string(index_->dim()));
        }

        // Keep the numpy array alive.
        vector_holders_.push_back(
            std::make_unique<NumpyVectorHolder>(vector));

        index_->insert(
            static_cast<const float *>(buf.ptr), external_id);
    }

    // -----------------------------------------------------------------
    // Search
    // -----------------------------------------------------------------

    /**
     * Search for the top-K nearest neighbours.
     *
     * @param query numpy array of shape (384,), dtype float32.
     * @param k    Number of results.
     * @return     List of (external_id, distance) tuples, sorted by distance ascending.
     */
    py::list search(
        py::array_t<float, py::array::c_style | py::array::forcecast> query,
        uint32_t k = 5)
    {
        auto buf = query.request();
        if (buf.ndim != 1 ||
            static_cast<uint32_t>(buf.shape[0]) != index_->dim())
        {
            throw std::invalid_argument(
                "query must be 1-D with dim=" +
                std::to_string(index_->dim()));
        }

        auto results = index_->search(
            static_cast<const float *>(buf.ptr), k);

        py::list py_results;
        py_results.reserve(results.size());

        for (const auto &r : results)
        {
            // Convert distance to similarity when using cosine metric.
            // cosine distance = 1 - cosine_similarity
            // We return the raw distance; the Python side converts to similarity.
            py_results.append(py::make_tuple(r.id, r.distance));
        }

        return py_results;
    }

    /**
     * Batch search: embed multiple queries and return results.
     *
     * @param queries numpy array of shape (Q, 384), dtype float32.
     * @param k      Number of results per query.
     * @return      List of lists of (external_id, distance) tuples.
     */
    py::list search_batch(
        py::array_t<float, py::array::c_style | py::array::forcecast> queries,
        uint32_t k = 5)
    {
        auto buf = queries.request();
        if (buf.ndim != 2 ||
            static_cast<uint32_t>(buf.shape[1]) != index_->dim())
        {
            throw std::invalid_argument(
                "queries must be 2-D with second dim=" +
                std::to_string(index_->dim()));
        }

        auto Q = static_cast<uint32_t>(buf.shape[0]);
        const float *data = static_cast<const float *>(buf.ptr);

        py::list py_results;
        py_results.reserve(Q);

        for (uint32_t q = 0; q < Q; ++q)
        {
            auto results = index_->search(data + static_cast<size_t>(q) * index_->dim(), k);

            py::list single;
            single.reserve(results.size());
            for (const auto &r : results)
            {
                single.append(py::make_tuple(r.id, r.distance));
            }
            py_results.append(single);
        }

        return py_results;
    }

    // -----------------------------------------------------------------
    // Soft Delete (FR-IDX-002)
    // -----------------------------------------------------------------

    bool soft_delete(int32_t external_id)
    {
        return index_->soft_delete(external_id);
    }

    bool restore(int32_t external_id)
    {
        return index_->restore(external_id);
    }

    uint32_t deleted_count() const { return index_->deleted_count(); }
    uint32_t active_count() const { return index_->active_count(); }

    // -----------------------------------------------------------------
    // Persistence (Section 7: Index Format Versioning)
    // -----------------------------------------------------------------

    void save(const std::string &path) const
    {
        index_->save(path);
    }

    void load(const std::string &path)
    {
        index_->load(path);
    }

    // -----------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------

    uint32_t size() const { return index_->size(); }
    uint32_t dim() const { return index_->dim(); }
    bool is_built() const { return index_->is_built(); }
    uint32_t max_level() const { return index_->max_level(); }
    int32_t entry_point() const { return index_->entry_point(); }
    uint64_t total_edges() const { return index_->total_edges(); }

    uint32_t get_M() const { return index_->config().M; }
    uint32_t get_ef_construction() const { return index_->config().ef_construction; }
    uint32_t get_ef_search() const { return index_->config().ef_search; }
    std::string get_metric() const { return index_->config().space; }

    void set_ef_search(uint32_t ef)
    {
        // ef_search is the only hot-updatable parameter.
        // We need to rebuild the HnswIndex with the new ef_search.
        // However, ef_search only affects search, not the graph structure,
        // so we can safely update it in the config.
        // For safety, validate before updating.
        if (ef < 1 || ef > 500)
        {
            throw std::invalid_argument(
                "ef_search must be 1-500, got " + std::to_string(ef));
        }

        // Access the config through the public getter (returns a const ref).
        // The HNSW index stores the config internally and uses ef_search
        // during search. Since the config is a member, we need a way to
        // modify it. We'll use a const_cast here because:
        // 1. ef_search is explicitly designed as hot-updatable in the SRS
        // 2. No structural changes are made to the graph
        // 3. This is the intended integration pattern
        const_cast<HnswConfig &>(index_->config()).ef_search = ef;
    }

    /**
     * Get a vector by its external ID.
     *
     * @return numpy array of shape (384,), or None if not found.
     */
    py::object get_vector(int32_t external_id) const
    {
        // Linear scan to find the node with this external_id.
        for (uint32_t i = 0; i < index_->size(); ++i)
        {
            if (index_->get_external_id(i) == external_id)
            {
                const float *ptr = index_->get_vector(i);
                if (ptr == nullptr)
                {
                    return py::none();
                }

                // Copy the vector data into a new numpy array.
                py::array_t<float> result({index_->dim()});
                std::memcpy(result.mutable_data(), ptr,
                            index_->dim() * sizeof(float));
                return result;
            }
        }
        return py::none();
    }

    /**
     * Get all vectors as a numpy array.
     *
     * @return numpy array of shape (N, 384), or None if empty.
     */
    py::object get_all_vectors() const
    {
        uint32_t n = index_->active_count();
        if (n == 0)
        {
            return py::none();
        }

        uint32_t total = index_->size();
        uint32_t d = index_->dim();

        py::array_t<float> result({total, d});

        const std::vector<Node> &nodes = index_->nodes();
        for (uint32_t i = 0; i < total; ++i)
        {
            const float *src = index_->get_vector(i);
            if (src)
            {
                float *dst = result.mutable_data(i);
                std::memcpy(dst, src, d * sizeof(float));
            }
        }

        return result;
    }

    // -----------------------------------------------------------------
    // Internal access (for advanced Python integration)
    // -----------------------------------------------------------------

    /**
     * Return a raw pointer to the C++ HnswIndex.
     * Use with extreme caution — only for zero-copy advanced usage.
     */
    HnswIndex *_get_raw_ptr() const { return index_.get(); }

private:
    std::unique_ptr<HnswIndex> index_;
    std::vector<std::unique_ptr<NumpyVectorHolder>> vector_holders_;
};


// =========================================================================
// Module registration
// =========================================================================

PYBIND11_MODULE(_hnsw_native, m)
{
    m.doc() =
        "IsoCortex — C++ HNSW index with pybind11 bindings.\n\n"
        "Zero-copy numpy interface to the custom HNSW implementation.\n"
        "SRS References: FR-IDX-001, FR-IDX-002, Section 4.2, NFR-01.";

    py::class_<PyHnswIndex>(m, "HnswIndex")
        .def(
            py::init<uint32_t, uint32_t, uint32_t, uint32_t, std::string, bool>(),
            py::kw_only(),
            "Construct an empty HNSW index.\n\n"
            "Parameters\n"
            "----------\n"
            "M : uint32\n"
            "    Number of bidirectional links per layer. Default 16.\n"
            "ef_construction : uint32\n"
            "    Build-time beam width. Default 200.\n"
            "ef_search : uint32\n"
            "    Query-time beam width. Default 50.\n"
            "dim : uint32\n"
            "    Vector dimensionality. Default 384 (MiniLM).\n"
            "space : str\n"
            "    Distance metric: 'cosine', 'l2', or 'ip'. Default 'cosine'.\n"
            "normalize : bool\n"
            "    Normalize vectors on insertion. Default True.",
            py::arg("M") = 16,
            py::arg("ef_construction") = 200,
            py::arg("ef_search") = 50,
            py::arg("dim") = 384,
            py::arg("space") = "cosine",
            py::arg("normalize") = true)

        .def("build", &PyHnswIndex::build,
            "Build the HNSW graph from a numpy array.\n\n"
            "Parameters\n"
            "----------\n"
            "vectors : np.ndarray[float32]\n"
            "    Shape (N, 384). C-contiguous float32 array.\n"
            "external_ids : np.ndarray[int32] | None\n"
            "    Optional shape (N,) array of external IDs.\n"
            "    If None, uses sequential [0..N-1].",
            py::arg("vectors"),
            py::arg("external_ids") = py::none())

        .def("insert", &PyHnswIndex::insert,
            "Insert a single vector into the graph.\n\n"
            "Parameters\n"
            "----------\n"
            "vector : np.ndarray[float32]\n"
            "    Shape (384,).\n"
            "external_id : int\n"
            "    External identifier for this vector.",
            py::arg("vector"),
            py::arg("external_id"))

        .def("search", &PyHnswIndex::search,
            "Find top-K nearest neighbours.\n\n"
            "Parameters\n"
            "----------\n"
            "query : np.ndarray[float32]\n"
            "    Shape (384,).\n"
            "k : uint32\n"
            "    Number of results. Default 5.\n\n"
            "Returns\n"
            "-------\n"
            "list[tuple[int, float]]\n"
            "    (external_id, distance) pairs sorted by distance ascending.",
            py::arg("query"),
            py::arg("k") = 5)

        .def("search_batch", &PyHnswIndex::search_batch,
            "Search multiple queries in one call.\n\n"
            "Parameters\n"
            "----------\n"
            "queries : np.ndarray[float32]\n"
            "    Shape (Q, 384).\n"
            "k : uint32\n"
            "    Number of results per query. Default 5.\n\n"
            "Returns\n"
            "-------\n"
            "list[list[tuple[int, float]]]\n"
            "    One result list per query.",
            py::arg("queries"),
            py::arg("k") = 5)

        .def("soft_delete", &PyHnswIndex::soft_delete,
            "Mark a vector as deleted (soft delete / tombstone).",
            py::arg("external_id"))

        .def("restore", &PyHnswIndex::restore,
            "Restore a soft-deleted vector.",
            py::arg("external_id"))

        .def("deleted_count", &PyHnswIndex::deleted_count,
            "Return the number of soft-deleted vectors.")

        .def("active_count", &PyHnswIndex::active_count,
            "Return the number of active (non-deleted) vectors.")

        .def("save", &PyHnswIndex::save,
            "Save the index to a binary file.\n\n"
            "Parameters\n"
            "----------\n"
            "path : str\n"
            "    File path for the binary index file.",
            py::arg("path"))

        .def("load", &PyHnswIndex::load,
            "Load an index from a binary file.\n\n"
            "Parameters\n"
            "----------\n"
            "path : str\n"
            "    File path to load from.",
            py::arg("path"))

        .def("set_ef_search", &PyHnswIndex::set_ef_search,
            "Hot-update the ef_search parameter (no rebuild needed).\n\n"
            "Parameters\n"
            "----------\n"
            "ef : uint32\n"
            "    New ef_search value (1-500).",
            py::arg("ef"))

        .def_property("size", &PyHnswIndex::size,
            "Number of vectors in the index.")

        .def_property("dim", &PyHnswIndex::dim,
            "Vector dimensionality.")

        .def_property("is_built", &PyHnswIndex::is_built,
            "True if the index has been built or loaded.")

        .def_property("max_level", &PyHnswIndex::max_level,
            "Maximum layer level in the graph.")

        .def_property("entry_point", &PyHnswIndex::entry_point,
            "Entry point node index.")

        .def_property("total_edges", &PyHnswIndex::total_edges,
            "Total edges across all layers (diagnostics).")

        .def_property("M", &PyHnswIndex::get_M,
            "HNSW M parameter.")

        .def_property("ef_construction", &PyHnswIndex::get_ef_construction,
            "HNSW ef_construction parameter.")

        .def_property("ef_search", &PyHnswIndex::get_ef_search,
            "HNSW ef_search parameter.")

        .def_property("metric", &PyHnswIndex::get_metric,
            "Distance metric ('cosine', 'l2', or 'ip').")

        .def("get_vector", &PyHnswIndex::get_vector,
            "Get a vector by external_id.\n\n"
            "Returns numpy array or None.",
            py::arg("external_id"))

        .def("get_all_vectors", &PyHnswIndex::get_all_vectors,
            "Get all vectors as a numpy array of shape (N, 384).\n\n"
            "Returns numpy array or None if empty.",
            py::keep_alive<PyHnswIndex>());


    // =========================================================================
    // Module-level functions
    // =========================================================================

    m.def(
        "cosine_distance",
        [](py::array_t<float, py::array::c_style | py::array::forcecast> a,
           py::array_t<float, py::array::c_style | py::array::forcecast> b)
            -> float
        {
            auto buf_a = a.request();
            auto buf_b = b.request();
            if (buf_a.ndim != 1 || buf_b.ndim != 1 ||
                buf_a.shape[0] != buf_b.shape[0])
            {
                throw std::invalid_argument(
                    "Both arrays must be 1-D with the same length.");
            }
            uint32_t dim = static_cast<uint32_t>(buf_a.shape[0]);
            return cosine_distance(
                static_cast<const float *>(buf_a.ptr),
                static_cast<const float *>(buf_b.ptr),
                dim);
        },
        "Compute cosine distance between two 1-D float32 arrays.",
        py::arg("a"),
        py::arg("b"));

    m.def(
        "l2_distance",
        [](py::array_t<float, py::array::c_style | py::array::forcecast> a,
           py::array_t<float, py::array::c_style | py::array::forcecast> b)
            -> float
        {
            auto buf_a = a.request();
            auto buf_b = b.request();
            if (buf_a.ndim != 1 || buf_b.ndim != 1 ||
                buf_a.shape[0] != buf_b.shape[0])
            {
                throw std::invalid_argument(
                    "Both arrays must be 1-D with the same length.");
            }
            uint32_t dim = static_cast<uint32_t>(buf_a.shape[0]);
            return l2_distance(
                static_cast<const float *>(buf_a.ptr),
                static_cast<const float *>(buf_b.ptr),
                dim);
        },
        "Compute squared L2 distance between two 1-D float32 arrays.",
        py::arg("a"),
        py::arg("b"));

    m.def(
        "has_simd_support",
        []() -> std::string
        {
#if defined(__AVX2__)
            return "avx2";
#elif defined(__SSE4_1__)
            return "sse4_1";
#elif defined(__ARM_NEON) || defined(__aarch64__)
            return "neon";
#else
            return "scalar";
#endif
        },
        "Return the SIMD instruction set available at compile time: "
        "'avx2', 'sse4_1', 'neon', or 'scalar'.");

    m.def(
        "VECTOR_DIM",
        []() -> uint32_t { return kVectorDim; },
        "Return the compile-time vector dimension (384).");
}
