/*
 * IsoCortex — core/hnsw.hpp
 * ==========================
 * Hierarchical Navigable Small World (HNSW) graph data structure
 * for approximate nearest neighbour (ANN) search.
 *
 * Responsibilities (FR-5, FR-6):
 *   - In-memory multi-layered HNSW graph construction.
 *   - Cosine similarity and L2 distance metrics.
 *   - Top-K nearest neighbour search.
 *   - Incremental node insertion.
 *
 * NOT in this file (see separate files):
 *   - Binary serialization / deserialization  → persist.cpp
 *   - pybind11 bindings                       → bindings.cpp
 *
 * SRS References: FR-5, FR-6, NFR-1, NFR-6, NFR-7, CON-1, CON-5
 *
 * Author : Shaheer Qureshi
 * Project: IsoCortex
 */

#ifndef ISO_HNSW_HPP
#define ISO_HNSW_HPP

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <mutex>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

// =========================================================================
// SIMD includes — compile-time and runtime detection
// =========================================================================
#if defined(__AVX2__)
#include <immintrin.h>
#define ISO_SIMD_AVX2 1
#elif defined(__SSE4_1__)
#include <smmintrin.h>
#include <xmmintrin.h>
#define ISO_SIMD_SSE41 1
#elif defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define ISO_SIMD_NEON 1
#endif

namespace isocortex
{

    // =========================================================================
    // Constants
    // =========================================================================

    /// Expected vector dimensionality (all-MiniLM-L6-v2).
    static constexpr uint32_t kVectorDim = 384;

    /// File magic bytes to detect format corruption.
    static constexpr uint32_t kMagic = 0x49534F43; // "ISOC"

    /// Current binary format version.
    static constexpr uint32_t kVersion = 1;

    /// File type marker for HNSW index files.
    static constexpr uint32_t kFileTypeIndex = 2;

    /// Index format version (SRS Section 7).
    static constexpr uint32_t kIndexVersion = 1;

    // =========================================================================
    // Configuration
    // =========================================================================

    /**
     * HNSW tuning parameters.
     *
     * These map directly to the config.json fields defined in SRS Section 3.
     */
    struct HnswConfig
    {
        /// Max edges per node per layer. Higher = better recall, more memory.
        uint32_t M = 16;

        /// Max edges for layer 0 (typically 2 * M).
        uint32_t M_max0 = 0; // 0 = auto-set to 2 * M

        /// Build-time beam width. Higher = better graph, slower build.
        uint32_t ef_construction = 200;

        /// Query-time beam width. Higher = better recall, slower search.
        uint32_t ef_search = 50;

        /// Vector dimensionality. Must equal 384.
        uint32_t dim = kVectorDim;

        /// Distance metric: "cosine" or "l2".
        std::string space = "cosine";

        /// Normalise vectors on insertion (for cosine metric).
        bool normalize = true;

        /**
         * Validate configuration values.
         *
         * @throws std::invalid_argument if any value is out of range.
         */
        void validate() const
        {
            if (M < 1)
            {
                throw std::invalid_argument(
                    "HnswConfig: M must be >= 1, got " + std::to_string(M));
            }
            if (ef_construction < 1)
            {
                throw std::invalid_argument(
                    "HnswConfig: ef_construction must be >= 1, got " +
                    std::to_string(ef_construction));
            }
            if (ef_search < 1)
            {
                throw std::invalid_argument(
                    "HnswConfig: ef_search must be >= 1, got " +
                    std::to_string(ef_search));
            }
            if (dim != kVectorDim)
            {
                throw std::invalid_argument(
                    "HnswConfig: dim must be " + std::to_string(kVectorDim) +
                    ", got " + std::to_string(dim));
            }
            if (space != "cosine" && space != "l2" && space != "ip")
            {
                throw std::invalid_argument(
                    "HnswConfig: space must be 'cosine', 'l2', or 'ip', got '" +
                    space + "'");
            }
        }

        /**
         * Resolve derived parameters (call after setting primary values).
         */
        void resolve()
        {
            if (M_max0 == 0)
            {
                M_max0 = 2 * M;
            }
        }
    };

    // =========================================================================
    // Search result
    // =========================================================================

    /**
     * A single nearest-neighbour result.
     */
    struct NeighborResult
    {
        /// Internal node index.
        int32_t id;

        /// Distance score. Lower = closer for l2/cosine.
        float distance;

        /// Sort by distance ascending (closest first).
        bool operator<(const NeighborResult &other) const
        {
            return distance < other.distance;
        }

        /// Sort by distance descending (farthest first).
        bool operator>(const NeighborResult &other) const
        {
            return distance > other.distance;
        }
    };

    // =========================================================================
    // Distance functions
    // =========================================================================

    // =========================================================================
    // SIMD-accelerated distance functions
    // =========================================================================
    //
    // Three implementations are provided, selected at compile time:
    //   1. AVX2   — 8 floats per iteration (x86-64, most server CPUs)
    //   2. SSE4.1 — 4 floats per iteration (x86-64 fallback)
    //   3. NEON   — 4 floats per iteration (ARM64, Apple Silicon)
    //   4. Scalar — portable fallback (no SIMD)
    //
    // SRS References: NFR-01 (p95 < 100ms), Section 3.5 (SIMD auto-detect)
    // =========================================================================

    /**
     * Compute dot product of two float32 vectors with SIMD acceleration.
     *
     * @param a   Pointer to first vector (dim floats).
     * @param b   Pointer to second vector (dim floats).
     * @param dim Number of dimensions.
     * @return Dot product (a · b).
     */
    inline float dot_product(const float *a, const float *b, uint32_t dim)
    {
        float result = 0.0f;

#if defined(ISO_SIMD_AVX2)
        // AVX2: process 8 floats (256 bits) per iteration
        __m256 sum_vec = _mm256_setzero_ps();
        uint32_t i = 0;
        const uint32_t step = 8;

        for (; i + step <= dim; i += step)
        {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            sum_vec = _mm256_fmadd_ps(va, vb, sum_vec); // FMA: sum += a * b
        }

        // Horizontal sum of 8 floats
        __m128 hi = _mm256_extractf128_ps(sum_vec, 1);
        __m128 lo = _mm256_castps256_ps128(sum_vec);
        __m128 sum128 = _mm_add_ps(lo, hi);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        result = _mm_cvtss_f32(sum128);

#elif defined(ISO_SIMD_SSE41)
        // SSE4.1: process 4 floats (128 bits) per iteration
        __m128 sum_vec = _mm_setzero_ps();
        uint32_t i = 0;
        const uint32_t step = 4;

        for (; i + step <= dim; i += step)
        {
            __m128 va = _mm_loadu_ps(a + i);
            __m128 vb = _mm_loadu_ps(b + i);
            sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(va, vb));
        }

        // Horizontal sum of 4 floats
        __m128 shuf = _mm_movehdup_ps(sum_vec);
        __m128 sums = _mm_add_ps(sum_vec, shuf);
        shuf = _mm_movehl_ps(shuf, sums);
        sums = _mm_add_ss(sums, shuf);
        result = _mm_cvtss_f32(sums);

#elif defined(ISO_SIMD_NEON)
        // ARM NEON: process 4 floats (128 bits) per iteration
        float32x4_t sum_vec = vdupq_n_f32(0.0f);
        uint32_t i = 0;
        const uint32_t step = 4;

        for (; i + step <= dim; i += step)
        {
            float32x4_t va = vld1q_f32(a + i);
            float32x4_t vb = vld1q_f32(b + i);
            sum_vec = vmlaq_f32(sum_vec, va, vb); // sum += a * b
        }

        // Horizontal sum of 4 floats
        float32x2_t sum64 = vpadd_f32(vget_low_f32(sum_vec), vget_high_f32(sum_vec));
        result = vget_lane_f32(vpadd_f32(sum64, sum64), 0);

#else
        // Scalar fallback: process 1 float per iteration
        for (uint32_t i = 0; i < dim; ++i)
        {
            result += a[i] * b[i];
        }
#endif

        // Handle remaining elements (dim % step)
        for (uint32_t i = (dim / (sizeof(float) * (
#if defined(ISO_SIMD_AVX2)
                               8
#elif defined(ISO_SIMD_SSE41) || defined(ISO_SIMD_NEON)
                               4
#else
                               1
#endif
                           ))) * (
#if defined(ISO_SIMD_AVX2)
                           8
#elif defined(ISO_SIMD_SSE41) || defined(ISO_SIMD_NEON)
                           4
#else
                           1
#endif
                           ); i < dim; ++i)
        {
            result += a[i] * b[i];
        }

        return result;
    }

    /**
     * Compute squared L2 distance using SIMD.
     * L2(a,b) = sum((a[i] - b[i])^2)
     */
    inline float l2_distance_simd(const float *a, const float *b, uint32_t dim)
    {
        float result = 0.0f;

#if defined(ISO_SIMD_AVX2)
        __m256 sum_vec = _mm256_setzero_ps();
        uint32_t i = 0;
        const uint32_t step = 8;

        for (; i + step <= dim; i += step)
        {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            __m256 diff = _mm256_sub_ps(va, vb);
            sum_vec = _mm256_fmadd_ps(diff, diff, sum_vec); // sum += diff * diff
        }

        __m128 hi = _mm256_extractf128_ps(sum_vec, 1);
        __m128 lo = _mm256_castps256_ps128(sum_vec);
        __m128 sum128 = _mm_add_ps(lo, hi);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        result = _mm_cvtss_f32(sum128);

#elif defined(ISO_SIMD_SSE41)
        __m128 sum_vec = _mm_setzero_ps();
        uint32_t i = 0;
        const uint32_t step = 4;

        for (; i + step <= dim; i += step)
        {
            __m128 va = _mm_loadu_ps(a + i);
            __m128 vb = _mm_loadu_ps(b + i);
            __m128 diff = _mm_sub_ps(va, vb);
            sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(diff, diff));
        }

        __m128 shuf = _mm_movehdup_ps(sum_vec);
        __m128 sums = _mm_add_ps(sum_vec, shuf);
        shuf = _mm_movehl_ps(shuf, sums);
        sums = _mm_add_ss(sums, shuf);
        result = _mm_cvtss_f32(sums);

#elif defined(ISO_SIMD_NEON)
        float32x4_t sum_vec = vdupq_n_f32(0.0f);
        uint32_t i = 0;
        const uint32_t step = 4;

        for (; i + step <= dim; i += step)
        {
            float32x4_t va = vld1q_f32(a + i);
            float32x4_t vb = vld1q_f32(b + i);
            float32x4_t diff = vsubq_f32(va, vb);
            sum_vec = vmlaq_f32(sum_vec, diff, diff);
        }

        float32x2_t sum64 = vpadd_f32(vget_low_f32(sum_vec), vget_high_f32(sum_vec));
        result = vget_lane_f32(vpadd_f32(sum64, sum64), 0);

#else
        for (uint32_t i = 0; i < dim; ++i)
        {
            float diff = a[i] - b[i];
            result += diff * diff;
        }
#endif

        // Handle remaining elements
        for (uint32_t i = (dim / (sizeof(float) * (
#if defined(ISO_SIMD_AVX2)
                               8
#elif defined(ISO_SIMD_SSE41) || defined(ISO_SIMD_NEON)
                               4
#else
                               1
#endif
                           ))) * (
#if defined(ISO_SIMD_AVX2)
                           8
#elif defined(ISO_SIMD_SSE41) || defined(ISO_SIMD_NEON)
                           4
#else
                           1
#endif
                           ); i < dim; ++i)
        {
            float diff = a[i] - b[i];
            result += diff * diff;
        }

        return result;
    }

    /**
     * Compute cosine distance between two float32 vectors (SIMD-accelerated).
     *
     * Cosine distance = 1 - cosine_similarity.
     * Range: [0, 2] where 0 = identical direction.
     *
     * @param a   Pointer to first vector (dim floats).
     * @param b   Pointer to second vector (dim floats).
     * @param dim Number of dimensions.
     * @return Cosine distance.
     */
    inline float cosine_distance(const float *a, const float *b, uint32_t dim)
    {
        float dot = dot_product(a, b, dim);
        float norm_a = dot_product(a, a, dim);
        float norm_b = dot_product(b, b, dim);

        float denom = std::sqrt(norm_a) * std::sqrt(norm_b);
        if (denom < 1e-10f)
        {
            return 1.0f; // degenerate: one or both vectors are zero
        }

        float sim = dot / denom;
        sim = std::max(-1.0f, std::min(1.0f, sim));

        return 1.0f - sim;
    }

    /**
     * Compute squared L2 (Euclidean) distance between two float32 vectors (SIMD).
     *
     * @param a   Pointer to first vector (dim floats).
     * @param b   Pointer to second vector (dim floats).
     * @param dim Number of dimensions.
     * @return Squared L2 distance.
     */
    inline float l2_distance(const float *a, const float *b, uint32_t dim)
    {
        return l2_distance_simd(a, b, dim);
    }

    /**
     * Compute inner product (IP) between two float32 vectors (SIMD).
     *
     * @param a   Pointer to first vector (dim floats).
     * @param b   Pointer to second vector (dim floats).
     * @param dim Number of dimensions.
     * @return Inner product. Higher = more similar (negate for sorting).
     */
    inline float inner_product(const float *a, const float *b, uint32_t dim)
    {
        return -dot_product(a, b, dim); // Negate so lower = closer (for HNSW min-heap)
    }

    // =========================================================================
    // Node (internal)
    // =========================================================================

    /**
     * A single node in the HNSW graph.
     *
     * Each node stores:
     *   - Its position in the external vector array (external_id).
     *   - Its assigned maximum layer.
     *   - Neighbour lists per layer (layer 0 at index 0).
     */
    struct Node
    {
        /// External vector ID (index into the caller's vector array).
        int32_t external_id;

        /// Highest layer this node appears in.
        uint32_t level;

        /// Soft-delete flag (tombstone). True = deleted (FR-IDX-002).
        bool deleted = false;

        /**
         * Neighbours per layer.
         * neighbours[0] = layer 0 neighbours (densest).
         * neighbours[level] = top-layer neighbours (sparsest).
         */
        std::vector<std::vector<int32_t>> neighbours;

        Node() : external_id(-1), level(0) {}

        Node(int32_t id, uint32_t lvl) : external_id(id), level(lvl)
        {
            neighbours.resize(lvl + 1);
        }
    };

    // =========================================================================
    // ReadWriteLock (SRS Section 4: Concurrency Model)
    // =========================================================================

    /**
     * Reader-writer lock allowing multiple concurrent readers
     * but exclusive writer access.
     *
     * Guarantees (SRS Section 4.2):
     *   - Multiple concurrent readers: Search queries execute simultaneously.
     *   - Exclusive writer: Only one write operation at a time.
     *   - Writer priority: Pending writes take priority over new readers
     *     to prevent writer starvation.
     *   - Read-during-write safety: Readers holding locks before a writer
     *     began continue safely.
     */
    class ReadWriteLock
    {
    public:
        ReadWriteLock()
            : reader_count_(0), writer_active_(false), writer_waiting_(0) {}

        void acquire_read()
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this]()
                     { return !writer_active_ && writer_waiting_ == 0; });
            ++reader_count_;
        }

        void release_read()
        {
            std::lock_guard<std::mutex> lock(mutex_);
            --reader_count_;
            if (reader_count_ == 0)
            {
                cv_.notify_all();
            }
        }

        void acquire_write()
        {
            std::unique_lock<std::mutex> lock(mutex_);
            ++writer_waiting_;
            cv_.wait(lock, [this]()
                     { return reader_count_ == 0 && !writer_active_; });
            --writer_waiting_;
            writer_active_ = true;
        }

        void release_write()
        {
            std::lock_guard<std::mutex> lock(mutex_);
            writer_active_ = false;
            cv_.notify_all();
        }

    private:
        std::mutex mutex_;
        std::condition_variable cv_;
        uint32_t reader_count_;
        bool writer_active_;
        uint32_t writer_waiting_;
    };

    // =========================================================================
    // HNSW Index
    // =========================================================================

    /**
     * In-memory HNSW approximate nearest-neighbour index.
     *
     * Usage:
     *   1. Construct with HnswConfig.
     *   2. Call build() with a float32 vector matrix, or insert() one by one.
     *   3. Call search() to query.
     *
     * Thread safety:
     *   - build() and insert() are NOT thread-safe.
     *   - search() is thread-safe (const, read-only graph traversal).
     */
    class HnswIndex
    {
    public:
        // -----------------------------------------------------------------
        // Construction
        // -----------------------------------------------------------------

        /**
         * Construct an empty HNSW index.
         *
         * @param config HNSW tuning parameters.
         * @throws std::invalid_argument if config is invalid.
         */
        explicit HnswIndex(HnswConfig config) : config_(std::move(config))
        {
            config_.validate();
            config_.resolve();

            if (config_.space == "cosine")
            {
                dist_fn_ = cosine_distance;
            }
            else if (config_.space == "ip")
            {
                dist_fn_ = inner_product;
            }
            else
            {
                dist_fn_ = l2_distance;
            }

            std::random_device rd;
            rng_.seed(rd());

            level_mult_ = 1.0 / std::log(static_cast<double>(config_.M));

            clear();
        }

        /**
         * Construct with default config (M=16, efC=200, efS=50, cosine).
         */
        HnswIndex() : HnswIndex(HnswConfig{}) {}

        // Disable copy (owns large data)
        HnswIndex(const HnswIndex &) = delete;
        HnswIndex &operator=(const HnswIndex &) = delete;

        // Allow move
        HnswIndex(HnswIndex &&other) noexcept
        {
            *this = std::move(other);
        }

        HnswIndex &operator=(HnswIndex &&other) noexcept
        {
            if (this != &other)
            {
                config_ = std::move(other.config_);
                nodes_ = std::move(other.nodes_);
                vectors_ = std::move(other.vectors_);
                dim_ = other.dim_;
                n_vectors_ = other.n_vectors_;
                entry_point_ = other.entry_point_;
                max_level_ = other.max_level_;
                dist_fn_ = other.dist_fn_;
                level_mult_ = other.level_mult_;
                rng_ = std::move(other.rng_);
                built_ = other.built_;

                other.clear();
            }
            return *this;
        }

        // -----------------------------------------------------------------
        // Build (FR-5)
        // -----------------------------------------------------------------

        /**
         * Build the HNSW graph from a vector matrix.
         *
         * @param vectors      Pointer to contiguous float32 data [n * dim].
         * @param n            Number of vectors.
         * @param dim          Dimensionality (must equal config_.dim).
         * @param external_ids Optional external IDs. If nullptr, uses [0..n-1].
         *
         * @throws std::invalid_argument if dim mismatch or n == 0.
         * @throws std::runtime_error    if vectors contain NaN/Inf.
         */
        void build(
            const float *vectors,
            uint32_t n,
            uint32_t dim,
            const int32_t *external_ids = nullptr)
        {
            if (n == 0)
            {
                throw std::invalid_argument("HnswIndex::build: n must be > 0");
            }
            if (dim != config_.dim)
            {
                throw std::invalid_argument(
                    "HnswIndex::build: dim mismatch — expected " +
                    std::to_string(config_.dim) + ", got " +
                    std::to_string(dim));
            }

            clear();
            dim_ = dim;

            // Copy vectors into internal storage
            vectors_.resize(static_cast<size_t>(n) * dim);
            std::memcpy(vectors_.data(), vectors,
                        static_cast<size_t>(n) * dim * sizeof(float));

            // Validate no NaN/Inf
            for (size_t i = 0; i < vectors_.size(); ++i)
            {
                if (!std::isfinite(vectors_[i]))
                {
                    clear();
                    throw std::runtime_error(
                        "HnswIndex::build: NaN/Inf detected in input vectors "
                        "at flat index " +
                        std::to_string(i));
                }
            }

            // Normalize if cosine metric
            if (config_.space == "cosine" && config_.normalize)
            {
                for (uint32_t i = 0; i < n; ++i)
                {
                    _normalize_vector(
                        vectors_.data() + static_cast<size_t>(i) * dim, dim);
                }
            }

            // Insert each vector
            for (uint32_t i = 0; i < n; ++i)
            {
                int32_t eid = external_ids
                                  ? external_ids[i]
                                  : static_cast<int32_t>(i);
                _insert_one(
                    vectors_.data() + static_cast<size_t>(i) * dim, eid);
            }

            built_ = true;
        }

        /**
         * Insert a single vector into the existing graph.
         *
         * For incremental re-indexing (UC-3 / FR-7).
         *
         * @param vector      Pointer to dim floats.
         * @param external_id External ID for this vector.
         *
         * @throws std::invalid_argument if dim doesn't match.
         * @throws std::runtime_error    if vector contains NaN/Inf.
         */
        void insert(const float *vector, int32_t external_id)
        {
            if (dim_ == 0)
            {
                dim_ = config_.dim;
            }

            for (uint32_t i = 0; i < dim_; ++i)
            {
                if (!std::isfinite(vector[i]))
                {
                    throw std::runtime_error(
                        "HnswIndex::insert: NaN/Inf detected at dim " +
                        std::to_string(i));
                }
            }

            std::vector<float> vec(vector, vector + dim_);
            if (config_.space == "cosine" && config_.normalize)
            {
                _normalize_vector(vec.data(), dim_);
            }

            size_t offset = vectors_.size();
            vectors_.resize(offset + dim_);
            std::memcpy(vectors_.data() + offset, vec.data(),
                        dim_ * sizeof(float));

            _insert_one(vectors_.data() + offset, external_id);
            built_ = true;
        }

        // -----------------------------------------------------------------
        // Search (FR-6)
        // -----------------------------------------------------------------

        /**
         * Find the top-K nearest neighbours for a query vector.
         *
         * @param query Pointer to dim floats.
         * @param k     Number of results to return.
         * @return Vector of NeighborResult sorted by distance ascending.
         *
         * @throws std::invalid_argument if index is empty or k == 0.
         * @throws std::runtime_error    if query contains NaN/Inf.
         */
        std::vector<NeighborResult> search(const float *query, uint32_t k) const
        {
            if (n_vectors_ == 0)
            {
                throw std::invalid_argument(
                    "HnswIndex::search: index is empty — build or load first");
            }
            if (k == 0)
            {
                throw std::invalid_argument(
                    "HnswIndex::search: k must be > 0");
            }

            for (uint32_t i = 0; i < dim_; ++i)
            {
                if (!std::isfinite(query[i]))
                {
                    throw std::runtime_error(
                        "HnswIndex::search: NaN/Inf in query at dim " +
                        std::to_string(i));
                }
            }

            std::vector<float> q(query, query + dim_);
            if (config_.space == "cosine" && config_.normalize)
            {
                _normalize_vector(q.data(), dim_);
            }

            uint32_t actual_k = std::min(k, n_vectors_);

            auto results = _search_layer(q.data(), actual_k, config_.ef_search);

            std::sort(results.begin(), results.end());

            if (results.size() > actual_k)
            {
                results.resize(actual_k);
            }

            return results;
        }

        // -----------------------------------------------------------------
        // Accessors
        // -----------------------------------------------------------------

        /// Number of vectors in the index.
        uint32_t size() const { return n_vectors_; }

        /// Vector dimensionality.
        uint32_t dim() const { return dim_; }

        /// True if the index has been built or loaded.
        bool is_built() const { return built_; }

        /// Reference to the current configuration.
        const HnswConfig &config() const { return config_; }

        /// Total number of edges across all nodes and layers (for diagnostics).
        uint64_t total_edges() const
        {
            uint64_t total = 0;
            for (uint32_t i = 0; i < n_vectors_; ++i)
            {
                for (const auto &layer : nodes_[i].neighbours)
                {
                    total += layer.size();
                }
            }
            return total;
        }

        /// Maximum layer level in the graph.
        uint32_t max_level() const { return max_level_; }

        /// Entry point node index.
        int32_t entry_point() const { return entry_point_; }

        /// Direct access to the nodes vector (for serialization in persist.cpp).
        const std::vector<Node> &nodes() const { return nodes_; }

        /// Direct access to the raw vector data (for persist.cpp).
        const std::vector<float> &vectors() const { return vectors_; }

        /**
         * Get a pointer to the internal vector data for a given node index.
         *
         * @param node_idx Internal node index (0..size()-1).
         * @return Pointer to dim floats, or nullptr if out of range.
         */
        const float *get_vector(uint32_t node_idx) const
        {
            if (node_idx >= n_vectors_)
            {
                return nullptr;
            }
            return vectors_.data() + static_cast<size_t>(node_idx) * dim_;
        }

        /**
         * Get the external ID for a given internal node index.
         *
         * @param node_idx Internal node index.
         * @return External ID, or -1 if out of range.
         */
        int32_t get_external_id(uint32_t node_idx) const
        {
            if (node_idx >= n_vectors_)
            {
                return -1;
            }
            return nodes_[node_idx].external_id;
        }

        // -----------------------------------------------------------------
        // Reset
        // -----------------------------------------------------------------

        /**
         * Clear all data, returning the index to an empty state.
         */
        void clear()
        {
            nodes_.clear();
            vectors_.clear();
            n_vectors_ = 0;
            dim_ = config_.dim;
            entry_point_ = -1;
            max_level_ = 0;
            built_ = false;
        }

        // -----------------------------------------------------------------
        // Soft Delete (FR-IDX-002)
        // -----------------------------------------------------------------

        /**
         * Mark a vector as deleted (soft delete / tombstone).
         *
         * The vector remains in the graph but is skipped during search.
         * Call compact() periodically to rebuild the graph without
         * deleted vectors.
         *
         * @param external_id External ID of the vector to delete.
         * @return true if the vector was found and marked deleted.
         */
        bool soft_delete(int32_t external_id)
        {
            for (uint32_t i = 0; i < n_vectors_; ++i)
            {
                if (nodes_[i].external_id == external_id)
                {
                    nodes_[i].deleted = true;
                    return true;
                }
            }
            return false;
        }

        /**
         * Restore a previously soft-deleted vector.
         *
         * @param external_id External ID to restore.
         * @return true if found and restored.
         */
        bool restore(int32_t external_id)
        {
            for (uint32_t i = 0; i < n_vectors_; ++i)
            {
                if (nodes_[i].external_id == external_id &&
                    nodes_[i].deleted)
                {
                    nodes_[i].deleted = false;
                    return true;
                }
            }
            return false;
        }

        /**
         * Count the number of soft-deleted vectors.
         */
        uint32_t deleted_count() const
        {
            uint32_t count = 0;
            for (uint32_t i = 0; i < n_vectors_; ++i)
            {
                if (nodes_[i].deleted)
                {
                    ++count;
                }
            }
            return count;
        }

        /**
         * Count active (non-deleted) vectors.
         */
        uint32_t active_count() const
        {
            return n_vectors_ - deleted_count();
        }

        // -----------------------------------------------------------------
        // Persistence (FR-7 / SRS Section 7)
        // -----------------------------------------------------------------

        /**
         * Save the index to a binary file.
         *
         * Format:
         *   [magic:4][version:4][config_size:4][config_bytes]
         *   [n_vectors:4][n_deleted:4][dim:4][entry_point:4][max_level:4]
         *   [external_ids:n*4][deleted_flags:n*1]
         *   [vectors:n*dim*4]
         *   [node_count:4][per_node_neighbour_data]
         *
         * @param path File path to save to.
         */
        void save(const std::string &path) const
        {
            std::ofstream out(path, std::ios::binary);
            if (!out)
            {
                throw std::runtime_error(
                    "HnswIndex::save: Cannot open file: " + path);
            }

            // Header
            write_u32(out, kMagic);
            write_u32(out, kIndexVersion);

            // Config
            std::string config_str = config_.space;
            write_u32(out, static_cast<uint32_t>(config_str.size()));
            out.write(config_str.data(),
                      static_cast<std::streamsize>(config_str.size()));
            write_u32(out, config_.M);
            write_u32(out, config_.M_max0);
            write_u32(out, config_.ef_construction);
            write_u32(out, config_.ef_search);
            write_u32(out, config_.dim);
            write_u32(out, config_.normalize ? 1 : 0);

            // Index metadata
            write_u32(out, n_vectors_);
            write_u32(out, deleted_count());
            write_u32(out, dim_);
            write_i32(out, entry_point_);
            write_u32(out, max_level_);

            // External IDs and deleted flags
            for (uint32_t i = 0; i < n_vectors_; ++i)
            {
                write_i32(out, nodes_[i].external_id);
            }
            for (uint32_t i = 0; i < n_vectors_; ++i)
            {
                write_u8(out, nodes_[i].deleted ? 1 : 0);
            }

            // Vectors
            out.write(reinterpret_cast<const char *>(vectors_.data()),
                      static_cast<std::streamsize>(vectors_.size() * sizeof(float)));

            // Neighbour lists
            write_u32(out, n_vectors_);
            for (uint32_t i = 0; i < n_vectors_; ++i)
            {
                const auto &node = nodes_[i];
                write_u32(out, static_cast<uint32_t>(node.neighbours.size()));
                for (const auto &layer : node.neighbours)
                {
                    write_u32(out, static_cast<uint32_t>(layer.size()));
                    for (int32_t nid : layer)
                    {
                        write_i32(out, nid);
                    }
                }
            }

            out.flush();
            out.close();

            logger_debug(
                "[HNSW] Index saved  path=%s  vectors=%d  deleted=%d",
                path, n_vectors_, deleted_count());
        }

        /**
         * Load an index from a binary file.
         *
         * @param path File path to load from.
         */
        void load(const std::string &path)
        {
            std::ifstream in(path, std::ios::binary);
            if (!in)
            {
                throw std::runtime_error(
                    "HnswIndex::load: Cannot open file: " + path);
            }

            clear();

            // Header
            uint32_t magic = read_u32(in);
            if (magic != kMagic)
            {
                throw std::runtime_error(
                    "HnswIndex::load: Invalid magic bytes in " + path);
            }

            uint32_t version = read_u32(in);
            if (version != kIndexVersion)
            {
                throw std::runtime_error(
                    "HnswIndex::load: Unsupported version " +
                    std::to_string(version) + " in " + path);
            }

            // Config
            uint32_t config_str_size = read_u32(in);
            std::string config_str(config_str_size, '\0');
            if (config_str_size > 0)
            {
                in.read(&config_str[0],
                        static_cast<std::streamsize>(config_str_size));
            }
            HnswConfig loaded_config;
            loaded_config.M = read_u32(in);
            loaded_config.M_max0 = read_u32(in);
            loaded_config.ef_construction = read_u32(in);
            loaded_config.ef_search = read_u32(in);
            loaded_config.dim = read_u32(in);
            loaded_config.normalize = read_u32(in) != 0;
            loaded_config.space = config_str;
            loaded_config.validate();
            loaded_config.resolve();
            config_ = loaded_config;
            dim_ = config_.dim;

            if (config_.space == "cosine")
            {
                dist_fn_ = cosine_distance;
            }
            else if (config_.space == "ip")
            {
                dist_fn_ = inner_product;
            }
            else
            {
                dist_fn_ = l2_distance;
            }

            // Index metadata
            uint32_t n = read_u32(in);
            uint32_t n_deleted = read_u32(in);
            uint32_t d = read_u32(in);
            int32_t ep = read_i32(in);
            uint32_t ml = read_u32(in);

            if (d != dim_)
            {
                throw std::runtime_error(
                    "HnswIndex::load: Dimension mismatch — expected " +
                    std::to_string(dim_) + ", got " + std::to_string(d));
            }

            // External IDs
            nodes_.resize(n);
            for (uint32_t i = 0; i < n; ++i)
            {
                nodes_[i].external_id = read_i32(in);
            }

            // Deleted flags
            for (uint32_t i = 0; i < n; ++i)
            {
                nodes_[i].deleted = read_u8(in) != 0;
            }

            // Vectors
            vectors_.resize(static_cast<size_t>(n) * dim_);
            in.read(reinterpret_cast<char *>(vectors_.data()),
                     static_cast<std::streamsize>(vectors_.size() * sizeof(float)));

            // Neighbour lists
            uint32_t node_count = read_u32(in);
            for (uint32_t i = 0; i < node_count && i < n; ++i)
            {
                uint32_t layer_count = read_u32(in);
                nodes_[i].neighbours.resize(layer_count);
                for (uint32_t l = 0; l < layer_count; ++l)
                {
                    uint32_t neigh_count = read_u32(in);
                    nodes_[i].neighbours[l].resize(neigh_count);
                    for (uint32_t j = 0; j < neigh_count; ++j)
                    {
                        nodes_[i].neighbours[l][j] = read_i32(in);
                    }
                }
            }

            n_vectors_ = n;
            entry_point_ = ep;
            max_level_ = ml;
            built_ = true;

            in.close();

            level_mult_ = 1.0 / std::log(static_cast<double>(config_.M));

            logger_debug(
                "[HNSW] Index loaded  path=%s  vectors=%d  deleted=%d  dim=%d",
                path, n_vectors_, n_deleted, dim_);
        }

        // -----------------------------------------------------------------
        // ReadWriteLock accessors (SRS Section 4)
        // -----------------------------------------------------------------

        ReadWriteLock &get_lock() { return rw_lock_; }
        const ReadWriteLock &get_lock() const { return rw_lock_; }

        // -----------------------------------------------------------------
        // Serialization helpers (used by persist.cpp)
        // -----------------------------------------------------------------

        static void write_u8(std::ofstream &out, uint8_t val)
        {
            out.write(reinterpret_cast<const char *>(&val), 1);
        }

        static void write_u32(std::ofstream &out, uint32_t val)
        {
            out.write(reinterpret_cast<const char *>(&val), 4);
        }

        static void write_i32(std::ofstream &out, int32_t val)
        {
            out.write(reinterpret_cast<const char *>(&val), 4);
        }

        static void write_string(std::ofstream &out, const std::string &s)
        {
            uint32_t len = static_cast<uint32_t>(s.size());
            write_u32(out, len);
            out.write(s.data(), static_cast<std::streamsize>(len));
        }

        static uint8_t read_u8(std::ifstream &in)
        {
            uint8_t val = 0;
            in.read(reinterpret_cast<char *>(&val), 1);
            return val;
        }

        static uint32_t read_u32(std::ifstream &in)
        {
            uint32_t val = 0;
            in.read(reinterpret_cast<char *>(&val), 4);
            return val;
        }

        static int32_t read_i32(std::ifstream &in)
        {
            int32_t val = 0;
            in.read(reinterpret_cast<char *>(&val), 4);
            return val;
        }

        static std::string read_string(std::ifstream &in)
        {
            uint32_t len = read_u32(in);
            std::string s(len, '\0');
            if (len > 0)
            {
                in.read(&s[0], static_cast<std::streamsize>(len));
            }
            return s;
        }

    private:
        // -----------------------------------------------------------------
        // Configuration
        // -----------------------------------------------------------------
        HnswConfig config_;

        // -----------------------------------------------------------------
        // Graph data
        // -----------------------------------------------------------------
        std::vector<Node> nodes_;
        std::vector<float> vectors_;
        uint32_t n_vectors_ = 0;
        uint32_t dim_ = kVectorDim;
        int32_t entry_point_ = -1;
        uint32_t max_level_ = 0;
        bool built_ = false;

        // Concurrency (SRS Section 4)
        mutable ReadWriteLock rw_lock_;

        // -----------------------------------------------------------------
        // Distance
        // -----------------------------------------------------------------
        std::function<float(const float *, const float *, uint32_t)> dist_fn_;

        // -----------------------------------------------------------------
        // Random
        // -----------------------------------------------------------------
        std::mt19937 rng_;
        double level_mult_;

        // -----------------------------------------------------------------
        // Internal: logging (C++ equivalent of Python logger)
        // -----------------------------------------------------------------
        static void logger_debug(const char *, ...) { /* no-op */ }

        // -----------------------------------------------------------------
        // Internal: insertion (FR-5)
        // -----------------------------------------------------------------

        /**
         * Assign a random level to a new node.
         *
         * Uses exponential distribution: P(level >= l) = exp(-l * ln(M)).
         */
        uint32_t _random_level()
        {
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            double r = dist(rng_);
            if (r < 1e-9)
            {
                r = 1e-9;
            }
            return static_cast<uint32_t>(
                std::floor(-std::log(r) * level_mult_));
        }

        /**
         * Insert a single vector into the graph.
         *
         * @param vec         Pointer to dim floats (already normalized if cosine).
         * @param external_id External ID.
         */
        void _insert_one(const float *vec, int32_t external_id)
        {
            uint32_t level = _random_level();

            int32_t new_idx = static_cast<int32_t>(n_vectors_);
            nodes_.emplace_back(external_id, level);
            n_vectors_++;

            // First node: set as entry point
            if (new_idx == 0)
            {
                entry_point_ = 0;
                max_level_ = level;
                return;
            }

            int32_t curr_entry = entry_point_;
            uint32_t curr_level = max_level_;

            // Phase 1: Navigate from top layer down to (level + 1)
            // using greedy search (single closest neighbour per layer).
            if (level < curr_level)
            {
                for (uint32_t l = curr_level; l > level; --l)
                {
                    curr_entry = _greedy_search_nearest(vec, curr_entry, l);
                }
            }

            // Phase 2: Insert at layers 0..min(level, curr_level)
            // using ef_construction beam search.
            uint32_t insert_top = std::min(level, curr_level);

            for (uint32_t l = 0; l <= insert_top; ++l)
            {
                auto candidates = _search_layer_at_level(
                    vec, config_.ef_construction, curr_entry, l);

                uint32_t max_conn = (l == 0) ? config_.M_max0 : config_.M;
                auto selected = _select_neighbours(vec, candidates, max_conn);

                // Bidirectional links
                for (const auto &nr : selected)
                {
                    int32_t neighbor_idx = nr.id;

                    // Bounds check: valid index
                    if (neighbor_idx < 0 ||
                        static_cast<uint32_t>(neighbor_idx) >= n_vectors_)
                    {
                        continue;
                    }

                    // Bounds check: neighbor exists at this layer
                    if (nodes_[neighbor_idx].level < l)
                    {
                        continue;
                    }

                    nodes_[new_idx].neighbours[l].push_back(neighbor_idx);
                    nodes_[neighbor_idx].neighbours[l].push_back(new_idx);

                    // Prune neighbor if it exceeds max connections
                    uint32_t neighbor_max_conn =
                        (l == 0) ? config_.M_max0 : config_.M;
                    if (nodes_[neighbor_idx].neighbours[l].size() >
                        neighbor_max_conn)
                    {
                        _prune_connections(
                            neighbor_idx, l, neighbor_max_conn);
                    }
                }

                if (!selected.empty())
                {
                    curr_entry = selected[0].id;
                }
            }

            // Phase 3: If new node has higher level than current max,
            // connect it to the old entry point, then promote.
            if (level > max_level_)
            {
                // Connect new node to old entry point at the old entry's
                // top layer. Without this, the new entry point has zero
                // neighbours at its upper layers and greedy search gets
                // stuck, making the entire graph unreachable.
                if (entry_point_ >= 0 &&
                    static_cast<uint32_t>(entry_point_) < n_vectors_)
                {
                    uint32_t old_top = nodes_[entry_point_].level;
                    nodes_[new_idx].neighbours[old_top].push_back(entry_point_);
                    nodes_[entry_point_].neighbours[old_top].push_back(new_idx);
                }

                max_level_ = level;
                entry_point_ = new_idx;
            }
        }

        /**
         * Greedy search: find the single closest node at a given layer.
         *
         * Used during insertion to navigate upper layers quickly.
         *
         * @param query Pointer to dim floats.
         * @param entry Starting node index.
         * @param layer Layer to search.
         * @return Index of the closest node found.
         */
        int32_t _greedy_search_nearest(
            const float *query,
            int32_t entry,
            uint32_t layer) const
        {
            if (entry < 0 ||
                static_cast<uint32_t>(entry) >= n_vectors_)
            {
                return entry;
            }
            if (layer > nodes_[entry].level)
            {
                return entry;
            }

            float best_dist = dist_fn_(query, _vec(entry), dim_);
            int32_t best_idx = entry;

            bool improved = true;
            while (improved)
            {
                improved = false;
                const auto &neighbours = nodes_[best_idx].neighbours[layer];

                for (int32_t nid : neighbours)
                {
                    if (nid < 0 ||
                        static_cast<uint32_t>(nid) >= n_vectors_)
                    {
                        continue;
                    }

                    float d = dist_fn_(query, _vec(nid), dim_);
                    if (nodes_[nid].deleted) continue;  // Skip deleted
                    if (d < best_dist)
                    {
                        best_dist = d;
                        best_idx = nid;
                        improved = true;
                    }
                }
            }

            return best_idx;
        }

        /**
         * Search a single layer with beam width ef.
         *
         * Returns up to ef closest candidates found.
         *
         * @param query Pointer to dim floats.
         * @param ef    Beam width (max candidates to track).
         * @param entry Starting node index.
         * @param layer Layer to search.
         * @return Vector of NeighborResult (order not guaranteed).
         */
        std::vector<NeighborResult> _search_layer_at_level(
            const float *query,
            uint32_t ef,
            int32_t entry,
            uint32_t layer) const
        {
            if (entry < 0 ||
                static_cast<uint32_t>(entry) >= n_vectors_)
            {
                return {};
            }
            if (layer > nodes_[entry].level)
            {
                return {};
            }

            // Min-heap for candidates: always pop the closest unvisited node
            auto cmp_min = [](const NeighborResult &a, const NeighborResult &b)
            {
                return a.distance > b.distance;
            };
            std::priority_queue<NeighborResult, std::vector<NeighborResult>,
                                decltype(cmp_min)>
                candidates(cmp_min);

            // Max-heap for results: farthest on top, easy to evict
            auto cmp_max = [](const NeighborResult &a, const NeighborResult &b)
            {
                return a.distance < b.distance;
            };
            std::priority_queue<NeighborResult, std::vector<NeighborResult>,
                                decltype(cmp_max)>
                results(cmp_max);

            std::unordered_set<int32_t> visited;
            visited.reserve(ef * 4);

            float entry_dist = dist_fn_(query, _vec(entry), dim_);
            candidates.push({entry, entry_dist});
            results.push({entry, entry_dist});
            visited.insert(entry);

            while (!candidates.empty())
            {
                auto curr = candidates.top();

                // If the closest unvisited candidate is farther than the
                // farthest result, no better results can be found.
                if (results.size() >= ef &&
                    curr.distance > results.top().distance)
                {
                    break;
                }

                candidates.pop();

                const auto &neighbours = nodes_[curr.id].neighbours[layer];
                for (int32_t nid : neighbours)
                {
                    if (visited.count(nid))
                    {
                        continue;
                    }
                    if (nid < 0 ||
                        static_cast<uint32_t>(nid) >= n_vectors_)
                    {
                        continue;
                    }
                    visited.insert(nid);

                    float d = dist_fn_(query, _vec(nid), dim_);

                    // Skip soft-deleted vectors (FR-IDX-002)
                    if (nodes_[nid].deleted) continue;

                    if (results.size() < ef || d < results.top().distance)
                    {
                        candidates.push({nid, d});
                        results.push({nid, d});

                        if (results.size() > ef)
                        {
                            results.pop();
                        }
                    }
                }
            }

            std::vector<NeighborResult> out;
            out.reserve(results.size());
            while (!results.empty())
            {
                out.push_back(results.top());
                results.pop();
            }

            return out;
        }

        /**
         * Select the closest max_conn neighbours from candidates.
         *
         * @param candidates Candidate results (any order).
         * @param max_conn   Maximum neighbours to select.
         * @return Sorted vector of the closest max_conn results.
         */
        std::vector<NeighborResult> _select_neighbours(
            const float * /* query */,
            const std::vector<NeighborResult> &candidates,
            uint32_t max_conn) const
        {
            std::vector<NeighborResult> sorted = candidates;
            std::sort(sorted.begin(), sorted.end());

            if (sorted.size() > max_conn)
            {
                sorted.resize(max_conn);
            }

            return sorted;
        }

        /**
         * Prune a node's connections at a given layer to max_conn.
         *
         * Keeps the closest max_conn neighbours.
         *
         * @param node_idx Internal node index.
         * @param layer    Layer to prune.
         * @param max_conn Maximum connections to keep.
         */
        void _prune_connections(
            int32_t node_idx,
            uint32_t layer,
            uint32_t max_conn)
        {
            if (node_idx < 0 ||
                static_cast<uint32_t>(node_idx) >= n_vectors_)
            {
                return;
            }
            if (layer > nodes_[node_idx].level)
            {
                return;
            }

            auto &nn = nodes_[node_idx].neighbours[layer];
            if (nn.size() <= max_conn)
            {
                return;
            }

            const float *node_vec = _vec(node_idx);

            std::sort(nn.begin(), nn.end(), [&](int32_t a, int32_t b)
                      { return dist_fn_(node_vec, _vec(a), dim_) <
                               dist_fn_(node_vec, _vec(b), dim_); });

            // Remove deleted neighbours
            nn.erase(
                std::remove_if(nn.begin(), nn.end(),
                    [&](int32_t nid) { return nodes_[nid].deleted; }),
                nn.end());

            nn.resize(max_conn);
        }

        // -----------------------------------------------------------------
        // Internal: search (FR-6)
        // -----------------------------------------------------------------

        /**
         * Full multi-layer search.
         *
         * 1. Navigate upper layers greedily (single closest per layer).
         * 2. Search layer 0 with beam width ef.
         *
         * @param query Pointer to dim floats.
         * @param k     Number of results desired.
         * @param ef    Beam width for layer 0 search.
         * @return Vector of NeighborResult (order not guaranteed).
         */
        std::vector<NeighborResult> _search_layer(
            const float *query,
            uint32_t k,
            uint32_t ef) const
        {
            if (entry_point_ < 0 || n_vectors_ == 0)
            {
                return {};
            }

            int32_t curr_entry = entry_point_;

            // Phase 1: Navigate upper layers (max_level_ down to 1)
            for (uint32_t l = max_level_; l > 0; --l)
            {
                curr_entry = _greedy_search_nearest(query, curr_entry, l);
            }

            // Phase 2: Search layer 0 with beam width
            uint32_t beam = std::max(ef, k);
            auto candidates = _search_layer_at_level(
                query, beam, curr_entry, 0);

            return candidates;
        }

        // -----------------------------------------------------------------
        // Internal: vector access
        // -----------------------------------------------------------------

        /**
         * Get pointer to the vector for internal node index idx.
         *
         * @param idx Internal node index.
         * @return Pointer to dim floats.
         */
        const float *_vec(int32_t idx) const
        {
            return vectors_.data() + static_cast<size_t>(idx) * dim_;
        }

        /**
         * Normalize a vector in-place to unit length.
         *
         * @param vec Pointer to dim floats.
         * @param dim Number of dimensions.
         */
        static void _normalize_vector(float *vec, uint32_t dim)
        {
            float norm = 0.0f;
            for (uint32_t i = 0; i < dim; ++i)
            {
                norm += vec[i] * vec[i];
            }
            norm = std::sqrt(norm);
            if (norm > 1e-10f)
            {
                for (uint32_t i = 0; i < dim; ++i)
                {
                    vec[i] /= norm;
                }
            }
        }
    };

    // =========================================================================
    // Simple logging for HNSW (avoids external dependencies)
    // =========================================================================
    inline void HnswIndex::logger_debug(const char *, ...) { /* no-op */ }

} // namespace isocortex

#endif // ISO_HNSW_HPP
