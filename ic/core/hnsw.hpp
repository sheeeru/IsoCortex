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
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

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
            if (space != "cosine" && space != "l2")
            {
                throw std::invalid_argument(
                    "HnswConfig: space must be 'cosine' or 'l2', got '" +
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

    /**
     * Compute cosine distance between two float32 vectors.
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
        float dot = 0.0f;
        float norm_a = 0.0f;
        float norm_b = 0.0f;

        for (uint32_t i = 0; i < dim; ++i)
        {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }

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
     * Compute squared L2 (Euclidean) distance between two float32 vectors.
     *
     * @param a   Pointer to first vector (dim floats).
     * @param b   Pointer to second vector (dim floats).
     * @param dim Number of dimensions.
     * @return Squared L2 distance.
     */
    inline float l2_distance(const float *a, const float *b, uint32_t dim)
    {
        float sum = 0.0f;
        for (uint32_t i = 0; i < dim; ++i)
        {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum;
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

} // namespace isocortex

#endif // ISO_HNSW_HPP
