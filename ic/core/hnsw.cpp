/*
 * IsoCortex — core/hnsw.cpp
 * ==========================
 * Standalone test and benchmark for the HNSW index.
 *
 * Usage:
 *   g++ -std=c++17 -O2 -o test_hnsw core/hnsw.cpp
 *   ./test_hnsw
 *
 * Author : Shaheer Qureshi
 * Project: IsoCortex
 */

#include "hnsw.hpp"

#include <chrono>
#include <cstdio>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

using namespace isocortex;

static int g_tests_run = 0;
static int g_tests_passed = 0;
static int g_tests_failed = 0;

// =========================================================================
// Macros
// =========================================================================

#define RUN_TEST(name)                            \
    do                                            \
    {                                             \
        g_tests_run++;                            \
        printf("  TEST %-50s", #name);            \
        try                                       \
        {                                         \
            test_##name();                        \
            g_tests_passed++;                     \
            printf("[PASS]\n");                   \
        }                                         \
        catch (const std::exception &e)           \
        {                                         \
            g_tests_failed++;                     \
            printf("[FAIL] %s\n", e.what());      \
        }                                         \
        catch (...)                               \
        {                                         \
            g_tests_failed++;                     \
            printf("[FAIL] unknown exception\n"); \
        }                                         \
    } while (0)

#define ASSERT_TRUE(cond)                                \
    do                                                   \
    {                                                    \
        if (!(cond))                                     \
        {                                                \
            throw std::runtime_error(                    \
                std::string("FAIL: ") + #cond +          \
                " at line " + std::to_string(__LINE__)); \
        }                                                \
    } while (0)

#define ASSERT_EQ(a, b)                                    \
    do                                                     \
    {                                                      \
        if ((a) != (b))                                    \
        {                                                  \
            throw std::runtime_error(                      \
                std::string("FAIL: ") + #a + " != " + #b + \
                " (" + std::to_string(a) +                 \
                " vs " + std::to_string(b) +               \
                ") at line " + std::to_string(__LINE__));  \
        }                                                  \
    } while (0)

#define ASSERT_THROWS(expr, ex_type)                                              \
    do                                                                            \
    {                                                                             \
        bool caught = false;                                                      \
        try                                                                       \
        {                                                                         \
            expr;                                                                 \
        }                                                                         \
        catch (const ex_type &)                                                   \
        {                                                                         \
            caught = true;                                                        \
        }                                                                         \
        catch (...)                                                               \
        {                                                                         \
            throw std::runtime_error(                                             \
                std::string("FAIL: wrong exception for ") + #expr + " at line " + \
                std::to_string(__LINE__));                                        \
        }                                                                         \
        if (!caught)                                                              \
        {                                                                         \
            throw std::runtime_error(                                             \
                std::string("FAIL: no exception for ") + #expr + " at line " +    \
                std::to_string(__LINE__));                                        \
        }                                                                         \
    } while (0)

// =========================================================================
// Helpers
// =========================================================================

static std::vector<float> random_vectors(
    uint32_t n, uint32_t dim, uint32_t seed = 42)
{
    if (n == 0 || dim == 0)
    {
        return {};
    }

    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> v(static_cast<size_t>(n) * dim);
    for (auto &x : v)
    {
        x = dist(rng);
    }

    for (uint32_t i = 0; i < n; ++i)
    {
        float norm = 0.0f;
        for (uint32_t j = 0; j < dim; ++j)
        {
            norm += v[i * dim + j] * v[i * dim + j];
        }
        norm = std::sqrt(norm);
        if (norm > 1e-10f)
        {
            for (uint32_t j = 0; j < dim; ++j)
            {
                v[i * dim + j] /= norm;
            }
        }
    }

    return v;
}

static std::vector<float> random_vectors_raw(
    uint32_t n, uint32_t dim, uint32_t seed = 42)
{
    if (n == 0 || dim == 0)
    {
        return {};
    }

    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> v(static_cast<size_t>(n) * dim);
    for (auto &x : v)
    {
        x = dist(rng);
    }

    return v;
}

static std::vector<NeighborResult> brute_force_search(
    const float *query,
    const float *vectors,
    uint32_t n,
    uint32_t dim,
    uint32_t k,
    std::function<float(const float *, const float *, uint32_t)> dist_fn)
{
    std::vector<NeighborResult> all;
    all.reserve(n);

    for (uint32_t i = 0; i < n; ++i)
    {
        float d = dist_fn(
            query,
            vectors + static_cast<size_t>(i) * dim,
            dim);
        all.push_back({static_cast<int32_t>(i), d});
    }

    std::sort(all.begin(), all.end());

    if (all.size() > k)
    {
        all.resize(k);
    }

    return all;
}

static float compute_recall(
    const std::vector<NeighborResult> &hnsw_results,
    const std::vector<NeighborResult> &gt_results,
    uint32_t k)
{
    if (gt_results.empty())
    {
        return 1.0f;
    }

    std::unordered_set<int32_t> gt_ids;
    gt_ids.reserve(gt_results.size());
    for (const auto &r : gt_results)
    {
        gt_ids.insert(r.id);
    }

    uint32_t hits = 0;
    uint32_t check = std::min(
        k, static_cast<uint32_t>(hnsw_results.size()));

    for (uint32_t i = 0; i < check; ++i)
    {
        if (gt_ids.count(hnsw_results[i].id))
        {
            hits++;
        }
    }

    uint32_t denom = std::min(
        k, static_cast<uint32_t>(gt_results.size()));

    if (denom == 0)
    {
        return 1.0f;
    }

    return static_cast<float>(hits) / static_cast<float>(denom);
}

// =========================================================================
// Test functions
// =========================================================================

static void test_config_validation()
{
    HnswConfig cfg;
    cfg.validate();

    HnswConfig bad1;
    bad1.M = 0;
    ASSERT_THROWS(bad1.validate(), std::invalid_argument);

    HnswConfig bad2;
    bad2.ef_construction = 0;
    ASSERT_THROWS(bad2.validate(), std::invalid_argument);

    HnswConfig bad3;
    bad3.ef_search = 0;
    ASSERT_THROWS(bad3.validate(), std::invalid_argument);

    HnswConfig bad4;
    bad4.dim = 128;
    ASSERT_THROWS(bad4.validate(), std::invalid_argument);

    HnswConfig bad5;
    bad5.space = "euclidean";
    ASSERT_THROWS(bad5.validate(), std::invalid_argument);

    HnswConfig cfg2;
    cfg2.M = 8;
    cfg2.resolve();
    ASSERT_EQ(cfg2.M_max0, 16u);
}

static void test_default_constructor()
{
    HnswIndex idx;
    ASSERT_EQ(idx.size(), 0u);
    ASSERT_EQ(idx.dim(), kVectorDim);
    ASSERT_TRUE(!idx.is_built());
    ASSERT_EQ(idx.max_level(), 0u);
    ASSERT_EQ(idx.entry_point(), -1);
}

static void test_build_basic()
{
    const uint32_t N = 100;
    const uint32_t DIM = kVectorDim;
    const uint32_t K = 5;

    auto vectors = random_vectors(N, DIM);

    HnswConfig cfg;
    cfg.M = 16;
    cfg.ef_construction = 200;
    cfg.ef_search = 50;
    cfg.space = "cosine";
    cfg.resolve();

    HnswIndex idx(cfg);
    idx.build(vectors.data(), N, DIM);

    ASSERT_EQ(idx.size(), N);
    ASSERT_TRUE(idx.is_built());
    ASSERT_TRUE(idx.total_edges() > 0);

    auto results = idx.search(vectors.data(), K);
    ASSERT_EQ(results.size(), K);
    ASSERT_EQ(results[0].id, 0);
    ASSERT_TRUE(results[0].distance < 0.01f);
}

static void test_build_with_external_ids()
{
    const uint32_t N = 50;
    const uint32_t DIM = kVectorDim;

    auto vectors = random_vectors(N, DIM);

    std::vector<int32_t> ext_ids(N);
    for (uint32_t i = 0; i < N; ++i)
    {
        ext_ids[i] = static_cast<int32_t>(i) + 1000;
    }

    HnswConfig cfg;
    cfg.resolve();

    HnswIndex idx(cfg);
    idx.build(vectors.data(), N, DIM, ext_ids.data());

    auto results = idx.search(vectors.data(), 1);
    ASSERT_EQ(results.size(), 1u);
    ASSERT_EQ(results[0].id, 0);
    ASSERT_EQ(idx.get_external_id(0), 1000);
}

static void test_search_recall()
{
    const uint32_t N = 500;
    const uint32_t DIM = kVectorDim;
    const uint32_t K = 10;
    const uint32_t Q = 20;

    auto vectors = random_vectors(N, DIM, 123);
    auto queries = random_vectors(Q, DIM, 456);

    HnswConfig cfg;
    cfg.M = 16;
    cfg.ef_construction = 200;
    cfg.ef_search = 100;
    cfg.space = "cosine";
    cfg.resolve();

    HnswIndex idx(cfg);
    idx.build(vectors.data(), N, DIM);

    float total_recall = 0.0f;

    for (uint32_t q = 0; q < Q; ++q)
    {
        const float *query =
            queries.data() + static_cast<size_t>(q) * DIM;

        auto hnsw_results = idx.search(query, K);
        auto gt_results = brute_force_search(
            query, vectors.data(), N, DIM, K, cosine_distance);

        total_recall += compute_recall(hnsw_results, gt_results, K);
    }

    float avg_recall = total_recall / static_cast<float>(Q);
    ASSERT_TRUE(avg_recall > 0.90f);
}

static void test_search_l2()
{
    const uint32_t N = 100;
    const uint32_t DIM = kVectorDim;
    const uint32_t K = 5;

    auto vectors = random_vectors_raw(N, DIM, 789);

    HnswConfig cfg;
    cfg.space = "l2";
    cfg.normalize = false;
    cfg.ef_search = 50;
    cfg.resolve();

    HnswIndex idx(cfg);
    idx.build(vectors.data(), N, DIM);

    auto results = idx.search(vectors.data(), K);
    ASSERT_EQ(results.size(), K);
    ASSERT_EQ(results[0].id, 0);
    ASSERT_TRUE(results[0].distance < 0.01f);
}

static void test_insert_incremental()
{
    const uint32_t N = 50;
    const uint32_t DIM = kVectorDim;

    auto vectors = random_vectors(N, DIM, 999);

    HnswConfig cfg;
    cfg.ef_construction = 200;
    cfg.ef_search = 50;
    cfg.resolve();

    HnswIndex idx(cfg);

    for (uint32_t i = 0; i < N; ++i)
    {
        idx.insert(
            vectors.data() + static_cast<size_t>(i) * DIM,
            static_cast<int32_t>(i));
    }

    ASSERT_EQ(idx.size(), N);
    ASSERT_TRUE(idx.is_built());

    auto results = idx.search(vectors.data(), 5);
    ASSERT_EQ(results.size(), 5u);
    ASSERT_EQ(results[0].id, 0);
}

static void test_single_vector()
{
    const uint32_t DIM = kVectorDim;

    auto vectors = random_vectors(1, DIM, 111);

    HnswConfig cfg;
    cfg.resolve();

    HnswIndex idx(cfg);
    idx.build(vectors.data(), 1, DIM);

    ASSERT_EQ(idx.size(), 1u);

    auto results = idx.search(vectors.data(), 1);
    ASSERT_EQ(results.size(), 1u);
    ASSERT_EQ(results[0].id, 0);
    ASSERT_TRUE(results[0].distance < 0.01f);
}

static void test_identical_vectors()
{
    const uint32_t N = 20;
    const uint32_t DIM = kVectorDim;

    std::vector<float> vectors(static_cast<size_t>(N) * DIM, 1.0f);

    float norm = std::sqrt(static_cast<float>(DIM));
    for (auto &x : vectors)
    {
        x /= norm;
    }

    HnswConfig cfg;
    cfg.ef_search = 50;
    cfg.resolve();

    HnswIndex idx(cfg);
    idx.build(vectors.data(), N, DIM);

    ASSERT_EQ(idx.size(), N);

    auto results = idx.search(vectors.data(), 5);
    ASSERT_EQ(results.size(), 5u);

    for (const auto &r : results)
    {
        ASSERT_TRUE(r.distance < 0.01f);
    }
}

static void test_zero_vector()
{
    const uint32_t DIM = kVectorDim;

    std::vector<float> zero(DIM, 0.0f);

    HnswConfig cfg;
    cfg.resolve();

    HnswIndex idx(cfg);
    idx.build(zero.data(), 1, DIM);

    auto results = idx.search(zero.data(), 1);
    ASSERT_EQ(results.size(), 1u);
    ASSERT_EQ(results[0].id, 0);
}

static void test_search_k_larger_than_index()
{
    const uint32_t N = 5;
    const uint32_t DIM = kVectorDim;

    auto vectors = random_vectors(N, DIM, 222);

    HnswConfig cfg;
    cfg.resolve();

    HnswIndex idx(cfg);
    idx.build(vectors.data(), N, DIM);

    auto results = idx.search(vectors.data(), 100);
    ASSERT_EQ(results.size(), N);
}

static void test_search_empty_index()
{
    HnswIndex idx;
    ASSERT_THROWS(
        idx.search(nullptr, 5),
        std::invalid_argument);
}

static void test_search_k_zero()
{
    auto vectors = random_vectors(10, kVectorDim, 333);

    HnswConfig cfg;
    cfg.resolve();

    HnswIndex idx(cfg);
    idx.build(vectors.data(), 10, kVectorDim);

    ASSERT_THROWS(
        idx.search(vectors.data(), 0),
        std::invalid_argument);
}

static void test_build_n_zero()
{
    HnswConfig cfg;
    cfg.resolve();

    HnswIndex idx(cfg);
    ASSERT_THROWS(
        idx.build(nullptr, 0, kVectorDim),
        std::invalid_argument);
}

static void test_build_dim_mismatch()
{
    auto vectors = random_vectors(10, 128, 444);

    HnswConfig cfg;
    cfg.resolve();

    HnswIndex idx(cfg);
    ASSERT_THROWS(
        idx.build(vectors.data(), 10, 128),
        std::invalid_argument);
}

static void test_build_nan_inf()
{
    const uint32_t N = 10;
    const uint32_t DIM = kVectorDim;

    auto vectors = random_vectors(N, DIM, 555);

    vectors[5] = std::numeric_limits<float>::quiet_NaN();

    HnswConfig cfg;
    cfg.resolve();

    HnswIndex idx(cfg);
    ASSERT_THROWS(
        idx.build(vectors.data(), N, DIM),
        std::runtime_error);

    ASSERT_EQ(idx.size(), 0u);
}

static void test_build_inf()
{
    const uint32_t N = 10;
    const uint32_t DIM = kVectorDim;

    auto vectors = random_vectors(N, DIM, 666);

    vectors[3 * DIM + 10] = std::numeric_limits<float>::infinity();

    HnswConfig cfg;
    cfg.resolve();

    HnswIndex idx(cfg);
    ASSERT_THROWS(
        idx.build(vectors.data(), N, DIM),
        std::runtime_error);
}

static void test_insert_nan()
{
    const uint32_t DIM = kVectorDim;

    std::vector<float> bad(DIM, std::numeric_limits<float>::quiet_NaN());

    HnswConfig cfg;
    cfg.resolve();

    HnswIndex idx(cfg);
    ASSERT_THROWS(
        idx.insert(bad.data(), 0),
        std::runtime_error);
}

static void test_move_constructor()
{
    const uint32_t N = 50;
    const uint32_t DIM = kVectorDim;

    auto vectors = random_vectors(N, DIM, 777);

    HnswConfig cfg;
    cfg.resolve();

    HnswIndex idx1(cfg);
    idx1.build(vectors.data(), N, DIM);

    uint32_t orig_size = idx1.size();

    HnswIndex idx2(std::move(idx1));

    ASSERT_EQ(idx2.size(), orig_size);
    ASSERT_TRUE(idx2.is_built());
    ASSERT_EQ(idx1.size(), 0u);
    ASSERT_TRUE(!idx1.is_built());

    auto results = idx2.search(vectors.data(), 3);
    ASSERT_EQ(results.size(), 3u);
}

static void test_move_assignment()
{
    const uint32_t N = 30;
    const uint32_t DIM = kVectorDim;

    auto vectors = random_vectors(N, DIM, 888);

    HnswConfig cfg;
    cfg.resolve();

    HnswIndex idx1(cfg);
    idx1.build(vectors.data(), N, DIM);

    HnswIndex idx2(cfg);
    idx2 = std::move(idx1);

    ASSERT_EQ(idx2.size(), N);
    ASSERT_TRUE(idx2.is_built());
    ASSERT_EQ(idx1.size(), 0u);
}

static void test_clear()
{
    const uint32_t N = 20;
    const uint32_t DIM = kVectorDim;

    auto vectors = random_vectors(N, DIM, 101);

    HnswConfig cfg;
    cfg.resolve();

    HnswIndex idx(cfg);
    idx.build(vectors.data(), N, DIM);

    ASSERT_TRUE(idx.is_built());

    idx.clear();

    ASSERT_EQ(idx.size(), 0u);
    ASSERT_TRUE(!idx.is_built());
    ASSERT_EQ(idx.max_level(), 0u);
    ASSERT_EQ(idx.entry_point(), -1);
}

static void test_accessor_bounds()
{
    const uint32_t N = 10;
    const uint32_t DIM = kVectorDim;

    auto vectors = random_vectors(N, DIM, 202);

    HnswConfig cfg;
    cfg.resolve();

    HnswIndex idx(cfg);
    idx.build(vectors.data(), N, DIM);

    ASSERT_TRUE(idx.get_vector(0) != nullptr);
    ASSERT_EQ(idx.get_external_id(0), 0);

    ASSERT_TRUE(idx.get_vector(999) == nullptr);
    ASSERT_EQ(idx.get_external_id(999), -1);

    ASSERT_TRUE(idx.get_vector(N - 1) != nullptr);
    ASSERT_EQ(idx.get_external_id(N - 1),
              static_cast<int32_t>(N - 1));
}

static void test_recall_different_k()
{
    const uint32_t N = 300;
    const uint32_t DIM = kVectorDim;
    const uint32_t Q = 10;

    auto vectors = random_vectors(N, DIM, 303);
    auto queries = random_vectors(Q, DIM, 404);

    HnswConfig cfg;
    cfg.M = 16;
    cfg.ef_construction = 200;
    cfg.ef_search = 100;
    cfg.space = "cosine";
    cfg.resolve();

    HnswIndex idx(cfg);
    idx.build(vectors.data(), N, DIM);

    uint32_t k_values[] = {1, 5, 10, 50};

    for (uint32_t K : k_values)
    {
        float total_recall = 0.0f;

        for (uint32_t q = 0; q < Q; ++q)
        {
            const float *query =
                queries.data() + static_cast<size_t>(q) * DIM;

            auto hnsw_results = idx.search(query, K);
            auto gt_results = brute_force_search(
                query, vectors.data(), N, DIM, K, cosine_distance);

            total_recall += compute_recall(
                hnsw_results, gt_results, K);
        }

        float avg_recall = total_recall / static_cast<float>(Q);
        ASSERT_TRUE(avg_recall > 0.85f);
    }
}

static void test_benchmark()
{
    const uint32_t N = 10000;
    const uint32_t DIM = kVectorDim;
    const uint32_t Q = 100;
    const uint32_t K = 10;

    printf("\n  BENCHMARK: n=%u, dim=%u, k=%u, queries=%u\n",
           N, DIM, K, Q);

    auto vectors = random_vectors(N, DIM, 505);
    auto queries = random_vectors(Q, DIM, 606);

    HnswConfig cfg;
    cfg.M = 16;
    cfg.ef_construction = 200;
    cfg.ef_search = 100;
    cfg.space = "cosine";
    cfg.resolve();

    HnswIndex idx(cfg);

    auto t0 = std::chrono::high_resolution_clock::now();
    idx.build(vectors.data(), N, DIM);
    auto t1 = std::chrono::high_resolution_clock::now();

    double build_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();

    printf("    Build time     : %.1f ms\n", build_ms);
    printf("    Vectors/sec    : %.0f\n",
           N * 1000.0 / std::max(build_ms, 0.001));
    printf("    Total edges    : %llu\n",
           static_cast<unsigned long long>(idx.total_edges()));
    printf("    Max level      : %u\n", idx.max_level());

    t0 = std::chrono::high_resolution_clock::now();
    for (uint32_t q = 0; q < Q; ++q)
    {
        const float *query =
            queries.data() + static_cast<size_t>(q) * DIM;
        auto results = idx.search(query, K);
        (void)results;
    }
    t1 = std::chrono::high_resolution_clock::now();

    double search_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    double safe_search = std::max(search_ms, 0.001);
    double qps = static_cast<double>(Q) * 1000.0 / safe_search;

    printf("    Search time    : %.1f ms total (%.3f ms/query)\n",
           search_ms, search_ms / static_cast<double>(Q));
    printf("    QPS            : %.0f\n", qps);

    float total_recall = 0.0f;
    for (uint32_t q = 0; q < Q; ++q)
    {
        const float *query =
            queries.data() + static_cast<size_t>(q) * DIM;

        auto hnsw_results = idx.search(query, K);
        auto gt_results = brute_force_search(
            query, vectors.data(), N, DIM, K, cosine_distance);

        total_recall += compute_recall(
            hnsw_results, gt_results, K);
    }

    float avg_recall = total_recall / static_cast<float>(Q);
    printf("    Recall@%u      : %.1f%%\n",
           K, avg_recall * 100.0f);

    t0 = std::chrono::high_resolution_clock::now();
    for (uint32_t q = 0; q < Q; ++q)
    {
        const float *query =
            queries.data() + static_cast<size_t>(q) * DIM;
        auto results = brute_force_search(
            query, vectors.data(), N, DIM, K, cosine_distance);
        (void)results;
    }
    t1 = std::chrono::high_resolution_clock::now();

    double bf_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    double speedup = bf_ms / safe_search;

    printf("    Brute-force    : %.1f ms total\n", bf_ms);
    printf("    Speedup        : %.1fx\n", speedup);
}

// =========================================================================
// Main
// =========================================================================

int main()
{
    printf("IsoCortex HNSW Test Suite\n");
    printf("=========================\n\n");

    printf("[Configuration]\n");
    RUN_TEST(config_validation);
    RUN_TEST(default_constructor);

    printf("\n[Build]\n");
    RUN_TEST(build_basic);
    RUN_TEST(build_with_external_ids);
    RUN_TEST(build_n_zero);
    RUN_TEST(build_dim_mismatch);
    RUN_TEST(build_nan_inf);
    RUN_TEST(build_inf);

    printf("\n[Search]\n");
    RUN_TEST(search_recall);
    RUN_TEST(search_l2);
    RUN_TEST(search_k_larger_than_index);
    RUN_TEST(search_empty_index);
    RUN_TEST(search_k_zero);
    RUN_TEST(recall_different_k);

    printf("\n[Insert]\n");
    RUN_TEST(insert_incremental);
    RUN_TEST(insert_nan);

    printf("\n[Edge Cases]\n");
    RUN_TEST(single_vector);
    RUN_TEST(identical_vectors);
    RUN_TEST(zero_vector);

    printf("\n[Lifecycle]\n");
    RUN_TEST(move_constructor);
    RUN_TEST(move_assignment);
    RUN_TEST(clear);
    RUN_TEST(accessor_bounds);

    printf("\n[Benchmark]\n");
    test_benchmark();

    printf("\n=========================\n");
    printf("Results: %d/%d passed, %d failed\n",
           g_tests_passed, g_tests_run, g_tests_failed);

    return g_tests_failed > 0 ? 1 : 0;
}
