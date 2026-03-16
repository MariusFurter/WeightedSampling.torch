# State-Space Model Benchmark — WeightedSampling.jl (SMC)
#
# Linear-Gaussian SSM:
#   x_0 ~ N(0, 1)
#   x_t = 0.9 * x_{t-1} + ε_t,  ε_t ~ N(0, 1)
#   y_t ~ N(x_t, 1)
#
# Usage: julia --project benchmarks/julia_comparison/ssm.jl

using WeightedSampling
using Distributions
using Random
using Printf
using Statistics

const A = 0.9   # AR coefficient
const Q = 1.0   # process noise std
const R = 1.0   # observation noise std

function generate_data(T)
    Random.seed!(42)
    xs = Float64[0.0]
    ys = Float64[]
    for t in 1:T
        x_new = A * xs[end] + Q * randn()
        push!(xs, x_new)
        push!(ys, x_new + R * randn())
    end
    return xs, ys
end

# Kalman filter for ground truth
function kalman_filter(ys)
    T = length(ys)
    μ = 0.0
    σ² = 1.0
    log_evidence = 0.0
    filtered_means = Float64[]

    for t in 1:T
        μ_pred = A * μ
        σ²_pred = A^2 * σ² + Q^2

        S = σ²_pred + R^2
        K = σ²_pred / S
        innov = ys[t] - μ_pred
        μ = μ_pred + K * innov
        σ² = (1 - K) * σ²_pred

        log_evidence += -0.5 * (log(2π * S) + innov^2 / S)
        push!(filtered_means, μ)
    end

    return filtered_means, log_evidence
end

@smc function ssm_filter(data, a, q, r)
    x ~ Normal(0.0, 1.0)
    for (t, y) in enumerate(data)
        x ~ Normal(a * x, q)
        y => Normal(x, r)
    end
end

function benchmark_ssm()
    T = 200
    n_particles_list = [1_000, 5_000, 10_000]
    n_runs = 5

    _, ys = generate_data(T)
    kf_means, kf_evidence = kalman_filter(ys)

    println("="^60)
    println("SSM Benchmark — WeightedSampling.jl (Bootstrap PF)")
    println("="^60)
    @printf("Timesteps: %d\n", T)
    @printf("Kalman log evidence: %.4f\n", kf_evidence)
    @printf("Runs per config: %d (median reported)\n", n_runs)

    # Warmup (compilation)
    Random.seed!(0)
    ssm_filter(ys, A, Q, R; n_particles=100, show_progress=false)

    println("\n" * "─"^60)
    println("  Resampling only")
    println("─"^60)

    for n_particles in n_particles_list
        println("\n--- n_particles = $n_particles ---")

        times = Float64[]
        local particles, evidence
        for run in 1:n_runs
            Random.seed!(42 + run)
            t = @elapsed begin
                particles, evidence = ssm_filter(ys, A, Q, R;
                    n_particles=n_particles, show_progress=false)
            end
            push!(times, t)
        end

        med_time = median(times)
        x_T_est = @E(x -> x, particles)

        @printf("Median time:      %.4f s  (range: %.4f – %.4f)\n",
            med_time, minimum(times), maximum(times))
        @printf("Steps/sec:        %.0f\n", T / med_time)
        @printf("Log evidence:     %.4f  (Kalman: %.4f, diff: %.4f)\n",
            evidence, kf_evidence, abs(evidence - kf_evidence))
        @printf("E[x_T]:           %.4f  (Kalman: %.4f)\n",
            x_T_est, kf_means[end])
    end

    println("\n" * "="^60)
end

benchmark_ssm()
