# Eight Schools Benchmark — WeightedSampling.jl (SMC)
#
# Hierarchical model from Gelman et al. (2003) "Bayesian Data Analysis", Sec 5.5.
# Runs two variants: resampling-only and with MH moves.
#
# Usage: julia --project benchmarks/julia_comparison/eight_schools.jl

using WeightedSampling
using Distributions
using Random
using Printf
using Statistics

# Data
J = 8
y = [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]
σ = [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]

@smc function eight_schools_no_move(J, y, σ)
    μ ~ Normal(0, 5)
    τ ~ Exponential(5)

    θ .= zeros(J)
    for j in 1:J
        θ[j] ~ Normal(μ, τ)
        y[j] => Normal(θ[j], σ[j])
    end
end

@smc function eight_schools_move(J, y, σ)
    μ ~ Normal(0, 5)
    τ ~ Exponential(5)

    θ .= zeros(J)
    for j in 1:J
        θ[j] ~ Normal(μ, τ)
        y[j] => Normal(θ[j], σ[j])
        if resampled
            μ << autoRW()
            τ << autoRW()
        end
    end
end

function run_variant(label, model_fn, n_particles_list, n_runs)
    println("\n" * "─"^60)
    println("  ", label)
    println("─"^60)

    # Warmup (compilation)
    Random.seed!(42)
    model_fn(J, y, σ; n_particles=100, show_progress=false)

    for n_particles in n_particles_list
        println("\n--- n_particles = $n_particles ---")

        times = Float64[]
        local particles, evidence
        for run in 1:n_runs
            Random.seed!(42 + run)
            t = @elapsed begin
                particles, evidence = model_fn(J, y, σ;
                    n_particles=n_particles, show_progress=false)
            end
            push!(times, t)
        end

        med_time = median(times)

        μ_mean = @E(μ -> μ, particles)
        n_unique_μ = length(unique(round.(particles.μ, digits=6)))

        @printf("Median time:  %.4f s  (range: %.4f – %.4f)\n",
            med_time, minimum(times), maximum(times))
        @printf("Log evidence: %.4f\n", evidence)
        @printf("μ mean:       %.2f\n", μ_mean)
        @printf("Unique μ:     %d / %d\n", n_unique_μ, n_particles)
    end
end

function benchmark_eight_schools()
    n_particles_list = [1_000, 5_000, 10_000]
    n_runs = 5

    println("="^60)
    println("Eight Schools Benchmark — WeightedSampling.jl (SMC)")
    println("="^60)
    @printf("Runs per config: %d (median reported)\n", n_runs)

    run_variant("Resampling only", eight_schools_no_move, n_particles_list, n_runs)
    run_variant("With MH moves", eight_schools_move, n_particles_list, n_runs)

    println("\n" * "="^60)
end

benchmark_eight_schools()
