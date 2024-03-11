using CSV
using DataFrames
using DataInterpolations
using StatsBase
using LinearAlgebra

using ModelingToolkit
using DifferentialEquations

using Optimization
using OptimizationOptimJL
import Optim
import ComponentArrays: ComponentArray

using Stheno
using Measurements
using LogExpFunctions

using CairoMakie

include("checkup.jl")
include("ecm.jl")
include("gp.jl")

include("benchmark/soh.jl")
include("benchmark/ocv.jl")
include("benchmark/rint.jl")
include("benchmark/sim.jl")

include("plot/dataset.jl")
include("plot/benchmark.jl")
include("plot/gp-ecm.jl")

# --- data
files = readdir("data/check-ups/", join=true)
data = load_data(files)

# --- fit models
ecms = fit_ecm_series(data)
gpms = fit_gp_series(data)

# --- benchmark
df_soh = benchmark_soh(ecms, gpms, data)
df_ocv = benchmark_ocv(ecms, gpms, data)
df_rint = benchmark_rint(ecms, gpms, data)
df_sim = benchmark_simulation(ecms, gpms, data)

# --- plot results
fig1 = plot_checkup_profile(data[:LGL13818]) # fresh cell
fig2 = plot_ocvs(data)
fig3 = plot_rints(data)

id = :MF014
fig4 = plot_gp_ecm(data[id], gpms[id])

ids = (:LGL13818, :MF240, :MF001, :MF199)
fig5 = plot_ocv_fit(ecms, gpms, data, ids)

fig6 = plot_soh_fit(df_soh)
fig7 = plot_rint_fit(df_rint)

fig8 = plot_sim(ecms, gpms, data)
