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
using ColorSchemes

include("checkup.jl")
include("ecm.jl")
include("gp.jl")

include("benchmark/soh.jl")
include("benchmark/ocv.jl")
include("benchmark/rint.jl")
include("benchmark/sim.jl")

include("plot/dataset.jl")
include("plot/benchmark.jl")

# --- data
files = readdir("data/", join=true)
data = load_data(files)

# --- fit models
ecms = fit_ecm_series(data)
gpms = fit_gpm_series(data)

# --- benchmark
df_cu = summarize_checkups(data)

df_sim = benchmark_sim(ecms, gpms, data)
df_soh = benchmark_soh(ecms, gpms, data)
df_ocv = benchmark_ocv(ecms, gpms, data)
r2_rdc = benchmark_rdc(ecms, gpms, data)

# --- plot results
fig1 = plot_checkup_profile(data[:Cell1]) # fresh cell

fig2 = plot_checkups(data)

fig3 = plot_sim(ecms, gpms, data)

fig4 = plot_ocv_fit(ecms, gpms, data)

fig5 = plot_gp_rint(gpms, data)

fig6 = plot_rint_fit(ecms, gpms, data)
