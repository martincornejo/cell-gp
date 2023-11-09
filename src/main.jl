using CSV
using DataFrames
using DataInterpolations

using ModelingToolkit
using DifferentialEquations

# using Stheno

using Optimization
using OptimizationOptimJL
import ComponentArrays: ComponentArray

# using GLMakie
using CairoMakie
CairoMakie.activate!(type="svg")

include("checkup.jl")
include("ecm.jl")


#
function plot_ocvs(files)
    s = 0:0.001:1.0
    ylabel = "Voltage in V"
    xlabel = "SOC in p.u."
    colormap = :dense
    colorrange = (0.6, 1.0)

    fig1 = Figure(resolution=(800, 400))
    ax1 = Axis(fig1[1, 1]; xlabel, ylabel, title="Charge pOCV")
    ax2 = Axis(fig1[1, 2]; xlabel, title="Discharge pOCV")
    linkyaxes!(ax1, ax2)
    hideydecorations!(ax2, ticks=false, grid=false)

    fig2 = Figure(resolution=(900, 400))
    ax3 = Axis(fig2[1, 1]; xlabel, ylabel, title="Mean pOCV")
    ax4 = Axis(fig2[1, 2]; xlabel, title="OCV Hystheresis")

    for file in files
        df = read_basytec(file)

        # c-cov
        ocv_c, cap_c = calc_cocv(df)
        c = s .* cap_c
        lines!(ax1, s, ocv_c(c); color=cap_c / 4.9, colorrange, colormap)

        # d-ocv
        ocv_d, cap_d = calc_docv(df)
        c = s .* cap_d
        lines!(ax2, s, ocv_d(c); color=cap_d / 4.9, colorrange, colormap)

        # ocv
        pocv, cap = calc_pocv(df)
        c = s .* cap
        lines!(ax3, s, pocv(c); color=cap / 4.9, colorrange, colormap)

        # hyst
        lines!(ax4, s, pocv(c) - ocv_c(c); color=cap / 4.9, colorrange, colormap)
    end

    Colorbar(fig1[:, 3]; colorrange, colormap, label="SOH")
    Colorbar(fig2[:, 3]; colorrange, colormap, label="SOH")
    return fig1, fig2
end


# ------
function main(files)
    tt = 0:60:(24*3600.0)
    focv = fresh_focv()
    res = Dict()
    for file in files
        id = get_cell_id(file)
        @info id
        df = read_basytec(file)

        cu = calc_capa_cccv(df) # check-up
        ecm, ode = fit_ecm(df, tt, focv) # model
        # ode = fit_ecm(df) # model

        res[Symbol(id)] = Dict(:cu => cu, :model => ecm, :ode => ode)
    end
    return res
end

function plot_ecm_series(res)
    tspan = (0.0, 3 * 24 * 3600.0)

    fig = Figure(resolution=(1200, 600))
    ax = [Axis(fig[i, 1]) for i in 1:3]
    ax[1].title = "Measured"
    ax[2].title = "Simulated"
    ax[3].title = "Error"

    linkxaxes!(ax...)

    xlims!(ax[1], tspan)
    xlims!(ax[2], tspan)
    xlims!(ax[3], tspan)
    ax[1].ylabel = "Voltage (V)"
    ax[2].ylabel = "Voltage (V)"
    ax[3].ylabel = "Voltage (V)"
    ax[3].xlabel = "Time (s)"

    hidexdecorations!(ax[1])
    hidexdecorations!(ax[2])

    colormap = :dense
    colorrange = (0.6, 1.0)

    for file in files
        id = get_cell_id(file)

        df_cell = read_basytec(file)
        data = load_profile(df_cell)

        cap = prob = res[Symbol(id)][:cu]
        soh = cap / 4.9

        prob = res[Symbol(id)][:ode]
        ecm = res[Symbol(id)][:model]
        new_prob = remake(prob; tspan)
        sol = solve(new_prob)

        δv = sol[ecm.v] - data.v(sol.t)

        lines!(ax[1], sol.t, data.v(sol.t); label=id, color=soh, colorrange, colormap)
        lines!(ax[2], sol.t, sol[ecm.v]; label=id, color=soh, colorrange, colormap)
        lines!(ax[3], sol.t, δv; label=id, color=soh, colorrange, colormap)

        l2 = sum(abs2, δv)

        @info id soh l2
    end

    Colorbar(fig[:, 2]; colorrange, colormap, label="SOH")

    fig
end

files = readdir("data/check-ups/", join=true)

# ocvs
fig1, fig2 = plot_ocvs(files)
fig1 |> save("figs/ocv1.svg")
fig2 |> save("figs/ocv2.svg")

# ECM
res = main(files)

for key in keys(res)
    estimated = res[key][:ode].p[1] / 4.9
    measured = res[key][:cu] / 4.9
    e = (estimated - measured) * 100
    @info key estimated measured e
end

fig = plot_ecm_series(res)

save("figs/ecm2.png", fig, px_per_unit=2)


save("figs/ecm.svg", fig)