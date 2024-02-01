using CSV
using DataFrames
using DataInterpolations
using StatsBase

using ModelingToolkit
using DifferentialEquations

using Optimization
using OptimizationOptimJL
import ComponentArrays: ComponentArray

using Stheno
using ParameterHandling
using Zygote

# using GLMakie
using CairoMakie
# CairoMakie.activate!(type="svg")

include("checkup.jl")
include("ecm.jl")
include("gp.jl")

# ------
files = readdir("data/check-ups/", join=true)
data = load_data(files)

fig = plot_checkup_profile(data[:LGL13818]) # fresh cell
save("figs/measurement-plan.pdf", fig, pt_per_unit=1)

fig = plot_ocvs(data)
save("figs/pOCV.pdf", fig, pt_per_unit=1)

fig = plot_rints(data; timestep=9.99)
save("figs/rint.pdf", fig, pt_per_unit=1)

# ------
function fit_ecm_series(data)
    tt = 0:60:(24*3600.0)
    focv = fresh_focv()
    res = Dict()
    for (id, df) in data
        cu = calc_capa_cccv(df) # check-up
        ecm, ode = fit_ecm(df, tt, focv) # model
        res[id] = Dict(:cu => cu, :model => ecm, :ode => ode)
    end
    return res
end

function plot_ecm_series(res, data)
    tspan = (0.0, 3 * 24 * 3600.0)

    fig = Figure(size=(1200, 600))
    ax = [Axis(fig[i, 1]) for i in 1:3]
    ax[1].title = "Measured"
    ax[2].title = "Simulated"
    ax[3].title = "Error"

    linkxaxes!(ax...)

    xlims!(ax[1], tspan)
    xlims!(ax[2], tspan)
    xlims!(ax[3], tspan)
    ylims!(ax[1], (3.3, 4.1))
    ylims!(ax[2], (3.3, 4.1))
    ylims!(ax[3], (-0.15, 0.15))
    ax[1].ylabel = "Voltage (V)"
    ax[2].ylabel = "Voltage (V)"
    ax[3].ylabel = "Voltage (V)"
    ax[3].xlabel = "Time (s)"

    hidexdecorations!(ax[1])
    hidexdecorations!(ax[2])

    colormap = :dense
    colorrange = (0.6, 1.0)

    for (id, df_cell) in data
        profile = load_profile(df_cell)

        cap = res[id][:cu]
        soh = cap / 4.9

        prob = res[id][:ode]
        ecm = res[id][:model]
        new_prob = remake(prob; tspan)
        sol = solve(new_prob; saveat=60)

        v = sol[ecm.v]
        v̄ = profile.v(sol.t)
        δv = v - v̄

        lines!(ax[1], sol.t, v̄; label=id, color=soh, colorrange, colormap)
        lines!(ax[2], sol.t, v; label=id, color=soh, colorrange, colormap)
        lines!(ax[3], sol.t, δv; label=id, color=soh, colorrange, colormap)

        l2 = sum(abs2, δv)

        @info id soh l2
    end

    Colorbar(fig[:, 2]; colorrange, colormap, label="SOH")

    fig
end


# ECM
res = fit_ecm_series(data)

for key in keys(res)
    estimated = res[key][:ode].p[1] / 4.9
    measured = res[key][:cu] / 4.9
    e = (estimated - measured) * 100
    @info key estimated measured e
end

fig = plot_ecm_series(res, data)
fig |> save("figs/ecm.pdf")


### GP
"SOC dependent R"
function model(θ)
    i = x -> x[2]
    return @gppp let
        ocv = θ.ocv.σ * GP(with_lengthscale(SEKernel(), θ.ocv.l) ∘ SelectTransform(1))
        r = θ.r.σ * GP(with_lengthscale(SEKernel(), θ.r.l) ∘ SelectTransform(1))
        vr = r * i
        v = ocv + vr
    end
end

function fit_gp_series(data)
    tt = 0:60:(24*3600.0)

    # # GP hyperparams
    dt = fit_zscore("") # TODO fix this
    θ = (
        kernel=(
            ocv=(σ=0.11650161909960115, l=0.07224889831676123),
            r=(σ=0.11086743740675992, l=0.0774432873218997)
        ),
        noise=2.8383204772342602e-5
    )
    res = Dict()
    for (id, df) in data
        # load data
        profile = load_profile(df)
        df_train = sample_dataset(profile, tt)

        cu = calc_capa_cccv(df) # check-up

        # fit model
        gp = build_gp(model, θ, df_train, dt)

        res[Symbol(id)] = Dict(:cu => cu, :gp => gp)
    end
    return res
end

function plot_gp_series(data, res)
    tt = 0.0:60.0:(3*24*3600.0)
    tspan = (tt[begin], tt[end])

    fig = Figure(size=(1200, 600))
    ax = [Axis(fig[i, 1]) for i in 1:3]
    ax[1].title = "Measured"
    ax[2].title = "Simulated"
    ax[3].title = "Error"

    linkxaxes!(ax...)

    xlims!(ax[1], tspan)
    xlims!(ax[2], tspan)
    xlims!(ax[3], tspan)
    ylims!(ax[1], (3.3, 4.1))
    ylims!(ax[2], (3.3, 4.1))
    ylims!(ax[3], (-0.15, 0.15))
    ax[1].ylabel = "Voltage (V)"
    ax[2].ylabel = "Voltage (V)"
    ax[3].ylabel = "Voltage (V)"
    ax[3].xlabel = "Time (s)"

    hidexdecorations!(ax[1])
    hidexdecorations!(ax[2])

    colormap = :dense
    colorrange = (0.6, 1.0)

    for (id, df_cell) in data

        profile = load_profile(df_cell)
        df_test = sample_dataset(profile, tt)

        cap = res[id][:cu]
        soh = cap / 4.9

        gp = res[id][:gp]
        vμ, vσ = simulate_gp(gp, df_test)

        v̄ = profile.v(df_test.t)
        δv = vμ - v̄

        lines!(ax[1], df_test.t, v̄; label=id, color=soh, colorrange, colormap)
        lines!(ax[2], df_test.t, vμ; label=id, color=soh, colorrange, colormap)
        lines!(ax[3], df_test.t, δv; label=id, color=soh, colorrange, colormap)

        l2 = sum(abs2, δv)

        @info id soh l2
    end

    Colorbar(fig[:, 2]; colorrange, colormap, label="SOH")

    fig
end

res_gp = fit_gp_series(data)
fig = plot_gp_series(data, res_gp)
save("figs/gp-ecm.svg", fig)


###
function plot_gp_ocv!(res, id, df)
    profile = load_profile(df)
    tt = 0:60:(24*2600)
    df_train = sample_dataset(profile, tt)

    focv, cap = calc_pocv(df)

    soh = res[id][:cu] / 4.9
    gp, dt = res[id][:gp]

    s = 0:0.01:1
    s2 = (s .- 0.385) .* soh
    i = zeros(size(s))
    data = RowVecs([s2 i])
    x = GPPPInput(:ocv, data)

    ocv_gp = gp(x)
    ocvμ = mean(ocv_gp) .+ 3.7
    ocvσ = sqrt.(var(ocv_gp))

    ocv_meas = focv.(s * cap)

    fig = Figure()
    colormap = :dense
    colorrange = (0.6, 1.0)

    e = [extrema(df_train.s)...]
    vl = (e ./ 4.9) .+ 0.385

    ax1 = Axis(fig[1, 1])
    ax1.ylabel = "OCV (V)"
    xlims!(ax1, (0, 1))
    ylims!(ax1, (3.2, 4.2))
    lines!(ax1, s, ocvμ)
    lines!(ax1, s, ocv_meas)
    band!(ax1, s, ocvμ - ocvσ, ocvμ + ocvσ)
    vlines!(ax1, vl, color=(:black, 0.5), linestyle=:dash)

    ax2 = Axis(fig[2, 1])
    ax2.xlabel = "SOC (p.u.)"
    ax2.ylabel = "Error (V)"
    xlims!(ax2, (0, 1))
    ylims!(ax2, (-0.1, 0.1))
    lines!(ax2, s, ocvμ - ocv_meas; color=soh, colorrange, colormap)
    vlines!(ax2, vl, color=(:black, 0.5), linestyle=:dash)

    return fig
end

function plot_gp_ecm(res, id, df)
    profile = load_profile(df)
    tt = 0:60:(24*2600)
    df_train = sample_dataset(profile, tt)

    focv, cap = calc_pocv(df)

    soh = res[id][:cu] / 4.9
    gp, dt = res[id][:gp]

    e = [extrema(df_train.s)...]
    vl = (e ./ 4.9) .+ 0.385

    fig = Figure(size=(1000, 500))

    # OCV
    s = 0:0.01:1
    s2 = (s .- 0.385) .* soh
    i = zeros(size(s))
    data = RowVecs([s2 i])
    x = GPPPInput(:ocv, data)

    ocv_gp = gp(x)
    ocvμ = mean(ocv_gp) .+ 3.7
    ocvσ = sqrt.(var(ocv_gp))

    ocv_meas = focv.(s * cap)

    colormap = :dense
    colorrange = (0.6, 1.0)

    ax1 = Axis(fig[1, 1])
    ax1.ylabel = "OCV (V)"
    xlims!(ax1, (0, 1))
    ylims!(ax1, (3.2, 4.2))
    lines!(ax1, s, ocvμ)
    lines!(ax1, s, ocv_meas)
    band!(ax1, s, ocvμ - ocvσ, ocvμ + ocvσ)
    vlines!(ax1, vl, color=(:black, 0.5), linestyle=:dash)

    # R
    x = GPPPInput(:r, data)

    r = gp(x)
    rμ = mean(r) / 4.9 * 1e3
    rσ = sqrt.(var(r)) / 4.9 * 1e3

    ax2 = Axis(fig[2, 1])
    ax2.xlabel = "SOC (p.u.)"
    ax2.ylabel = "R (mΩ)"
    xlims!(ax2, (0, 1))
    # ylims!(ax2, (-0.1, 0.1))
    lines!(ax2, s, rμ; color=soh, colorrange, colormap)
    band!(ax2, s, rμ - rσ, rμ + rσ)
    vlines!(ax2, vl, color=(:black, 0.5), linestyle=:dash)

    return fig
end

fig = plot_gp_ecm(res_gp, files[end])
fig |> save("figs/ecm-params-fresh-cell.svg")

function plot_gp_ocv_series(files)
    fig = Figure(size=(1000, 600))

    g1 = GridLayout(fig[1, 1])
    ci = CartesianIndices((1:2, 1:5))
    ax = [Axis(g1[Tuple(i)|>first, Tuple(i)|>last]) for i in ci]

    for x in ax[1, :]
        hidexdecorations!(x, grid=false, ticks=false)
    end
    for x in ax[:, 2:end]
        hideydecorations!(x, grid=false, ticks=false)
    end
    for x in ax
        xlims!(x, (0, 1))
        ylims!(x, (3.3, 4.1))
    end

    g2 = GridLayout(fig[2, 1])
    ax2 = Axis(g2[1, 1])
    ax2.xlabel = "SOC (p.u.)"
    ax2.ylabel = "Error (V)"
    xlims!(ax2, (0, 1))
    ylims!(ax2, (-0.1, 0.1))

    colormap = :dense
    colorrange = (0.6, 1.0)

    for (i, file) in enumerate(files)
        df = read_basytec(file)
        focv, cap = calc_pocv(df)

        profile = load_profile(df)
        tt = 0:60:(24*3600)
        df_train = sample_dataset(profile, tt)

        ŝ = df_train.s / 4.9
        î = df_train.i ./ 4.9
        v̂ = df_train.v .- 3.7
        x = GPPPInput(:v, RowVecs([ŝ î]))

        θ = (
            kernel=(
                ocv=(σ=0.11650161909960115, l=0.07224889831676123),
                r=(σ=0.11086743740675992, l=0.0774432873218997)
            ),
            noise=2.8383204772342602e-5
        )
        fp = model(θ.kernel)
        fx = fp(x, θ.noise)
        gp = posterior(fx, v̂)

        plot_gp_ocv!(gp, focv, cap, df_train, ax[i], ax2)

        soh = round((cap / 4.9) * 100, digits=2)
        text = "SOH=$(soh)%"

        # text!(ax[i], 0.45, 3.4; text=text, fontsize = 10)
        ax[i].title = text

    end
    Colorbar(g2[1, 2]; colorrange, colormap, label="SOH")
    return fig
end

fig = plot_gp_ocv_series(files)

save("figs/multi-ocv.svg", fig)




file = "data/check-ups/2098LG_INR21700-M50L_SammyLGL13818NewFullCU.txt"
fig = plot_checkup_profile(file)
fig |> save("figs/checkup-2.svg")

##


# function fit_gp_rc_series(files)
#     res = Dict()

#     # θ = fit_gp_hyperparams(model, θ0, df)
#     θ = (
#         kernel=(
#             ocv=(σ=0.11650161909960115, l=0.07224889831676123),
#             r=(σ=0.11086743740675992, l=0.0774432873218997)
#         ),
#         noise=2.8383204772342602e-5
#     )
#     tt = 0:60:(24*3600)

#     for file in files
#         df = read_basytec(file)
#         id = get_cell_id(file)
#         # pocv, cap = calc_pocv(df)

#         cu = calc_capa_cccv(df) # check-up

#         data = load_profile(df)
#         df_train = sample_dataset(data, tt)

#         s = df_train.s / 4.9
#         i = df_train.i ./ 4.9
#         v = df_train.v .- 3.7
#         x = GPPPInput(:v, RowVecs([s i]))

#         fi = t -> data.i(t)
#         @mtkbuild rc = RC(; fi)
#         tspan = (tt[begin], tt[end])
#         ode = ODEProblem(rc, [rc.v1 => 0.0], tspan, [])
#         p = fit_gp_rc(rc, ode, model, θ, df_train)
#         ode = remake(ode; p)
#         sol = solve(ode, saveat=60)
#         vrc = sol[rc.v1]

#         fp = model(θ.kernel)
#         fx = fp(x, θ.noise)
#         gp = posterior(fx, v - vrc)

#         res[id] = Dict(:cu => cu, :gp => gp, :ode => ode, :model => rc)
#     end
#     return res
# end
function build_loss_rc2(mtk_model, prob, fx, v, t)
    return loss_rc(x) = begin
        @unpack u0, p = x
        newprob = remake(prob; u0, p)
        sol = solve(newprob; saveat=t) # TODO: rate as param
        vrc = sol[mtk_model.v1]
        -logpdf(fx, v - vrc)
    end
end
function fit_gp_rc_series(files)
    res = Dict()

    # θ = fit_gp_hyperparams(model, θ0, df)
    θ = (
        kernel=(
            ocv=(σ=0.11650161909960115, l=0.07224889831676123),
            r=(σ=0.11086743740675992, l=0.0774432873218997)
        ),
        noise=2.8383204772342602e-5
    )
    tt = 0:60:(24*3600)

    for file in files
        df = read_basytec(file)
        id = get_cell_id(file)
        # pocv, cap = calc_pocv(df)

        cu = calc_capa_cccv(df) # check-up

        profile = load_profile(df)
        df_train = sample_dataset(profile, tt)

        s = df_train.s / 4.9
        i = df_train.i ./ 4.9
        v = df_train.v .- 3.7
        x = GPPPInput(:v, RowVecs([s i]))
        fp = model(θ.kernel)
        fx = fp(x, θ.noise)

        fi = t -> profile.i(t)
        @mtkbuild rc = RC(; fi)
        tspan = (tt[begin], tt[end])
        ode = ODEProblem(rc, [rc.v1 => 0.0], tspan, [])

        loss = build_loss_rc2(rc, ode, fx, v, df_train.t)
        p0 = ComponentArray((;
            u0=ode.u0,
            p=ode.p
        ))

        adtype = AutoForwardDiff()
        f = OptimizationFunction((u, p) -> loss(u), adtype)
        prob = OptimizationProblem(f, p0)

        sol = solve(prob, LBFGS(); show_trace=true)

        ode = remake(ode; sol.u...)
        sol = solve(ode, saveat=60)
        vrc = sol[rc.v1]

        fp = model(θ.kernel)
        fx = fp(x, θ.noise)
        gp = posterior(fx, v - vrc)

        res[id] = Dict(:cu => cu, :gp => gp, :ode => ode, :model => rc)
    end
    return res
end

res = fit_gp_rc_series(files)

function plot_gp_rc_series(files)
    fig = Figure(size=(1200, 600))
    ax = [Axis(fig[i, 1]) for i in 1:3]
    ax[1].title = "Measured"
    ax[2].title = "Simulated"
    ax[3].title = "Error"

    tspan = (0.0, 3 * 24 * 3600.0)
    xlims!(ax[1], tspan)
    xlims!(ax[2], tspan)
    xlims!(ax[3], tspan)
    ylims!(ax[1], (3.3, 4.1))
    ylims!(ax[2], (3.3, 4.1))
    ylims!(ax[3], (-0.15, 0.15))
    ax[1].ylabel = "Voltage (V)"
    ax[2].ylabel = "Voltage (V)"
    ax[3].ylabel = "Voltage (V)"
    ax[3].xlabel = "Time (s)"

    hidexdecorations!(ax[1])
    hidexdecorations!(ax[2])

    colormap = :dense
    colorrange = (0.6, 1.0)

    for file in files
        df = read_basytec(file)
        id = get_cell_id(file)
        pocv, cap = calc_pocv(df)
        soh = cap / 4.9

        profile = load_profile(df)

        tt = 0:60:(3*24*3600)
        df_test = sample_dataset(profile, tt)

        # rc
        ode = res[id][:ode]
        rc = res[id][:model]

        tspan = (tt[begin], tt[end])
        ode = remake(ode; tspan)
        sol = solve(ode, saveat=df_test.t)
        vrc = sol[rc.v1]

        # gp
        ŝ = df_test.s / 4.9
        î = df_test.i ./ 4.9
        v̄ = df_test.v
        x = GPPPInput(:v, RowVecs([ŝ î]))

        gp = res[id][:gp]

        # results
        v = gp(x)
        vμ = mean(v) + vrc .+ 3.7
        δv = vμ - v̄

        lines!(ax[1], df_test.t, v̄; color=soh, colorrange, colormap)
        lines!(ax[2], df_test.t, vμ; color=soh, colorrange, colormap)
        lines!(ax[3], df_test.t, δv; color=soh, colorrange, colormap)

        l2 = sum(abs2, δv)

        @info id soh l2

    end

    Colorbar(fig[:, 2]; colorrange, colormap, label="SOH")
    return fig
end

res_rc = fit_gp_rc_series(files)
fig = plot_gp_rc_series(files)

fig |> save("figs/gp-rc-ecm.svg")

###
begin
    fig = Figure()
    ax = [Axis(fig[i, 1]) for i in 1:3]
    colormap = :dense
    colorrange = (0.6, 1.0)

    for file in files

        df_cell = read_basytec(file)
        profile = load_profile(df_cell)
        cap = calc_capa_cccv(df_cell)
        soh = cap / 4.9
        tt = 0.0:60.0:(24*3600.0)

        df = sample_dataset(profile, tt)

        lines!(ax[1], df.t, df.v; color=soh, colorrange, colormap)
        lines!(ax[2], df.t, df.i; color=soh, colorrange, colormap)
        lines!(ax[3], df.t, df.s / cap; color=soh, colorrange, colormap)
    end
    Colorbar(fig[:, 2]; colorrange, colormap, label="SOH")
    fig
end