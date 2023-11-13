using CSV
using DataFrames
using DataInterpolations

using ModelingToolkit
using DifferentialEquations

using Stheno

using Optimization
using OptimizationOptimJL
import ComponentArrays: ComponentArray

# using GLMakie
using CairoMakie
CairoMakie.activate!()

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
    # ax4 = Axis(fig2[1, 2]; xlabel, title="OCV Hystheresis")
    ax4 = Axis(fig2[1, 2]; xlabel, title="Deviation")

    focv = fresh_focv()

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
        # lines!(ax4, s, pocv(c) - ocv_c(c); color=cap / 4.9, colorrange, colormap)

        # degradation
        if !occursin("LGL13818", file)
            lines!(ax4, s, pocv(c) - focv(s); color=cap / 4.9, colorrange, colormap)
        end
    end

    Colorbar(fig1[:, 3]; colorrange, colormap, label="SOH")
    Colorbar(fig2[:, 3]; colorrange, colormap, label="SOH")
    return fig1, fig2
end


# ------
function fit_ecm_series(files)
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
        id = get_cell_id(file)

        df_cell = read_basytec(file)
        data = load_profile(df_cell)

        cap = prob = res[Symbol(id)][:cu]
        soh = cap / 4.9

        prob = res[Symbol(id)][:ode]
        ecm = res[Symbol(id)][:model]
        new_prob = remake(prob; tspan)
        sol = solve(new_prob; saveat=60)

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
res = fit_ecm_series(files)

for key in keys(res)
    estimated = res[key][:ode].p[1] / 4.9
    measured = res[key][:cu] / 4.9
    e = (estimated - measured) * 100
    @info key estimated measured e
end

fig = plot_ecm_series(res)

fig |> save("figs/ecm.svg")



### GP
θ = (;
    kernel=(;
        ocv=(; σ=positive(0.1), l=positive(0.1)),
        # hys=(; σ=positive(0.1), l=positive(0.1)),
        r=(; σ=positive(0.1), l=positive(0.1))
        # r=positive(0.1)
    ),
    noise=positive(1e-5)
)
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


function fit_gp_hyperparams(model, θ, df)

    ŝ = df.s / 4.9
    î = df.i ./ 4.9
    v̂ = df.v .- 3.7
    x = GPPPInput(:v, RowVecs([ŝ î]))

    loss = create_loss_function(model, x, v̂)
    θ0, unflatten = ParameterHandling.value_flatten(θ)
    loss_packed = loss ∘ unflatten

    adtype = AutoZygote()
    f = OptimizationFunction((u, p) -> loss_packed(u), adtype)
    prob = OptimizationProblem(f, θ0)

    sol = solve(prob, LBFGS(); show_trace=true)
    θ_opt = unflatten(sol.u)

    return θ_opt
end


# θ = fit_gp_hyperparams(model, θ, df_train)
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


function plot_gp_ocv(gp, focv, cap, df, ax1, ax2)
    soh = cap / 4.9
    s = 0:0.01:1
    s2 = (s .- 0.385) .* soh
    i = zeros(size(s))
    data = RowVecs([s2 i])
    x = GPPPInput(:ocv, data)
    ocv_gp = gp(x)
    ocvμ = mean(ocv_gp) .+ 3.7
    ocvσ = sqrt.(var(ocv_gp))

    ocv_meas = focv.(s * cap)

    # fig = Figure()

    e = [extrema(df.s)...]
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
    lines!(ax2, s, ocvμ - ocv_meas)
    vlines!(ax2, vl, color=(:black, 0.5), linestyle=:dash)

    return fig
end

# fig = plot_gp_ocv(gp)
# save("figs/gp-ocv1.png", fig, px_per_unit=2)


for file in files
    df = read_basytec(file)
    focv, cap = calc_pocv(df)

    data = load_profile(df)
    tt = 0:60:(24*3600)
    df_train = sample_dataset(data, tt)

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

    fig = plot_gp_ocv(gp, focv, cap, df_train)
    fig |> display
end



function plot_gp_series(files)
    fig = Figure(resolution=(1200, 600))
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

        data = load_profile(df)

        tt = 0:60:(24*3600)
        df_train = sample_dataset(data, tt)

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


        tt = 0:60:(3*24*3600.0)
        df_test = sample_dataset(data, tt)

        ŝ = df_test.s / 4.9
        î = df_test.i ./ 4.9
        v̄ = df_test.v
        x = GPPPInput(:v, RowVecs([ŝ î]))

        v = gp(x)
        vμ = mean(v) .+ 3.7
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

fig = plot_gp_series(files)
save("figs/gp-ecm.svg", fig)


###
function plot_gp_ocv!(gp, focv, cap, df, ax1, ax2)
    colormap = :dense
    colorrange = (0.6, 1.0)

    soh = cap / 4.9
    s = 0:0.01:1
    s2 = (s .- 0.385) .* soh
    i = zeros(size(s))
    data = RowVecs([s2 i])
    x = GPPPInput(:ocv, data)
    ocv_gp = gp(x)
    ocvμ = mean(ocv_gp) .+ 3.7
    ocvσ = sqrt.(var(ocv_gp))

    ocv_meas = focv.(s * cap)

    # fig = Figure()

    e = [extrema(df.s)...]
    vl = (e ./ 4.9) .+ 0.385

    # ax1 = Axis(fig[1, 1])
    # ax1.ylabel = "OCV (V)"
    # xlims!(ax1, (0, 1))
    # ylims!(ax1, (3.2, 4.2))
    lines!(ax1, s, ocvμ)
    lines!(ax1, s, ocv_meas)
    band!(ax1, s, ocvμ - ocvσ, ocvμ + ocvσ)
    vlines!(ax1, vl, color=(:black, 0.5), linestyle=:dash)

    # ax2 = Axis(fig[2, 1])
    # ax2.xlabel = "SOC (p.u.)"
    # ax2.ylabel = "Error (V)"
    # xlims!(ax2, (0, 1))
    # ylims!(ax2, (-0.1, 0.1))
    lines!(ax2, s, ocvμ - ocv_meas; color=soh, colorrange, colormap)
    vlines!(ax2, vl, color=(:black, 0.5), linestyle=:dash)

    # return fig
end


function plot_gp_ocv_series(files)
    fig = Figure(resolution=(1000, 600))

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

        data = load_profile(df)
        tt = 0:60:(24*3600)
        df_train = sample_dataset(data, tt)

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


begin
    file = "data/check-ups/2098LG_INR21700-M50L_SammyLGL13818NewFullCU.txt"
    df = read_basytec(file)

    fig = Figure(resolution=(1200, 400))
    ax = Axis(fig[1, 1])
    n = nrow(df)
    lines!(ax, df[1:60:n, "Time[h]"], df[1:60:n, "U[V]"])

    xlims!(ax, (df[begin, "Time[h]"], df[end, "Time[h]"]))
    ax.xlabel = "Time (h)"
    ax.ylabel = "Voltage (V)"
    fig
end

save("figs/checkup.svg", fig)

