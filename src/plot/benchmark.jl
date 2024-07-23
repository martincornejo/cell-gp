function plot_sim(ecms, gpms, data)
    fig = Figure(size=(512, 250), fontsize=10, figure_padding=5)
    gl = GridLayout(fig[1, 1])
    ax = [Axis(gl[i, 1]) for i in 1:3]
    rowgap!(gl, 5)
    ax[1].title = "Measured"
    ax[2].title = "Error ECM"
    ax[3].title = "Error GP-ECM"

    tt = 0:60:(6.9*24*3600)
    tspan = (tt[begin], tt[end])
    tlims = tspan ./ 3600
    xlims!(ax[1], tlims)
    xlims!(ax[2], tlims)
    xlims!(ax[3], tlims)
    ylims!(ax[1], (3.2, 4.2))
    ylims!(ax[2], (-75, 75))
    ylims!(ax[3], (-75, 75))
    ax[1].xticks = 0:12:(7*24)
    ax[2].xticks = 0:12:(7*24)
    ax[3].xticks = 0:12:(7*24)
    ax[1].yticks = [3.4, 3.8, 4.2]
    ax[2].yticks = [-50, 0, 50]
    ax[3].yticks = [-50, 0, 50]
    ax[1].yminorticks = [3.2, 3.6, 4.0]
    ax[2].yminorticks = [-75, -25, 25, 75]
    ax[3].yminorticks = [-75, -25, 25, 75]
    ax[1].yminorticksvisible = true
    ax[2].yminorticksvisible = true
    ax[3].yminorticksvisible = true
    ax[1].yminorgridvisible = true
    ax[2].yminorgridvisible = true
    ax[3].yminorgridvisible = true
    ax[1].ylabel = "Voltage / V"
    ax[2].ylabel = "Error / mV"
    ax[3].ylabel = "Error / mV"
    ax[3].xlabel = "Time / h"

    hidexdecorations!(ax[1], ticks=false, grid=false)
    hidexdecorations!(ax[2], ticks=false, grid=false)

    for axis in ax
        vlines!(axis, 72, color=:gray, linestyle=:dashdot)
    end

    colormap = ColorSchemes.dense
    colorrange = (0.6, 1.0)
    linewidth = 1.2

    ids = sort_cell_ids(data)
    sim = simulation_voltage(ecms, gpms, data)

    for id in reverse(ids)
        # SOH
        df = data[id]
        soh = calc_capa_cccv(df) / 4.8
        color = get(colormap, soh, colorrange)

        # simulation results
        (; t, v̄, δv_ecm, δv_gp) = sim[id]

        # plot
        lines!(ax[1], t / 3600, v̄; color, linewidth)
        lines!(ax[2], t / 3600, δv_ecm; color, linewidth)
        lines!(ax[3], t / 3600, δv_gp; color, linewidth)
    end

    linkxaxes!(ax...)
    Colorbar(fig[:, 2]; colorrange, colormap, label="SOH / p.u.")
    return fig
end

function plot_ocv_fit(ecms, gpms, data)
    fig = Figure(size=(512, 700), fontsize=10, figure_padding=5)
    gl = [GridLayout(fig[ci[2], ci[1]]) for ci in CartesianIndices((1:2, 1:5))]

    # params
    focv = fresh_focv(data)
    s = 0:0.001:1

    ids = sort_cell_ids(data)
    for (i, id) in enumerate(ids)
        df = data[id]
        ecm = ecms[id]
        gpm = gpms[id]

        ax1 = Axis(gl[i][1, 1])
        ax2 = Axis(gl[i][1, 1], yaxisposition=:right)
        ax1.ylabel = "Voltage / V"
        ax1.xlabel = "Capacity / Ah"
        ylims!(ax1, 3.3, 4.3)
        ylims!(ax2, 0, 1e3)
        xlims!(ax1, 0, 5)
        xlims!(ax2, 0, 5)

        # parameters
        pocv = calc_pocv(df)
        cap_cu = calc_capa_cccv(df)
        cap_ecm = ecm.ode.p[1]
        soc0 = initial_soc(df)

        # ECM
        q_ecm = s .* cap_ecm
        ocv_ecm = focv(s)
        ln2 = lines!(ax1, q_ecm, ocv_ecm; label="ECM")

        # GP-ECM
        (; gp, dt) = gpm
        q_gp = 0:0.005:5
        ŝ = StatsBase.transform(dt.q, q_gp .- soc0 * cap_cu)
        x = GPPPInput(:ocv, RowVecs([ŝ zero.(ŝ)]))
        ocv_gp = gp(x)
        μ = StatsBase.reconstruct(dt.v, mean(ocv_gp))
        σ = StatsBase.reconstruct(dt.σ, sqrt.(var(ocv_gp)))
        ln3 = lines!(ax1, q_gp, μ; label="GP-ECM", color=Cycled(2))
        bnd = band!(ax1, q_gp, μ - 2σ, μ + 2σ; label="GP-ECM", color=(Makie.wong_colors()[2], 0.3))

        # Measured pOCV
        q_cu = s .* cap_cu
        ocv_cu = pocv(s)
        ln1 = lines!(ax1, q_cu, ocv_cu; label="pOCV", color=:black, linestyle=:dot)

        # SOC distribution
        tt = 0:60:(3*24*3600)
        profile = load_profile(df)
        df_train = sample_dataset(profile, tt)

        hst = hist!(ax2, df_train.q .+ soc0 * cap_cu, color=(:gray, 0.3), label="SOC")
        smin, smax = extrema(df_train.q) .+ soc0 * cap_cu
        vlines!(ax1, [smin, smax], color=:gray, linestyle=:dashdot)
        hidedecorations!(ax2)
        hidespines!(ax2)

        # hide ticklabels
        if i % 2 == 0
            hideydecorations!(ax1, ticks=false, grid=false)
        end
        if i ∉ (9, 10)
            hidexdecorations!(ax1, ticks=false, grid=false)
        end

        # legend
        if i == 10
            Legend(fig[6, :],
                [hst, ln1, ln2, [ln3, bnd]],
                ["SOC distribution in \nthe training dataset", "Measured pOCV", "ECM OCV", "GP-ECM OCV"],
                orientation=:horizontal, tellwidth=false, tellheight=true,
                nbanks=1,
            )
        end

        # text box
        soh = round(cap_cu / 4.8 * 100; digits=1)
        soh = soh > 100 ? 100 : soh
        text = "Cell $i \nSOH: $soh %"
        poly!(ax1, Rect(0.1, 3.95, 1.45, 0.29), color=:white, strokecolor=:black, strokewidth=1)
        text!(ax1, 0.15, 4.2; text, font=:bold, align=(:left, :top))

        linkxaxes!(ax1, ax2)
    end

    fig
end

function plot_gp_rint(gpms, data)
    fig = Figure(size=(252, 252), fontsize=10, figure_padding=7)
    gl = GridLayout(fig[1, 1])

    colormap = ColorSchemes.dense
    colorrange = (0.6, 1.0)
    Colorbar(gl[1, 1]; vertical=false, colorrange, colormap, label="SOH / p.u.")

    ax = Axis(gl[2, 1])
    ax.xlabel = "SOC / p.u."
    ax.ylabel = "R₀ / mΩ"
    xlims!(ax, (0.2, 0.75))
    ylims!(ax, (12, 115))

    ids = sort_cell_ids(data)
    for id in ids
        df = data[id]
        gpm = gpms[id]
        (; gp, dt) = gpm

        soc0 = initial_soc(df)
        cap = calc_capa_cccv(df)
        soh = cap / 4.8
        color = get(colormap, soh, (0.6, 1.0))

        # R0
        s = 0.2:0.01:0.75
        ŝ = StatsBase.transform(dt.q, (s .- soc0) * cap)
        i = zeros(size(s))
        x = GPPPInput(:r, RowVecs([ŝ i]))
        r0 = 1e3 * dt.σ.scale[1] / dt.i.scale[1] # scale to mΩ

        r = gp(x)
        rμ = mean(r) * r0
        rσ = sqrt.(var(r)) * r0

        lines!(ax, s, rμ; color)
        band!(ax, s, rμ - 2rσ, rμ + 2rσ; color=(color, 0.5))
    end

    rowgap!(gl, 5)
    return fig
end

function plot_rint_fit(ecms, gpms, data)
    fig = Figure(size=(252, 280), fontsize=10, figure_padding=7)
    gl = GridLayout(fig[1, 1])

    ## colorbars
    cmap_ecm = ColorSchemes.Blues
    cmap_gpm = ColorSchemes.Oranges
    colorrange = (0.0, 1.0)

    Colorbar(gl[1, 1]; vertical=false, colorrange, colormap=cmap_ecm, label="SOC / p.u.")
    Colorbar(gl[2, 1]; vertical=false, colorrange, colormap=cmap_gpm, ticksvisible=false, ticklabelsvisible=false)

    ## scatter
    ax = Axis(gl[3, 1],
        xlabel="Measured pulse resistance / mΩ",
        ylabel="Estimated pulse resistance / mΩ"
    )
    ylims!(ax, 16, 115)
    xlims!(ax, 16, 115)
    ablines!(ax, 0, 1, linestyle=:dashdot, color=:black)
    ablines!(ax, 0, 1.1, linestyle=:dashdot, color=:gray)
    ablines!(ax, 0, 0.9, linestyle=:dashdot, color=:gray)
    # ablines!(ax, 0, 1.05, linestyle=:dashdot, color=:gray)
    # ablines!(ax, 0, 0.95, linestyle=:dashdot, color=:gray)

    focv = fresh_focv(data)
    socs = 0.3:0.1:0.7
    ids = sort_cell_ids(data)

    # checkup
    df_cup = DataFrame(; soc=0.1:0.1:0.9)
    for id in ids
        df_id = calc_rint(data[id])
        df_cup[!, id] .= df_id.r
    end
    filter!(:soc => ∈(socs), df_cup)

    # ecm
    df_ecm = DataFrame(; soc=socs)
    for id in ids
        df_id = [simulate_pulse_resistance_ecm(ecms[id], focv; soc) for soc in socs]
        df_ecm[!, id] .= df_id
    end

    # gp-ecm
    df_gpm = DataFrame(; soc=socs)
    for id in ids
        df_id = [simulate_pulse_resistance_gpm(gpms[id], data[id]; soc) for soc in socs]
        df_gpm[!, id] .= df_id
    end

    # plot
    for (i, soc) in enumerate(socs)
        # df
        r_cup = df_cup[i, ids] |> Array |> vec
        r_ecm = df_ecm[i, ids] |> Array |> vec
        r_gpm = df_gpm[i, ids] |> Array |> vec

        # ecm
        color = get(cmap_ecm, soc)
        sc1 = scatter!(ax, r_cup, r_ecm; color)

        # gp-ecm
        color = get(cmap_gpm, soc)
        μ = Measurements.value.(r_gpm)
        σ = Measurements.uncertainty.(r_gpm)
        sc2 = scatter!(ax, r_cup, μ; color)
        err = errorbars!(ax, r_cup, μ, 2σ; color, whiskerwidth=5)

        # legend
        if soc == 0.6
            axislegend(ax, [sc1, [sc2, err]], ["ECM", "GP-ECM"], position=:lt)
        end
    end

    rowgap!(gl, 5)
    fig
end
