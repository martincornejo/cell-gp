
function plot_ocv_fit(ecms, gpms, data, ids; correct_soc=true)
    fig = Figure(size=(512, 700), fontsize=10, figure_padding=5)
    gl = [GridLayout(fig[ci[2], ci[1]]) for ci in CartesianIndices((1:2, 1:5))]

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

        ## ECM
        # function ocv(ecm, df)
        pocv = calc_pocv(df)
        focv = fresh_focv()
        s = 0:0.001:1

        cap_real = calc_capa_cccv(df)
        cap_sim = ecm.ode.p[1]

        soc0 = initial_soc(df)
        soc0_sim = ecm.ode.u0[1]
        Δsoc = correct_soc ? soc0_sim - soc0 : 0.0

        # real
        c1 = s .* cap_real
        ocv1 = pocv(s)
        ln1 = lines!(ax1, c1, ocv1; label="pOCV", color=:black)

        # estimated
        c2 = (s .- Δsoc) .* cap_sim
        ocv2 = focv(s)
        ln2 = lines!(ax1, c2, ocv2; label="ECM")

        ## GP
        @unpack gp, dt = gpm
        ŝ = StatsBase.transform(dt.s, (s .- soc0) * cap_real)
        x = GPPPInput(:ocv, RowVecs([ŝ zero.(ŝ)]))
        ocv3 = gp(x)
        μ = StatsBase.reconstruct(dt.v, mean(ocv3))
        σ = StatsBase.reconstruct(dt.σ, sqrt.(var(ocv3)))
        ln3 = lines!(ax1, c1, μ; label="GP-ECM", color=Cycled(2))
        bnd = band!(ax1, c1, μ - 2σ, μ + 2σ; label="GP-ECM", color=(Makie.wong_colors()[2], 0.3))

        ## SOC
        tt = 0:60:(3*24*3600)
        profile = load_profile(df)
        df_train = sample_dataset(profile, tt)

        hst = hist!(ax2, df_train.s .+ soc0 * cap_real, color=(:gray, 0.3), label="SOC")
        smin, smax = extrema(df_train.s) .+ soc0 * cap_real
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
        soh = round(cap_real / 4.85 * 100; digits=1)
        text = "Cell $i \nSOH: $soh %"
        poly!(ax1, Rect(0.1, 3.95, 1.5, 0.29), color=:white, strokecolor=:black, strokewidth=1)
        text!(ax1, 0.15, 4.2; text, font=:bold, align=(:left, :top))

        linkxaxes!(ax1, ax2)
    end

    fig
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

    focv = fresh_focv()
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
