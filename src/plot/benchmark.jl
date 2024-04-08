
function plot_ocv_fit(ecms, gpms, data, ids; correct_soc=true)
    n = length(ids)
    fig = Figure(size=(400 * n, 400))

    for (i, id) in enumerate(ids)
        df = data[id]
        ecm = ecms[id]
        gpm = gpms[id]

        ax1 = Axis(fig[1, i])
        ax2 = Axis(fig[1, i], yaxisposition=:right)
        ax1.ylabel = "Voltage / V"
        ax1.xlabel = "Capacity / Ah"
        ylims!(ax1, 3.3, 4.3)
        ylims!(ax2, 0, 1e3)

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

        soh = round(cap_real / 4.9 * 100; digits=2)
        ax1.title = "ID:$id SOH: $soh %"

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
        hideydecorations!(ax2)
        # axislegend(ax1; position=:rb, merge=true)
        # axislegend(ax2; position=:lt)
        if i == n
            Legend(fig[2, :],
                [hst, ln1, ln2, [ln3, bnd]],
                ["SOC distribution in the training dataset", "Measured pOCV", "ECM fitted OCV", "GP-ECM fitted OCV"],
                orientation=:horizontal, tellwidth=false, tellheight=true
            )
        end

        smin, smax = extrema(df_train.s) .+ soc0 * cap_real
        vlines!(ax1, [smin, smax], color=:gray, linestyle=:dashdot)

        linkxaxes!(ax1, ax2)
    end
    fig
end

function plot_rint_fit(df)
    fig = Figure(size=(252, 252), fontsize=10, figure_padding=5)
    ax = Axis(fig[1, 1], xlabel="Measured pulse resistance / mΩ", ylabel="Estimated pulse resistance / mΩ")
    ylims!(ax, 18, 95)
    xlims!(ax, 18, 95)
    ablines!(ax, 0, 1, linestyle=:dashdot, color=:black)

    # ECM
    scatter!(ax, df.r_cu, df.r_ecm, label="ECM")

    # ECM-GP
    μ = Measurements.value.(df.r_gp)
    σ = Measurements.uncertainty.(df.r_gp)
    scatter!(ax, df.r_cu, μ, label="GP-ECM")
    errorbars!(ax, df.r_cu, μ, 2σ; label="GP-ECM", whiskerwidth=5, color=Makie.wong_colors()[2])

    # text = rich(
    #     rich("Pulse conditions:", font=:bold), "\n",
    #     "SOC = 50 %", "\n",
    #     "t = 10 s", "\n",
    #     "i = C/3",
    # )
    # text!(ax, 26, 91; text, align=(:left, :top))

    axislegend(ax; merge=true, position=:rb)
    fig
end


function plot_soh_fit(df)
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="Check-up SOH / %", ylabel="Estimated SOH / %")
    ablines!(ax, 0, 1, linestyle=:dashdot, color=:black)

    # ECM
    scatter!(ax, df.soh_cu, df.soh_ecm, label="ECM")

    # ECM-GP
    μ = Measurements.value.(df.soh_ocv_gp)
    σ = Measurements.uncertainty.(df.soh_ocv_gp)
    scatter!(ax, df.soh_cu, μ, label="GP-ECM")
    errorbars!(ax, df.soh_cu, μ, 2σ; label="GP-ECM", whiskerwidth=5, color=Makie.wong_colors()[2])
    # scatter!(ax, df.soh_cu, df.soh_ocv_ecm, label="ECM-OCV")

    axislegend(ax; merge=true, position=:rb)
    fig
end
