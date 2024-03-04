function plot_gp_ecm(df, gp_model)
    fig = Figure()

    # soc bound
    cap = calc_capa_cccv(df)
    profile = load_profile(df)
    tt = 0:(3*24*2600)
    df_train = sample_dataset(profile, tt)

    colormap = :dense
    colorrange = (0.6, 1.0)

    soc0 = initial_soc(df)

    # model
    focv = calc_pocv(df)
    cap = calc_capa_cccv(df)
    soh = cap / 4.9
    @unpack gp, dt = gp_model

    # OCV
    s = 0:0.01:1
    ŝ = StatsBase.transform(dt.s, (s .- soc0) * cap)
    i = zeros(size(s))
    data = RowVecs([ŝ i])
    x = GPPPInput(:ocv, data)

    ocv_gp = gp(x)
    ocvμ = StatsBase.reconstruct(dt.v, mean(ocv_gp))
    ocvσ = StatsBase.reconstruct(dt.σ, sqrt.(var(ocv_gp)))

    ocv_meas = focv(s)

    ax1 = Axis(fig[1, 1])
    ax1.ylabel = "OCV (V)"
    xlims!(ax1, (0, 1))
    ylims!(ax1, (3.2, 4.2))
    lines!(ax1, s, ocvμ)
    lines!(ax1, s, ocv_meas)
    band!(ax1, s, ocvμ - 2ocvσ, ocvμ + 2ocvσ)

    vl = [((extrema(df_train.s) ./ cap) .+ soc0)...]
    vlines!(ax1, vl, color=(:black, 0.5), linestyle=:dash)

    # s̄ = (df_train.s ./ cap) .+ soc0
    # scatter!(s̄, df_train.v; color=abs.(df_train.i), alpha=0.01)

    # R
    x = GPPPInput(:r, data)
    r0 = 1e3 * dt.σ.scale[1] / dt.i.scale[1] # scale to mΩ

    r = gp(x)
    rμ = mean(r) * r0
    rσ = sqrt.(var(r)) * r0

    ax2 = Axis(fig[2, 1])
    ax2.xlabel = "SOC (p.u.)"
    ax2.ylabel = "R (mΩ)"
    xlims!(ax2, (0, 1))
    ylims!(ax2, (60, 120))
    # ylims!(ax2, (0.0, 80))
    lines!(ax2, s, rμ; color=soh, colorrange, colormap=colormap)
    band!(ax2, s, rμ - 2rσ, rμ + 2rσ)
    vlines!(ax2, vl, color=(:black, 0.5), linestyle=:dash)

    return fig
end

