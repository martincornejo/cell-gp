function plot_gp_ecm(df, gp_model)
    profile = load_profile(df)
    tt = 0:60:(24*2600)
    df_train = sample_dataset(profile, tt)

    focv = calc_pocv(df)

    cap = calc_capa_cccv(df)
    soh = calc_capa_cccv(df) / 4.9
    @unpack gp, dt = gp_model

    vl = @. (extrema(df_train.s) / cap) + 0.385

    fig = Figure()

    # OCV
    s = 0:0.01:1
    ŝ = StatsBase.transform(dt.s, (s .- 0.385) .* cap)
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
    band!(ax1, s, ocvμ - ocvσ, ocvμ + ocvσ)
    vlines!(ax1, vl, color=(:black, 0.5), linestyle=:dash)

    # s̄ = (df_train.s ./ cap) .+ 0.385
    # scatter!(s̄, df_train.v; color=abs.(df_train.i), alpha=0.5)

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
    ylims!(ax2, (0.0, 90))
    lines!(ax2, s, rμ; color=soh, colorrange, colormap)
    band!(ax2, s, rμ - rσ, rμ + rσ)
    vlines!(ax2, vl, color=(:black, 0.5), linestyle=:dash)

    return fig
end

