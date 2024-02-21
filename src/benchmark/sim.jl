function simulation_error(ecm_model, gp_model, df, tt)
    # real voltage
    profile = load_profile(df)
    df_test = sample_dataset(profile, tt)
    v̄ = df_test.v

    # ECM
    @unpack ecm, ode = ecm_model
    ode = remake(ode; tspan=(tt[begin], tt[end]))
    sol = solve(ode, Tsit5(); saveat=tt)
    v = sol[ecm.v]
    l2_ecm = sum(abs2, v - v̄)

    # GP-ECM
    v = simulate_gp_rc(gp_model, df_test)
    l2_gp = sum(abs2, v.μ - v̄)

    return (; l2_ecm, l2_gp)
end

function benchmark_simulation(ecms, gpms, data, tt)
    sohs = [calc_capa_cccv(df) / 4.9 for df in values(data)]
    ids = collect(keys(data))[sortperm(sohs)] # sort cell-id by soh

    sim = ids .|> id -> simulation_error(ecms[id], gpms[id], data[id], tt)
    l2_ecm = sim .|> first
    l2_gp = sim .|> last
    DataFrame(; id=ids, l2_ecm, l2_gp)
end

function plot_sim(ecm_models, gp_models, data)
    fig = Figure(size=(1200, 600))
    ax = [Axis(fig[i, 1]) for i in 1:3]
    ax[1].title = "Measured"
    ax[2].title = "Error ECM"
    ax[3].title = "Error GP-ECM"

    tt = 0:60:(3*24*3600)
    tspan = (tt[begin], tt[end])
    tlims = tspan ./ 3600
    xlims!(ax[1], tlims)
    xlims!(ax[2], tlims)
    xlims!(ax[3], tlims)
    ylims!(ax[1], (3.3, 4.1))
    ylims!(ax[2], (-75, 75))
    ylims!(ax[3], (-75, 75))
    ax[1].xticks = 0:12:(3*24)
    ax[2].xticks = 0:12:(3*24)
    ax[3].xticks = 0:12:(3*24)
    ax[1].ylabel = "Voltage (V)"
    ax[2].ylabel = "Voltage (mV)"
    ax[3].ylabel = "Voltage (mV)"
    ax[3].xlabel = "Time (h)"

    hidexdecorations!(ax[1], ticks=false, grid=false)
    hidexdecorations!(ax[2], ticks=false, grid=false)

    colormap = :dense
    colorrange = (0.6, 1.0)


    ids = collect(keys(data))[sortperm([calc_capa_cccv(df) / 4.9 for df in values(data)])]
    for id in ids
        # SOH
        df = data[id]
        soh = calc_capa_cccv(df) / 4.9

        # real voltage
        profile = load_profile(df)
        df_test = sample_dataset(profile, tt)
        v̄ = df_test.v
        t = df_test.t ./ 3600

        # ECM
        @unpack ecm, ode = ecm_models[id]
        ode = remake(ode; tspan)
        sol = solve(ode, Tsit5(); saveat=tt)
        v = sol[ecm.v]
        δv_ecm = (v - v̄) * 1e3

        # GP-ECM
        gp = gp_models[id]
        v = simulate_gp_rc(gp, df_test)
        δv_gp = (v.μ - v̄) * 1e3

        # plot
        lines!(ax[1], t, v̄; color=soh, colorrange, colormap)
        lines!(ax[2], t, δv_ecm; color=soh, colorrange, colormap)
        lines!(ax[3], t, δv_gp; color=soh, colorrange, colormap)
    end

    linkxaxes!(ax...)
    Colorbar(fig[:, 2]; colorrange, colormap, label="SOH")
    return fig
end
