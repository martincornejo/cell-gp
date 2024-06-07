function simulation_voltage(ecms, gpms, data)
    # time-span
    tt = 0:60:(6.9*24*3600)

    res = Dict()
    ids = sort_cell_ids(data)
    for id in ids
        df = data[id]
        ecm_model = ecms[id]
        gp_model = gpms[id]

        # real voltage
        profile = load_profile(df)
        df_test = sample_dataset(profile, tt)
        v = df_test.v

        # ECM
        (; ecm, ode) = ecm_model
        ode = remake(ode; tspan=(tt[begin], tt[end]))
        sol = solve(ode, Tsit5(); saveat=tt)
        v_ecm = sol[ecm.v]

        # GP-ECM
        v_gp = simulate_gp_rc(gp_model, df_test)
        v_gpμ = v_gp.μ
        v_gpσ = v_gp.σ
        res[id] = DataFrame(; v, v_ecm, v_gpμ, v_gpσ)
    end

    return res
end


function simulation_error(ecm_model, gp_model, df)
    # time-span
    tt = 0:60:(6.9*24*3600)

    # real voltage
    profile = load_profile(df)
    df_test = sample_dataset(profile, tt)
    v̄ = df_test.v

    # ECM
    (; ecm, ode) = ecm_model
    ode = remake(ode; tspan=(tt[begin], tt[end]))
    sol = solve(ode, Tsit5(); saveat=tt)
    v = sol[ecm.v]
    rmse_ecm = sqrt(mean(abs2, (v - v̄))) * 1e3

    # GP-ECM
    v = simulate_gp_rc(gp_model, df_test)
    rmse_gpm = sqrt(mean(abs2, (v.μ - v̄))) * 1e3

    return (; rmse_ecm, rmse_gpm)
end

function benchmark_sim(ecms, gpms, data)
    ids = sort_cell_ids(data)
    sim = [simulation_error(ecms[id], gpms[id], data[id]) for id in ids]
    rmse_ecm = sim .|> first
    rmse_gpm = sim .|> last
    DataFrame(; id=ids, rmse_ecm, rmse_gpm)
end
