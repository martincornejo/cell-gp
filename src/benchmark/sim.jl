function simulation_voltage(ecms, gpms, data)
    # time-span
    tt = 0:60:(6.9*24*3600)

    sim = Dict()
    ids = sort_cell_ids(data)
    for id in ids
        df = data[id]
        ecm_model = ecms[id]
        gp_model = gpms[id]

        # real voltage
        profile = load_profile(df)
        df_test = sample_dataset(profile, tt)
        v̄ = df_test.v

        # ECM
        (; ecm, ode) = ecm_model
        ode = remake(ode; tspan=(tt[begin], tt[end]))
        sol = solve(ode, Tsit5(); saveat=tt)
        v_ecm = sol[ecm.v]
        δv_ecm = (v_ecm - v̄) * 1e3 # V -> mV

        # GP-ECM
        v_gp = simulate_gp_rc(gp_model, df_test)
        σv_gp = v_gp.σ * 1e3 # V -> mV
        δv_gp = (v_gp.μ - v̄) * 1e3 # V -> mV

        # output
        sim[id] = DataFrame(; t=df_test.t, v̄, δv_ecm, δv_gp, σv_gp)
    end

    return sim
end


function benchmark_sim(ecms, gpms, data)
    ids = sort_cell_ids(data)
    sim = simulation_voltage(ecms, gpms, data)

    # rmse
    map(ids) do id
        (; δv_ecm, δv_gp, σv_gp) = sim[id]

        rmse_ecm = sqrt(mean(abs2, δv_ecm))
        rmse_gpm = sqrt(mean(abs2, δv_gp))

        # q50_ecm = quantile(abs.(δv_ecm), 0.5)
        # q50_gpm = quantile(abs.(δv_gp), 0.5)

        q95_ecm = quantile(abs.(δv_ecm), 0.95)
        q95_gpm = quantile(abs.(δv_gp), 0.95)

        # q99_ecm = quantile(abs.(δv_ecm), 0.99)
        # q99_gpm = quantile(abs.(δv_gp), 0.99)

        # Skip the first 10 min of the dataset:
        # High temperatures resulted from quickly charging the cells to the initial SOC.
        # This high temperautes caused the high discrepancies with the model (until cooldown).
        # We ommit these so that we can capture other peak discrepancies (caused by OCV fitting).
        # Omiting these 10 min has a minimal impact on the RMSE or quantiles, they represent 0.1% of the dataset.
        max_ecm = maximum(abs.(δv_ecm[10:end]))
        max_gpm = maximum(abs.(δv_gp[10:end]))

        q95_2σ = quantile(2σv_gp, 0.95)
        max_2σ = maximum(2σv_gp)

        (;
            rmse_ecm, q95_ecm, max_ecm,
            rmse_gpm, q95_gpm, max_gpm,
            q95_2σ, max_2σ
        )
    end |> DataFrame
end
