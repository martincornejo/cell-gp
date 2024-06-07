function simulate_pulse_resistance_ecm(model, focv; i=1.6166, soc=0.5, Δt=9.99)
    @unpack ode = model
    pulse(t) = i # constant current pulse
    @mtkbuild ecm = ECM(; focv, fi=pulse)
    prob = ODEProblem(ecm, [ecm.soc => soc], (0, Δt))
    sol = solve(prob, Tsit5(); p=ode.p, saveat=Δt)
    Δv = abs(sol[ecm.vr][end] + sol[ecm.v1][end] + sol[ecm.v2][end])
    return Δv / i * 1e3 # mΩ
end

function simulate_pulse_resistance_gpm(model, df; i=1.6166, soc=0.5, Δt=9.99)
    @unpack ode, gp, dt = model

    # RC
    pulse(t) = i # constant current pulse
    @mtkbuild rc = RC(; fi=pulse)
    prob = ODEProblem(rc, [], (0, Δt))
    sol = solve(prob, Tsit5(); p=ode.p, saveat=Δt)
    Δv = abs(sol[rc.v1][end] + sol[rc.v2][end])
    r_rc = Δv / i

    # rint from GP
    cap = calc_capa_cccv(df)
    soc0 = initial_soc(df)
    c = cap * (soc - soc0)
    ĉ = StatsBase.transform(dt.s, [c])
    x = GPPPInput(:r, RowVecs([ĉ zero.(ĉ)]))
    r = gp(x)
    rμ = mean(r) |> first
    rσ = sqrt.(var(r)) |> first

    r0 = dt.σ.scale[1] / dt.i.scale[1] # scale to Ω
    rint_μ = rμ * r0
    rint_σ = rσ * r0

    return ((rint_μ ± rint_σ) + r_rc) * 1e3 # mΩ
end

function r2(y, f)
    ȳ = mean(y)
    res = sum(abs2, y .- f)
    tot = sum(abs2, y .- ȳ)
    1 - (res / tot)
end

function benchmark_rdc(ecms, gpms, data)
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

    r_cup = df_cup[:, ids] |> Array
    r_ecm = df_ecm[:, ids] |> Array
    r_gpm = df_gpm[:, ids] |> Array

    r2_ecm = r2(r_cup, r_ecm)
    r2_gpm = r2(r_cup, r_gpm)

    (; r2_ecm, r2_gpm)
end

