function simulate_pulse_resistance_ecm(model, focv; i=3.233, soc=0.5, Δt=9.99)
    @unpack ode = model
    pulse(t) = i # constant current pulse
    @mtkbuild ecm = ECM(; focv, fi=pulse)
    prob = ODEProblem(ecm, [ecm.soc => soc], (0, Δt))
    sol = solve(prob, Tsit5(); p=ode.p, saveat=Δt)
    Δv = abs(sol[ecm.vr][end] + sol[ecm.v1][end] + sol[ecm.v2][end])
    return Δv / i
end

function simulate_pulse_resistance_gp(model, df; i=3.233, soc=0.5, Δt=9.99)
    @unpack ode, gp, dt = model

    # RC
    pulse(t) = i # constant current pulse
    @mtkbuild rc = RC(; fi=pulse)
    prob = ODEProblem(rc, [], (0, Δt))
    sol = solve(prob, Tsit5(); p=ode.p, saveat=Δt)
    Δv = abs(sol[rc.v1][end] + sol[ecm.v2][end])
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

    return (rint_μ ± rint_σ) + r_rc
end

function benchmark_rint(ecms, gpms, data)
    focv = fresh_focv()
    sohs = [calc_capa_cccv(df) / 4.9 for df in values(data)]
    ids = collect(keys(data))[sortperm(sohs)] # sort cell-id by soh

    r_ecm = ids .|> id -> simulate_pulse_resistance_ecm(ecms[id], focv) * 1e3
    r_gp = ids .|> id -> simulate_pulse_resistance_gp(gpms[id], data[id]) * 1e3
    r_cu = ids .|> id -> calc_rint(data[id])[5] * 1e3
    DataFrame(; id=ids, r_cu, r_ecm, r_gp) # mΩ
end

