function simulate_pulse_resistance(ode; i=3.233, soc=0.5, Δt=10)
    pulse(t) = i # constant current pulse
    focv = fresh_focv()
    @mtkbuild ecm = ECM(; focv, fi=pulse)
    tspan = (0, Δt)
    prob = ODEProblem(ecm, [ecm.v1 => 0.0, ecm.soc => soc], tspan)
    prob = remake(prob; p=ode.p)
    sol = solve(prob, Tsit5(); saveat=Δt)
    Δv = abs(sol[ecm.vr][end] + sol[ecm.v1][end])
    return Δv / i
end

function simulate_pulse_gp(model, df; i=3.233, soc=0.5, Δt=10)
    @unpack ode, gp, dt = model

    # RC
    pulse(t) = i # constant current pulse
    @mtkbuild rc = RC(; fi=pulse)
    tspan = (0, Δt)
    prob = ODEProblem(rc, [rc.v1 => 0.0], tspan)
    prob = remake(prob; p=ode.p)
    sol = solve(prob, Tsit5())
    Δv = abs(sol[rc.v1][end]) # TODO add 2nd RC
    r1 = Δv / i

    # rint from GP
    capa = calc_capa_cccv(df)
    c = capa * (soc - 0.385)
    ĉ = StatsBase.transform(dt.s, [c])
    x = GPPPInput(:r, RowVecs([ĉ zero.(ĉ)]))
    r = gp(x)
    rμ = mean(r) |> first
    rσ = sqrt.(var(r)) |> first

    r0 = dt.σ.scale[1] / dt.i.scale[1] # scale to Ω
    rint = rμ * r0

    return r1 + rint
end

function benchmark_rint(ecms, gpms; timestep=9.99, soc=5)
    r_sims = Float64[]
    r_mess = Float64[]
    r_errs = Float64[]

    r_sim = simulate_pulse_resistance(ode; Δt) * 1e3
    df = data[id]
    r_meas = calc_rint(df; timestep=Δt)[soc] * 1e3
    r_error = (r_sim - r_meas) / r_meas * 100
end

function analyze_rint_gp(gps, data)
    r_cu = Float64[]
    r_gp = Float64[]
    for id in ids
        df = data[id]
        r1 = calc_rint(df; timestep=9.99)[5] * 1e3

        gp = gps[id]
        r2 = simulate_pulse_gp(gp, df) * 1e3
        # r_error = (r_gp - r_meas) / r_meas * 100
        # @info id r_gp r_meas r_error
        push!(r_cu, r1)
        push!(r_gp, r2)
    end
    DataFrame(; ids, r_cu, r_gp)
end