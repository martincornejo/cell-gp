function analyze_ocv_gp(gp_model, df)
    profile = load_profile(df)
    tt = 0:60:(24*2600)
    df_train = sample_dataset(profile, tt)

    # soh range
    cap = calc_capa_cccv(df)
    cmin, cmax = extrema(df_train.s) # charge
    crange = cmin:0.01:cmax
    srange = (crange ./ cap) .+ 0.385

    # real
    pocv = calc_pocv(df)
    ocv_meas = pocv(srange)

    # gp
    @unpack gp, dt = gp_model
    ŝ = StatsBase.transform(dt.s, crange)
    i = zero.(ŝ)
    x = GPPPInput(:ocv, RowVecs([ŝ i]))
    ocv_gp = gp(x)
    ocvμ = StatsBase.reconstruct(dt.v, mean(ocv_gp))
    # ocvσ = StatsBase.reconstruct(dt.σ, sqrt.(var(ocv_gp)))

    # error
    e = ocvμ - ocv_meas
    mae = mean(abs, e) * 1e3
    max_e = maximum(abs, e) * 1e3
    return (; mae, max_e)
end

function analyze_ocv_ecm(ecm, df)
    profile = load_profile(df)
    tt = 0:60:(24*2600)
    df_train = sample_dataset(profile, tt)

    # function ocv(ecm, df)
    pocv = calc_pocv(df)
    focv = fresh_focv()
    s = 0:0.001:1

    cap_real = calc_capa_cccv(df)
    cap_sim = ecm.ode.p[1]

    soc_real = 0.385
    soc_sim = ecm.ode.u0[1]
    Δsoc = soc_sim - soc_real
    # Δsoc = 0.0

    # real
    c1 = s .* cap_real
    ocv1 = pocv(s)
    ocv_real = ConstantInterpolation(ocv1, c1)

    # estimated
    c2 = (s .- Δsoc) .* cap_sim
    ocv2 = focv(s)
    ocv_sim = ConstantInterpolation(ocv2, c2)

    # soh range
    cap = calc_capa_cccv(df)
    smin, smax = extrema(df_train.s) .+ 0.385 * cap
    v_real = ocv_real(smin:0.01:smax)
    v_sim = ocv_sim(smin:0.01:smax)

    e = v_real - v_sim
    mae = mean(abs, e) * 1e3
    max_e = maximum(abs, e) * 1e3

    return (; mae, max_e)
end