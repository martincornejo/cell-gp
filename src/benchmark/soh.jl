function gp_soh(gp_model, df; v=(3.6, 3.85))
    profile = load_profile(df)
    tt = 0:60:(24*2600)
    df_train = sample_dataset(profile, tt)

    cap = calc_capa_cccv(df)
    cmin, cmax = extrema(df_train.s) # charge
    crange = cmin:0.01:cmax

    @unpack gp, dt = gp_model
    ŝ = StatsBase.transform(dt.s, crange)
    i = zero.(ŝ)
    x = GPPPInput(:ocv, RowVecs([ŝ i]))
    ocv_gp = gp(x)
    μ = StatsBase.reconstruct(dt.v, mean(ocv_gp))

    f = ConstantInterpolation(crange .+ 0.385 * cap, sort(μ))
    v1, v2 = v
    return f(v2) - f(v1)
end


function ecm_soh(ecm, df, focv; v=(3.6, 3.85))
    profile = load_profile(df)
    tt = 0:60:(24*2600)
    df_train = sample_dataset(profile, tt)

    cap = calc_capa_cccv(df)
    cmin, cmax = extrema(df_train.s) # charge
    crange = (cmin:0.01:cmax) .+ 0.385 * cap
    srange = crange ./ cap

    cap_ecm = ecm.ode.p[1]
    ocv = focv(srange)
    f = ConstantInterpolation(srange .* cap_ecm, ocv)
    v1, v2 = v
    return f(v2) - f(v1)
end

function analyze_soh(gps, ecms, data)
    v1 = 3.6
    v2 = 3.8

    focv = fresh_focv()
    invf = ConstantInterpolation((0:0.01:1) .* 4.9, focv(0:0.01:1))
    soh0 = invf(v2) - invf(v1)

    soh_cu = Float64[]
    soh_gp = Float64[]
    soh_ecm = Float64[]
    soh_ecm2 = Float64[]

    for id in ids
        gp = gps[id]
        df = data[id]
        ecm = ecms[id]
        soh = calc_capa_cccv(df) / 4.9 * 100
        soh2 = ecm.ode.p[1] / 4.9 * 100
        soh3 = gp_soh(gp, df; v=(v1, v2)) / soh0 * 100
        soh4 = ecm_soh(ecm, df, focv; v=(v1, v2)) / soh0 * 100

        push!(soh_cu, soh)
        push!(soh_ecm, soh2)
        push!(soh_gp, soh3)
        push!(soh_ecm2, soh4)
    end

    return DataFrame(; ids, soh_cu, soh_ecm, soh_gp, soh_ecm2)
end

