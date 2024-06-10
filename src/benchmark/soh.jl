
function ocv_capa_gp(gp_model, df; v=(3.6, 3.8))
    profile = load_profile(df)
    tt = 0:60:(24*2600)
    df_train = sample_dataset(profile, tt)

    cap = calc_capa_cccv(df)
    cmin, cmax = extrema(df_train.s) # charge
    crange = cmin:0.001:cmax
    soc0 = initial_soc(df)
    s = crange .+ soc0 * cap
    v1, v2 = v

    (; gp, dt) = gp_model
    ŝ = StatsBase.transform(dt.s, crange)
    i = zero.(ŝ)
    x = GPPPInput(:ocv, RowVecs([ŝ i]))
    ocv_gp = gp(x)
    μ = StatsBase.reconstruct(dt.v, mean(ocv_gp))
    σ = StatsBase.reconstruct(dt.σ, sqrt.(var(ocv_gp)))

    c1μ = s[findfirst(>=(v1), μ)]
    c1σ = c1μ - s[findfirst(>=(v1), μ + σ)]
    c1 = c1μ ± c1σ

    c2μ = s[findfirst(>=(v2), μ)]
    c2σ = s[findfirst(>=(v2), μ - σ)] - c2μ
    c2 = c2μ ± c2σ

    return c2 - c1
end

function ocv_capa_ecm(ecm, df, focv; v=(3.6, 3.8))
    profile = load_profile(df)
    tt = 0:60:(24*2600)
    df_train = sample_dataset(profile, tt)

    cap = calc_capa_cccv(df)
    soc0 = initial_soc(df)
    cmin, cmax = extrema(df_train.s) # charge
    crange = (cmin:0.001:cmax) .+ soc0 * cap
    srange = crange ./ cap

    cap_ecm = ecm.ode.p[1]
    ocv = focv(srange)
    f = ConstantInterpolation(srange .* cap_ecm, ocv)
    v1, v2 = v
    return f(v2) - f(v1)
end

function benchmark_soh(ecms, gpms, data; v=(3.6, 3.8))
    v1, v2 = v
    focv = fresh_focv(data)
    invf = ConstantInterpolation((0:0.01:1) .* 4.9, focv(0:0.01:1))
    soh0 = invf(v2) - invf(v1)

    ids = sort_cell_ids(data)
    soh_cu = ids .|> id -> calc_capa_cccv(data[id]) / 4.9
    soh_ocv_gp = ids .|> id -> ocv_capa_gp(gpms[id], data[id]; v) / soh0
    soh_ocv_ecm = ids .|> id -> ocv_capa_ecm(ecms[id], data[id], focv; v) / soh0
    soh_ecm = ids .|> id -> ecms[id].ode.p[1] / 4.9

    return DataFrame(; id=ids, soh_cu, soh_ocv_gp, soh_ocv_ecm, soh_ecm)
end

