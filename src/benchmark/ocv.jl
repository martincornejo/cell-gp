function analyze_ocv_gp(model, df)
    profile = load_profile(df)
    tt = 0:60:(24*2600)
    df_train = sample_dataset(profile, tt)

    # soh range
    cap = calc_capa_cccv(df)
    soc0 = initial_soc(df)
    cmin, cmax = extrema(df_train.s) # charge
    crange = cmin:0.01:cmax
    srange = (crange ./ cap) .+ soc0

    # real
    pocv = calc_pocv(df)
    ocv_meas = pocv(srange)

    # gp
    (; gp, dt) = model
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

function analyze_ocv_ecm(ecm, df; correct_soc=true)
    (; ode) = ecm
    profile = load_profile(df)
    tt = 0:60:(24*2600)
    df_train = sample_dataset(profile, tt)

    pocv = calc_pocv(df)
    focv = fresh_focv()

    cap_real = calc_capa_cccv(df)
    cap_sim = ode.p[1]

    soc0 = initial_soc(df)
    soc0_sim = ode.u0[1]
    Δsoc = correct_soc ? soc0_sim - soc0 : 0.0

    s = 0:0.001:1

    # OCV measured
    c1 = s .* cap_real
    ocv1 = pocv(s)
    ocv_real = ConstantInterpolation(ocv1, c1)

    # OCV estimated
    c2 = (s .- Δsoc) .* cap_sim
    ocv2 = focv(s)
    ocv_sim = ConstantInterpolation(ocv2, c2)

    # soc range
    smin, smax = extrema(df_train.s) .+ soc0 * cap_real
    v_real = ocv_real(smin:0.01:smax)
    v_sim = ocv_sim(smin:0.01:smax)

    e = v_real - v_sim
    mae = mean(abs, e) * 1e3
    max_e = maximum(abs, e) * 1e3

    return (; mae, max_e)
end

function benchmark_ocv(ecms, gpms, data)
    ids = sort_cell_ids(data)

    ecm_ocv = ids .|> id -> analyze_ocv_ecm(ecms[id], data[id])
    ecm_mae = ecm_ocv .|> first
    ecm_max = ecm_ocv .|> last

    gp_ocv = ids .|> id -> analyze_ocv_gp(gpms[id], data[id])
    gp_mae = gp_ocv .|> first
    gp_max = gp_ocv .|> last

    DataFrame(; id=ids, ecm_mae, ecm_max, gp_mae, gp_max)
end
