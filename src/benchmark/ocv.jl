
function benchmark_ocv(ecms, gpms, data)
    ids = sort_cell_ids(data)

    # params
    focv = fresh_focv(data)
    tt = 0:60:(24*3*3600)

    map(ids) do id
        # 
        df = data[id]
        ecm = ecms[id]
        gpm = gpms[id]

        # soc bounds
        profile = load_profile(df)
        df_train = sample_dataset(profile, tt)
        qmin, qmax = extrema(df_train.s) # charge

        # capa / soh range
        cap = calc_capa_cccv(df)
        soc0 = initial_soc(df)
        q = qmin:0.01:qmax
        s = (q ./ cap) .+ soc0

        # measured pOCV
        pocv = calc_pocv(df)
        ocv_meas = pocv(s)

        # gp
        (; gp, dt) = gpm
        ŝ = StatsBase.transform(dt.s, q)
        i = zero.(ŝ)
        x = GPPPInput(:ocv, RowVecs([ŝ i]))
        ocv_gp = gp(x)
        ocvμ = StatsBase.reconstruct(dt.v, mean(ocv_gp))
        δocv_gp = ocvμ - ocv_meas

        # ecm
        (; ode) = ecm
        cap_ecm = ode.p[1]
        δQ = cap / cap_ecm
        ocv_ecm = focv(s .* δQ)
        δocv_ecm = ocv_ecm - ocv_meas

        # return 
        mae_ecm = mean(abs, δocv_ecm) * 1e3
        max_ecm = maximum(abs, δocv_ecm) * 1e3

        mae_gpm = mean(abs, δocv_gp) * 1e3
        max_gpm = maximum(abs, δocv_gp) * 1e3

        (; id, mae_ecm, mae_gpm, max_ecm, max_gpm)
    end |> DataFrame
end
