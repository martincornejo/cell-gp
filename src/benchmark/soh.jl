
function benchmark_soh(ecms, gpms, data; v=(3.6, 3.9))
    focv = fresh_focv(data)
    tt = 0:60:(24*2600)
    s = 0.2:0.001:0.8

    ids = sort_cell_ids(data)
    res = map(ids) do id
        df = data[id]
        ecm = ecms[id]
        gpm = gpms[id]

        profile = load_profile(df)
        df_train = sample_dataset(profile, tt)

        v1, v2 = v
        cap = calc_capa_cccv(df)
        soc0 = initial_soc(df)
        q = (s .- soc0) .* cap

        # measured
        pocv = calc_pocv(df)
        ocv = pocv(s)
        q1 = q[findfirst(>=(v1), ocv)]
        q2 = q[findfirst(>=(v2), ocv)]
        Δq = q2 - q1

        # ecm
        (; ode) = ecm
        cap_ecm = ode.p[1]
        δQ = cap / cap_ecm
        ocv_ecm = focv(s .* δQ)
        q1_ecm = q[findfirst(>=(v1), ocv_ecm)]
        q2_ecm = q[findfirst(>=(v2), ocv_ecm)]
        Δq_ecm = q2_ecm - q1_ecm

        # gp
        (; gp, dt) = gpm
        ŝ = StatsBase.transform(dt.q, q)
        i = zero.(ŝ)
        x = GPPPInput(:ocv, RowVecs([ŝ i]))
        ocv_gp = gp(x)
        μ = StatsBase.reconstruct(dt.v, mean(ocv_gp))
        σ = StatsBase.reconstruct(dt.σ, sqrt.(var(ocv_gp)))

        q1μ = q[findfirst(>=(v1), μ)]
        q1σ = q1μ - q[findfirst(>=(v1), μ + σ)]
        q1_gpm = q1μ ± q1σ

        q2μ = q[findfirst(>=(v2), μ)]
        q2σ = q[findfirst(>=(v2), μ - σ)] - q2μ
        q2_gpm = q2μ ± q2σ

        Δq_gpm = q2_gpm - q1_gpm

        # results
        (; id, Δq, Δq_ecm, Δq_gpm)
    end |> DataFrame


    # SOH
    soh = [calc_capa_cccv(data[id]) / 4.8 for id in ids]
    res[!, :soh] = soh .* 100
    res[!, :Δsoh_ecm] = ([ecms[id].ode.p[1] / 4.8 for id in ids] - soh) .* 100

    # OCV based SOH
    Δq0 = res[1, :Δq]
    res[!, :soh´] = res[:, :Δq] ./ Δq0 .* 100
    res[!, :Δsoh´_ecm] = (res[:, :Δq_ecm] - res[:, :Δq]) ./ Δq0 .* 100
    res[!, :Δsoh´_gpm] = (res[:, :Δq_gpm] - res[:, :Δq]) ./ Δq0 .* 100

    # 
    select!(res, [:id, :soh, :Δsoh_ecm, :soh´, :Δsoh´_ecm, :Δsoh´_gpm])
    return res
end
