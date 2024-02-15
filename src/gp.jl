
## data pre-processing
function fit_zscore(df)
    v = StatsBase.fit(ZScoreTransform, df.v)
    σ = StatsBase.fit(ZScoreTransform, df.v, center=false)
    i = StatsBase.fit(ZScoreTransform, df.i, center=false)
    s = StatsBase.fit(ZScoreTransform, df.s)
    T = StatsBase.fit(ZScoreTransform, df.T)
    return (; v, σ, i, s, T)
end

function normalize_data(df, dt)
    v̂ = StatsBase.transform(dt.v, df.v)
    î = StatsBase.transform(dt.i, df.i)
    ŝ = StatsBase.transform(dt.s, df.s)
    T̂ = StatsBase.transform(dt.s, df.s)
    return DataFrame(; df.t, v̂, î, ŝ, T̂)
end

## GP model
function build_gp_loss(f, x, y)
    function loss(θ)
        gp = f(θ.kernel)
        fx = gp(x, θ.noise) # finite gp
        return -logpdf(fx, y)
    end
end

function fit_gp_hyperparams(model, θ, df)
    dt = fit_zscore(df)
    dfs = normalize_data(df, dt)
    x = GPPPInput(:v, RowVecs([dfs.ŝ dfs.î]))
    y = dfs.v̂

    loss = build_gp_loss(model, x, y)
    θ0, unflatten = ParameterHandling.value_flatten(θ)
    loss_packed = loss ∘ unflatten

    adtype = AutoZygote()
    f = OptimizationFunction((u, p) -> loss_packed(u), adtype)
    prob = OptimizationProblem(f, θ0)

    sol = solve(prob, LBFGS(); reltol=1e-3)
    θ_opt = unflatten(sol.u)

    return θ_opt
end

# function build_gp(model, θ, df, dt)
#     # normalize data
#     dfs = normalize_data(df, dt)
#     x = GPPPInput(:v, RowVecs([dfs.ŝ dfs.î]))
#     y = dfs.v̂

#     # build gp
#     fp = model(θ.kernel)
#     fx = fp(x, θ.noise)
#     gp = posterior(fx, y)
#     return gp, dt
# end

# function simulate_gp(gp_model, df)
#     # normalize data
#     gp, dt = gp_model
#     dfs = normalize_data(df, dt)
#     x = GPPPInput(:v, RowVecs([dfs.ŝ dfs.î]))

#     # simulate
#     y = gp(x)
#     yμ = mean(y)
#     yσ = sqrt.(var(y))

#     # de-normalize results
#     vμ = StatsBase.reconstruct(dt.v, yμ)
#     vσ = StatsBase.reconstruct(dt.σ, yσ)
#     return vμ, vσ
# end


## GP+RC model
@mtkmodel RC begin
    begin
        @variables t
        D = Differential(t)
    end
    @parameters begin
        R1 = 0.5e-3
        τ1 = 60
        R2 = 0.05e-3
        τ2 = 3600
    end
    @variables begin
        i(t)
        v1(t) = 0.0
        v2(t) = 0.0
    end
    @structural_parameters begin
        fi
    end
    @equations begin
        D(v1) ~ -v1 / τ1 + i * (R1 / τ1) # check sign
        D(v2) ~ -v2 / τ2 + i * (R2 / τ2) # check sign
        i ~ fi(t)
    end
end

function build_loss_rc(mtk_model, prob, fx, v̂, t, dt)
    return loss_rc(x) = begin
        @unpack u0, p = x
        newprob = remake(prob; u0, p)
        sol = solve(newprob, Tsit5(); saveat=t) # TODO: rate as param
        vrc = sol[mtk_model.v1] + sol[mtk_model.v2]
        v̂rc = StatsBase.transform(dt.σ, vrc)
        -logpdf(fx, v̂ - v̂rc)
    end
end

function fit_rc_params(rc_model, rc_ode, gp_model, gp_θ, df, dt)
    dfs = normalize_data(df, dt)
    x = GPPPInput(:v, RowVecs([dfs.ŝ dfs.î]))
    v̂ = dfs.v̂

    fp = gp_model(gp_θ.kernel)
    fx = fp(x, gp_θ.noise)

    loss = build_loss_rc(rc_model, rc_ode, fx, v̂, df.t, dt)
    p0 = ComponentArray((;
        u0=rc_ode.u0,
        p=rc_ode.p
    ))

    adtype = AutoForwardDiff()
    f = OptimizationFunction((u, p) -> loss(u), adtype)
    prob = OptimizationProblem(f, p0)
    alg = LBFGS(;
        alphaguess=Optim.LineSearches.InitialStatic(; alpha=10),
        linesearch=Optim.LineSearches.BackTracking(),
    )
    sol = solve(prob, alg; reltol=1e-4)
    return sol.u
end

function fit_gp_rc(model, θ0, profile, tt)
    # load dataset
    df_train = sample_dataset(profile, tt)
    dt = fit_zscore(df_train)

    # fit gp hyperparams
    θ = fit_gp_hyperparams(model, θ0, df_train)

    # fit rc params
    fi = t -> profile.i(t)
    @mtkbuild rc = RC(; fi)

    tspan = (tt[begin], tt[end])
    ode = ODEProblem(rc, [], tspan, [])

    u = fit_rc_params(rc, ode, model, θ, df_train, dt)

    # build rc model
    ode = remake(ode; u...)
    sol = solve(ode, Tsit5(); saveat=tt)
    vrc = sol[rc.v1] + sol[rc.v2]
    v̂rc = StatsBase.transform(dt.σ, vrc)

    # build GP model
    dfs = normalize_data(df_train, dt)
    x = GPPPInput(:v, RowVecs([dfs.ŝ dfs.î]))
    v̂ = dfs.v̂

    fp = model(θ.kernel)
    fx = fp(x, θ.noise)
    gp = posterior(fx, v̂ - v̂rc)

    return (; gp, ode, rc, dt, θ)
end

function simulate_gp_rc(model, df)
    gp, ode, rc, dt = model

    tspan = (df[begin, "t"], df[end, "t"])
    ode = remake(ode; tspan)
    sol = solve(ode, Tsit5(); saveat=df.t)
    vrc = sol[rc.v1] + sol[rc.v2]

    # 
    dfs = normalize_data(df, dt)
    x = GPPPInput(:v, RowVecs([dfs.ŝ dfs.î]))
    y = gp(x)
    yμ = mean(y)
    yσ = sqrt.(var(y))

    μ = StatsBase.reconstruct(dt.v, yμ) + vrc
    σ = StatsBase.reconstruct(dt.σ, yσ)

    return (; μ, σ)
end


#####
"SOC dependent R"
function model(θ)
    i = x -> x[2]
    return @gppp let
        ocv = θ.ocv.σ * GP(with_lengthscale(SEKernel(), θ.ocv.l) ∘ SelectTransform(1))
        r = θ.r.σ * GP(with_lengthscale(SEKernel(), θ.r.l) ∘ SelectTransform(1))
        vr = r * i
        v = ocv + vr
    end
end

function fit_gp_series(data)
    tt = 0:60:(3600*24)
    θ0 = (;
        kernel=(;
            ocv=(;
                σ=positive(1.0),
                l=positive(1.0)
            ),
            r=(;
                σ=positive(1.0),
                l=positive(1.0)
            )
        ),
        noise=positive(0.001)
    )

    res = Dict()
    Threads.@threads for id in collect(eachindex(data))
        df = data[id]
        profile = load_profile(df)
        gp = fit_gp_rc(model, θ0, profile, tt)
        res[id] = gp
    end
    return res
end
