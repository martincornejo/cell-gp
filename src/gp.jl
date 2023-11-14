
## data pre-processing
function fit_zscore(df)
    # v = StatsBase.fit(ZScoreTransform, df.v)
    # σ = StatsBase.fit(ZScoreTransform, df.v, center=false)
    # i = StatsBase.fit(ZScoreTransform, df.i, center=false)
    # s = StatsBase.fit(ZScoreTransform, df.s, center=false)
    v = ZScoreTransform(1, 1, [3.7], [1.0])
    σ = ZScoreTransform(1, 1, Float64[], [1.0])
    i = ZScoreTransform(1, 1, Float64[], [4.9])
    s = ZScoreTransform(1, 1, Float64[], [4.9])
    return (; v, σ, i, s)
end

function normalize_data(df, dt)
    v̂ = StatsBase.transform(dt.v, df.v)
    î = StatsBase.transform(dt.i, df.i)
    ŝ = StatsBase.transform(dt.s, df.s)
    return DataFrame(; df.t, v̂, î, ŝ)
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

    sol = solve(prob, LBFGS(); show_trace=true)
    θ_opt = unflatten(sol.u)

    return θ_opt
end

function build_gp(model, θ, df, dt)
    # normalize data
    # dt = fit_zscore(df)
    dfs = normalize_data(df, dt)
    x = GPPPInput(:v, RowVecs([dfs.ŝ dfs.î]))
    y = dfs.v̂

    # build gp
    fp = model(θ.kernel)
    fx = fp(x, θ.noise)
    gp = posterior(fx, y)
    return gp, dt
end

function simulate_gp(gp_model, df)
    # normalize data
    gp, dt = gp_model
    dfs = normalize_data(df, dt)
    x = GPPPInput(:v, RowVecs([dfs.ŝ dfs.î]))

    # simulate
    y = gp(x)
    yμ = mean(y)
    yσ = sqrt.(var(y))

    # de-normalize results
    vμ = StatsBase.reconstruct(dt.v, yμ)
    vσ = StatsBase.reconstruct(dt.σ, yσ)
    return vμ, vσ
end


## GP+RC model
@mtkmodel RC begin
    begin
        @variables t
        D = Differential(t)
    end
    @parameters begin
        R1 = 1.0e-3
        τ1 = 50
    end
    @variables begin
        i(t)
        v1(t)
    end
    @structural_parameters begin
        fi
    end
    @equations begin
        D(v1) ~ -v1 / τ1 + i * (R1 / τ1) # check sign
        i ~ fi(t)
    end
end

function build_loss_rc(mtk_model, prob, fx, v̂, t, dt)
    return loss_rc(x) = begin
        @unpack u0, p = x
        newprob = remake(prob; u0, p)
        sol = solve(newprob; saveat=t) # TODO: rate as param
        vrc = sol[mtk_model.v1]
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

    sol = solve(prob, LBFGS(); show_trace=true)
    return sol.u
end

function fit_gp_rc(model, θ, data, tt)
    # load dataset
    df_train = sample_dataset(data, tt)
    dt = fit_zscore(df_train)

    # build rc model
    fi = t -> data.i(t)
    @mtkbuild rc = RC(; fi)

    tspan = (tt[begin], tt[end])
    ode = ODEProblem(rc, [rc.v1 => 0.0], tspan, [])

    u = fit_rc_params(rc, ode, model, θ, df_train, dt)

    ode = remake(ode; u...)
    sol = solve(ode, saveat=60) # todo
    vrc = sol[rc.v1]
    v̂rc = StatsBase.transform(dt.σ, vrc)

    # build GP model
    dfs = normalize_data(df_train, dt)
    x = GPPPInput(:v, RowVecs([dfs.ŝ dfs.î]))
    v̂ = dfs.v̂

    fp = model(θ.kernel)
    fx = fp(x, θ.noise)
    gp = posterior(fx, v̂ - v̂rc)

    return gp, ode, rc, dt
end

function simulate_gp_rc(model, df)
    gp, ode, rc, dt = model

    tspan = (df[begin, "t"], df[end, "t"])
    ode = remake(ode; tspan)
    sol = solve(ode, saveat=df.t)
    vrc = sol[rc.v1]

    # 
    dfs = normalize_data(df, dt)
    x = GPPPInput(:v, RowVecs([dfs.ŝ dfs.î]))
    y = gp(x)
    yμ = mean(y)
    yσ = sqrt.(var(y))

    vμ = StatsBase.reconstruct(dt.v, yμ) + vrc
    vσ = StatsBase.reconstruct(dt.σ, yσ)

    return vμ, vσ
end
