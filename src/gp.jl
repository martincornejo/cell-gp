
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
    T̂ = StatsBase.transform(dt.T, df.T)
    return DataFrame(; df.t, v̂, î, ŝ, T̂)
end

## GP-ECM model
# RCM model
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

# GP model
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

function build_nlml(rc_model, gp_model, x, v̂, t, dt)
    return loss(ϑ) = begin
        @unpack ode, rc = rc_model
        @unpack u0, p = ϑ.rc

        # RC
        prob = remake(ode; u0, p)
        sol = solve(prob, Tsit5(); saveat=t)
        vrc = sol[rc.v1] + sol[rc.v2]
        v̂rc = StatsBase.transform(dt.σ, vrc) # normalized RC-voltage drop

        # GP
        θ = softplus.(ϑ.gp) # transform to ensure parameters are positive
        fp = gp_model(θ.kernel)
        fx = fp(x, θ.noise) # finite gp
        -logpdf(fx, v̂ - v̂rc)
    end
end

function fit_gp_ecm_params(rc_model, gp_model, θ0, df, dt)
    x = GPPPInput(:v, RowVecs([df.ŝ df.î]))

    loss = build_nlml(rc_model, gp_model, x, df.v̂, df.t, dt)
    p0 = ComponentArray((;
        rc=(;
            u0=rc_model.ode.u0,
            p=rc_model.ode.p
        ),
        gp=θ0
    ))
    p0.gp = invsoftplus.(p0.gp) # transform inital parameters (to later ensure they are positive with softplus)

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

function fit_gp_ecm(gp_model, θ0, df, tt)
    # load dataset
    profile = load_profile(df)
    df_train = sample_dataset(profile, tt)
    dt = fit_zscore(df_train)
    dfn = normalize_data(df_train, dt)

    # fit rc params
    fi = t -> profile.i(t)
    @mtkbuild rc = RC(; fi)
    tspan = (tt[begin], tt[end])
    ode = ODEProblem(rc, [], tspan, [])
    rc_model = (; ode, rc)

    # fit gp-ecm hyperparams
    u = fit_gp_ecm_params(rc_model, gp_model, θ0, dfn, dt)

    # build rc model
    ode = remake(ode; u.rc...)
    sol = solve(ode, Tsit5(); saveat=tt)
    vrc = sol[rc.v1] + sol[rc.v2]

    # build GP model
    df_train[!, :v] = df_train.v - vrc
    dfn = normalize_data(df_train, dt)

    x = GPPPInput(:v, RowVecs([dfn.ŝ dfn.î]))
    v̂ = dfn.v̂

    θ = softplus.(u.gp)
    fp = model(θ.kernel)
    fx = fp(x, θ.noise)
    gp = posterior(fx, v̂)

    return (; gp, ode, rc, dt, θ)
end

function simulate_gp_rc(model, df)
    @unpack gp, ode, rc, dt = model

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


###

function fit_gp_series(data)
    tt = 0:60:(3*3600*24)
    θ0 = (;
        kernel=(;
            ocv=(; σ=1.0, l=1.0),
            r=(; σ=1.0, l=1.0)
        ),
        noise=0.001
    )

    res = Dict()
    Threads.@threads for id in collect(eachindex(data))
        df = data[id]
        try
            gp = fit_gp_ecm(model, θ0, df, tt)
            res[id] = gp
            @info "$id GP-ECM fit complete."
        catch e
            if isa(e, PosDefException)
                @warn "$id encountered a numerical instability while fitting. The model will be ommited."
            end
        end
    end
    return res
end
