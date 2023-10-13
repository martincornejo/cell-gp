

function load_profile(file, ti)
    dateformat = dateformat"d-u-y H:M:S"
    df = CSV.File(file; dateformat) |> DataFrame

    t0, t1 = ti
    filter!(:Time => t -> t0 <= t <= t1, df)
    df[:, :t] = (df.Time .- df.Time[begin]) * 3600 # hours -> seconds

    i = ConstantInterpolation(df.I, df.t)
    v = ConstantInterpolation(df.U, df.t)
    s = ConstantInterpolation(df.Ah, df.t)
    T = ConstantInterpolation(df.T1, df.t)

    return (; i, s, v, T)
end

function sample_dataset(data, tt)
    return DataFrame(
        :t => tt,
        :i => data.i(tt),
        :v => data.v(tt),
        :s => data.s(tt),
        :T => data.T(tt)
    )
end
