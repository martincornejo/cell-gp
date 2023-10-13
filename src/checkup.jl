
function line_cap(df, line)
    df_line = filter(:Line => ==(line), df)
    charge = df_line[:, :Ah_Step]
    charge .|> abs |> maximum
end

"Averaged capacity from the discharge of the first 2 check-up cycles (C/4)"
function reference_capacity(df::DataFrame)
    lines = [18, 19] # discharge CC and CV phase
    df = filter(:Line => ∈(lines), df)

    # separate 1st and 2nd cycles
    idx = df[:, :Line] |> diff .|> ==(-1) |> findfirst
    df1 = df[begin:idx, :]
    df2 = df[(idx+1):end, :]

    # Ah per CC/CV discharge
    discharge_1_cc = line_cap(df1, 18) # CC
    discharge_1_cv = line_cap(df1, 19) # CV
    discharge_1 = discharge_1_cc + discharge_1_cv
    discharge_2_cc = line_cap(df2, 18)
    discharge_2_cv = line_cap(df2, 19)
    discharge_2 = discharge_2_cc + discharge_2_cv

    return (discharge_1 + discharge_2) / 2
end


function reference_capacity(file)
    dateformat = dateformat"d-u-y H:M:S"
    df = CSV.File(file; dateformat) |> DataFrame
    reference_capacity(df)
end
