

function extract_ocv(file)
    dateformat = dateformat"d-u-y H:M:S"
    df = CSV.File(file; dateformat) |> DataFrame
    return extract_ocv(df)
end

function extract_ocv(df::DataFrame)
    ocv = Float64[]
    cap = Float64[]

    # define lines of the BaySyTec-Profile at which the cells were relaxed to obtain OCV-Values
    relaxation_lines = [16, 19, 23, 27, 29, 32, 36, 40, 42]

    # find the OCV values and the charge level at said lines
    for line in relaxation_lines
        df_line = filter(:Line => ==(line), df)
        cycles = df_line[:, :Cyc_Count] |> unique
        for cycle in cycles
            df_cycle = filter(:Cyc_Count => ==(cycle), df_line)
            append!(ocv, df_cycle[end, :U])
            append!(cap, df_cycle[end, :Ah])
        end
    end
    cap = cap .- minimum(cap) # set lowest charge status to 0Ah

    # seperate vectors into charge and discharge, create interpolation functions
    idx = findfirst(cap .== 0)

    ocv_ch = ocv[begin:idx] |> reverse
    cap_ch = cap[begin:idx] |> reverse
    charge = LinearInterpolation(ocv_ch, cap_ch)

    ocv_dch = ocv[idx:end]
    cap_dch = cap[idx:end]
    discharge = LinearInterpolation(ocv_dch, cap_dch)

    focv(s) = (charge(s) + discharge(s)) / 2

    return charge, discharge, focv
end


