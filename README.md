# Data-driven model parametrization of late-life lithium-ion batteries
Accompanying code to the paper *Data-driven model parametrization of late-life lithium-ion batteries*.

## Dataset
To get started, you will need to download the dataset used in this research. You can download the dataset from the following link:

[https://zenodo.org/doi/10.5281/zenodo.13353324](https://zenodo.org/doi/10.5281/zenodo.13353324)

Once downloaded, please place the dataset in a local directory named `data/` within the root of this repository.

## Setup
1. **Install Julia**: To run the code, you will need to have Julia installed on your machine. We recommend using `juliaup` for easy installation and management of Julia versions: https://julialang.org/downloads/

2. **Create the project environment**: Navigate to the project directory and start Julia by typing `julia` in your terminal. Activate the environment and install all  the required packages by running:
    ```julia
    julia> using Pkg
    julia> Pkg.activate(".")
    julia> Pkg.instantiate()
    ```

4. **Run the code**: The entry point of the code is the file `src/main.jl`. To run the code, execute the following command in your REPL:
    ```
    julia> include("src/main.jl")
    ```
This will execute the script and produce the results as described in the paper.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
