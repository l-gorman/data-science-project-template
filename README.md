# Model Data Science Project

## Tutorial Notes

To run the tutorial environment

* [Pixi tutorial](https://pixi.sh/latest/tutorials/multi_environment/#glossary)

## Development Environment

This project includes a development container configuration. To use it:

1. Install [Visual Studio Code](https://code.visualstudio.com/).
2. Install the [Remote - Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension.
3. Open the project in VS Code and select "Reopen in Container" when prompted.


## Creating environment from scratch

1. Install pixi (see [here](https://pixi.sh/latest/))
2. Init project (usually as pyproject):
    ```bash
    pixi init --format pyproject
    ```

3. Add platforms you may think are relevant, by modifying pyproject.toml:
    ```toml
    [tool.pixi.workspace]
    channels = ["conda-forge"]
    platforms = ["osx-arm64", "linux-64", "win-64", "linux-64"]
    ```

3. Initialise environment with packages you need for main project:
    ```bash
    pixi add python>3.10 pymc>5 pandas arviz numpy ipykernel
    pixi add --feature dev-feat pytest black pylint jupyterlab
    ```
4. Add any additional packages you need for development:
    ```bash
    pixi add --feature dev-feat pytest black pylint
    pixi workspace environment add dev-env --feature dev-feat
    pixi task add test --feature dev-feat pytest
    pixi task add lint --feature dev-feat pylint
    pixi task add format --feature dev-feat black
    pixi task add jupyterlab --feature dev-feat jupyterlab
    ``` 





## Thanks

Thank you to @jatonline, the use of just, docker... for making this project run more smoothly.
Test changes to file in container