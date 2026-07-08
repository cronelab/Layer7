# Layer7
A repository for processing data from Precision Neuroscience 


This repo is set up to use Astal's uv to manage project dependencies. 

To install uv on Linux / MacOs: 

curl -LsSf https://astral.sh/uv/install.sh | sh

To install uv on Windows: 
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

NOTE: The pyproject.toml is currently set up assuming your machine is using CUDA 12.8. To use different version, check the comments in the file