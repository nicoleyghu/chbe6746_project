# ChBE 6746 - Group 10 Project

N. Hu 2021/03/05

## Package AMPtorch installation instructions

1. In a designated folder, clone the active branch of AMPtorch from github in terminal. (Using macOS/Unix syntax here. Windows may have slightly different syntax for cmd or powershell.)
   
   Create a folder for the cloned package under root directory (or a specified directory): `mkdir ~/amptorch_MCSH_paper1; cd ~/amptorch_MCSH_paper1`

   Git clone the package with specific branch:`git clone --branch MCSH_paper1 https://github.com/nicoleyghu/amptorch.git`
2. Navigate into the cloned folder by `cd amptorch` and install the package following the instructions for AMPtorch installation as stated in `README.md`. 
   
   Update Conda: `conda update conda`

   Install environment for CPU machines: `conda env create -f env_cpu.yml`

   Activate the environment by: `conda activate amptorch_MCSH`

   Install the package with: `pip install -e .`

3. After Installing the package, clone the provided dataset and training python script through another github repository:
   
    Create a folder for the cloned package under root directory (or a specified directory): `mkdir ~/chbe6746_project; cd ~/chbe6746_project`

   Git clone the package with specific branch:`git clone https://github.com/nicoleyghu/chbe6746_project.git`

4. Go to the cloned folder and execute the `train.py` file to build the neural network potential for the dataset stored in `./data/`
   
   Move to the folder: `cd chbe6746_project`

   Make sure the virtual environment `amptorch_MCSH` is active. If not, `conda activate amptorch_MCSH`

   Run the training script by `python train.py`
