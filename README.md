# Adversary-Aware-Concept-Embedding-Model
## How to Run the Code

To execute the code, please follow the detailed steps outlined below:

1. **Create the virtual environment:**
   First, you need to set up the virtual environment using the provided `environment.yml` file. This can be done by running the following command:
   ```sh
   conda env create -f environment.yml
   ```

2. **Activate the environment:**
   Once the virtual environment is created, you need to activate it. Use the command below to activate the environment named `v_cem`:
   ```sh
   conda activate v_cem
   ```

3. **Run the experiments:**
   With the environment activated, you can now run the experiments by executing the main script. Use the following command to start the experiments:
   ```sh
   python main.py
   ```

**Important Notes:** 
- We utilize the `wandb` logger to monitor and track the training process across different configurations. To successfully run the code, you must either connect to a `wandb` account or modify the code to remove the logger if you prefer not to use it.
- Additionally, we use Hydra to manage different configurations. To perform different experiments, you need to modify the `sweep.yaml` file located in the `config` folder.

