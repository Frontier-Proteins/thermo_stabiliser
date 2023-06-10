# %% [markdown]
# # Fixed Backbone design from LM
# 
# This notebook demonstrates the Fixed Backbone design task from the paper [Language models generalize beyond natural proteins
# ](https://www.biorxiv.org/content/10.1101/2022.12.21.521521v1).
# 
# Given an input structure as .pdb file, the LM is used iteratively in an MCMC optimization to find a sequence that folds to that structure
# 

# %%
# First install additional dependencies
# !pip install -r additional_requirements.txt


# %%
# Imports
import os
import time
import hydra
import py3Dmol
from lm_design import Designer

# Params
pdb_fn = os.getcwd() + '/2N2U.pdb'
# pdb_fn = os.getcwd() + '/1cqw.pdb'
# pdb_fn = os.getcwd() + '/1bli.pdb'
# pdb_fn = os.getcwd() + '/AF-P0A8Y3-F1-model_v4.pdb'

seed = 0  # Use different seeds to get different sequence designs for the same structure
TASK = "fixedbb"

# %%
# Load hydra config from config.yaml
with hydra.initialize_config_module(config_module="conf"):
    cfg = hydra.compose(
        config_name="config", 
        overrides=[
            f"task={TASK}", 
            f"seed={seed}", 
            f"pdb_fn={pdb_fn}", 
            # 'tasks.fixedbb.num_iter=100'  # DEBUG - use a smaller number of iterations
        ])

# %%
# Create a designer from configuration
des = Designer(cfg, pdb_fn)

# %%
import transformers
transformers.__version__

# %%

# Run the designer
start_time = time.time()
des.run_from_cfg()
print("finished after %s hours", (time.time() - start_time) / 3600)

# %%
print("Output seq:", des.output_seq)

# %%

# Fold output with ESMFold API
output_seq = des.output_seq
# Fold with api:
#  curl -X POST --data "GENGEIPLEIRATTGAEVDTRAVTAVEMTEGTLGIFRLPEEDYTALENFRYNRVAGENWKPASTVIYVGGTYARLCAYAPYNSVEFKNSSLKTEAGLTMQTYAAEKDMRFAVSGGDEVWKKTPTANFELKRAYARLVLSVVRDATYPNTCKITKAKIEAFTGNIITANTVDISTGTEGSGTQTPQYIHTVTTGLKDGFAIGLPQQTFSGGVVLTLTVDGMEYSVTIPANKLSTFVRGTKYIVSLAVKGGKLTLMSDKILIDKDWAEVQTGTGGSGDDYDTSFN" https://api.esmatlas.com/foldSequence/v1/pdb/
import requests
import json
url = 'https://api.esmatlas.com/foldSequence/v1/pdb/'
r = requests.post(url, data=output_seq)
output_struct = r.text



# %%
# Visualize output structure
view = py3Dmol.view(width=800, height=800)
view.addModel(output_struct, 'pdb')
view.setStyle({'cartoon': {'color': 'spectrum'}})
view.zoomTo()
view.show()

# %%
# Visualize wild type structure
wt_struct_file = pdb_fn
view = py3Dmol.view(width=800, height=800)
view.addModel(open(wt_struct_file).read(), 'pdb')
view.setStyle({'cartoon': {'color': 'spectrum'}})
view.zoomTo()
view.show()

# %%



