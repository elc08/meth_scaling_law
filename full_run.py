import os
# PROCESSING DATA
# PROCESSING MAMMALIAN CONSORTIUM DATA
base_path = os.getcwd()
os.chdir(base_path + '/preprocessing/')

with open("pan_mammal.py") as f:
    exec(f.read())

with open("chimps.py") as f:
    exec(f.read())

os.chdir(base_path + '/scripts/')


# PROCESS SKIN DATA
tissue = 'Skin'
n_cores = 7

with open("pca.py") as f:
    exec(f.read())

with open("r2_sensitivity_opt.py") as f:
    exec(f.read())

with open("r2_sensitivity_random_opt.py") as f:
    exec(f.read())

# PROCESS BLOOD DATA

tissue = 'Blood'

with open("pca.py") as f:
    exec(f.read())

with open("r2_sensitivity_opt.py") as f:
    exec(f.read())

with open("r2_sensitivity_random_opt.py") as f:
    exec(f.read())

with open("basic_random.py") as f:
    exec(f.read())
