## define params that we need to iteratre through

lst_n_layers = [1, 2, 3]
lst_d_model = [6, 12]
lst_d_mlp = [lst_d_model[i] * 4 for i in range(len(lst_d_model))]
lst_n_heads = [1, 2, 3]
n_epoch = 1600
lst_checkpoint_every = 50


headerstr = "/usr/bin/python3 tsfm.py "

# create a list of all the possible combinations of the parameters

params_str = []

for n_layers in lst_n_layers:
    for i in range(len(lst_d_model)):
        d_model = lst_d_model[i]
        d_mlp = lst_d_mlp[i]

        for n_heads in lst_n_heads:
            str_param = headerstr
            str_param += f"--n_layers {n_layers} "
            str_param += f"--d_model {d_model} "
            str_param += f"--d_mlp {d_mlp} "
            str_param += f"--n_heads {n_heads} "
            str_param += f"--d_head {d_model // n_heads} "
            str_param += f"--n_epochs {n_epoch} "
            str_param += f"--checkpoint_every {lst_checkpoint_every}"
            str_param += "\n"

            params_str.append(str_param)

print(params_str)
# create the config file

import os

script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "config_file.txt")

# Open the file using the full path
with open(config_path, "w") as config_file:
    for param in params_str:
        config_file.write(param)

config_file.close()
