# graphical-model-capstone

To specify an experiment, create a new experiment directory in `experiment` and add a `config.json` file, which should look like: 


    { "DESCRIPTION": "TEMPLATE OF FILES GENERATED IN ONE EXPERIMENT" ,
    "SEED": 44,
    "N_REPS": [5,5,5,5,5],
    "N_NODES": [5,5,5,5,5],
    "N_DATA": [250,250,250,250,250],
    "PRIOR": "uniform",
    "PROPOSAL": "uniform",
    "LIKELIHOOD": "G_Wishart_Ratio" ,
    "ITER": [7500,7500,7500,7500,7500], 
    "BURNIN": [2500,2500,2500,2500,2500]
    }


`Param.pkl` and `Data.pkl` are lists of inputs to the MCMC model. If these are not specified in the experiment directory, the script will generate these file using `generator.py`.

Once the three files above are specified, user can run the MCMC specified by `experiment_dir` by calling 
    
    python experiment.py experiments/experiment_dir   

Check out an example given in `experiments/five_on_five.py`. 
