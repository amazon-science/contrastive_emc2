import wandb

def wandb_init(conf, name=None):
    # start a new wandb run to track this script
    project_name = "dist_contrastive"
    wandb.init(
        # set the wandb project where this run will be logged
        project=project_name,
        name=name,
        # track hyperparameters and run metadata
        config={
            **conf,
        }
    )