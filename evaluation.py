import wandb
import eval_class.model_eval as me
from datamodules.lucas_processing import ModelReadyDataset
import torch
import numpy as np


def evaluate_main(cfg, datamodule, eval_model):
    
    eval_dataset = ModelReadyDataset(
        shots=[datamodule.original_data[i] for i in datamodule.test_inds],
        inds = datamodule.test_inds,
        end_cutoff_timesteps=0,
        end_cutoff=False,
        machine_hyperparameters={
            "cmod": [.1, .9],
            "d3d": [.1, .9],
            "east": [.1, .9]
        },
        taus=cfg.dataset.params.taus,
        # max_length=cfg.dataset.max_length,
    )
        
    print("Evaluating model performance w the ENI class...")
    eval_high_thresh = cfg.test.eval_high_thresh
    eval_low_thresh = cfg.test.eval_low_thresh
    if eval_high_thresh < eval_low_thresh:
        eval_low_thresh = eval_high_thresh

    # will --  where is the model stored? 
    eval_dict = create_eval_dict(eval_dataset, eval_model=eval_model, master_dataset=datamodule.original_data)
    params, metrics = return_eval_params()

    params['high_thr'] = eval_high_thresh
    params['low_thr'] = eval_low_thresh
    params['t_hysteresis'] = cfg.test.eval_hysteresis

    performance_calculator = me.model_performance()

    val_metrics_report = performance_calculator.eval(
        unrolled_proba=eval_dict,
        metrics=metrics,
        params_dict=params,
    )

    print("Saving evaluation metrics to wandb...")
    wandb.log(val_metrics_report)

    return



def create_eval_dict(test_dataset, eval_model, master_dataset):
    """Save the unrolled probabilities into the eval class created by ENI.
    
    Args:
        test_dataset (AutoformerSeqtoLabelDataset): Test dataset.
        trainer (object): Trainer object from huggingface.
    """

    eval_dict = {}

    ind = 0

    for shot in test_dataset:
        # probs = get_probs_from_seq_to_lab_model(shot=shot, eval_model=trainer.model.eval().cpu())
        # disruptivity = utils.moving_average_with_buffer(probs[:, 1])

        shot_xs, labels, len = shot

        # transpose shot_xs so that the dimensions are (1, shot_xs.shape[1], shot_xs.shape[0])
        shot_xs = shot_xs.transpose(0,1).unsqueeze(0)
        
        # TODO: Will: generate unrollled predictions and put them in disruptivity
        disruptivity = eval_model.network.forward_unrolled(shot_xs)
  
        time = np.array(list(range(disruptivity.shape[1]))) * .005  # cmod is always the test machine
        label = labels[0][1] > .5 ## Will, the labels are the same value, intentional?
        time_until_disrupt = [np.nan] * disruptivity.shape[1]

        if label:
            time_until_disrupt = max(time) - time
        
        eval_dict[ind] = {
            "proba_shot": disruptivity.squeeze().detach().numpy(),
            "time_untill_disrupt": time_until_disrupt,
            "time_shot": time,
            "label_shot": label,                    
        }

        ind += 1 
    
    return eval_dict


def return_eval_params():
    # Necessary inputs
    params_dict = {
        'high_thr':.5,
        'low_thr':.5,
        't_hysteresis':0,
        't_useful':.03
        }

    metrics = [
        'f1_score', 
        'f2_score', 
        'recall_score', 
        'precision_score', 
        'roc_auc_score', 
        'accuracy_score', 
        'confusion_matrix', 
        'tpr', 
        'fpr', 
        'AUC_zhu']
    
    return params_dict, metrics