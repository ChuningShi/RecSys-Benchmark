# @Time   : 2020/10/6, 2022/7/18
# @Author : Shanlei Mu, Lei Wang
# @Email  : slmu@ruc.edu.cn, zxcptss@gmail.com

# UPDATE:
# @Time   : 2022/7/8, 2022/07/10, 2022/07/13, 2023/2/11
# @Author : Zhen Tian, Junjie Zhang, Gaowei Zhang
# @Email  : chenyuwuxinn@gmail.com, zjj001128@163.com, zgw15630559577@163.com

"""
recbole.quick_start
########################
"""
import logging
import sys
import torch.distributed as dist
from collections.abc import MutableMapping
from logging import getLogger

from ray import tune

from recbole.config import Config
from recbole.data import (
    create_dataset,
    data_preparation,
)
from recbole.data.transform import construct_transform
from recbole.utils import (
    init_logger,
    get_model,
    get_trainer,
    init_seed,
    set_color,
    get_flops,
    get_environment,
)

import pandas as pd
import os
import numpy as np


def save_interaction_data(dataset, data_dict, output_dir, file_name):
    """Save interaction data to CSV file.

    Args:
        dataset: The dataset object containing field mappings
        data_dict: Dictionary containing interaction data
        output_dir: Directory to save the CSV file
        file_name: Name of the output CSV file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract user_id, item_id, and timestamp
    interaction_data = {
        'user_id': data_dict['user_id'],
        'item_id': data_dict['item_id']
    }
    
    # Check if timestamp exists in the data
    if 'timestamp' in data_dict:
        interaction_data['timestamp'] = data_dict['timestamp']
    
    # Convert to pandas DataFrame and save to CSV
    df = pd.DataFrame(interaction_data)
    output_path = os.path.join(output_dir, file_name)
    df.to_csv(output_path, index=False)
    
    print(f"Saved interaction data to {output_path}")
    return output_path


def save_item_details(dataset, output_dir):
    """Save item details to CSV file without using dtype.

    Args:
        dataset: The dataset object containing item features
        output_dir: Directory to save the CSV file
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract item features directly as in your example
        item_feat = dataset.item_feat.numpy()
        item_ids = item_feat["item_id"]
        
        # Initialize data dictionary with item_id
        data = {'item_id': [int(id) for id in item_ids]}
        
        # Try to add movie_title if available
        if "movie_title" in item_feat and "movie_title" in dataset.field2id_token:
            title_ids = item_feat["movie_title"]
            title_tokens = dataset.field2id_token["movie_title"]
            data['movie_title'] = [str(title_tokens[int(tid)]) for tid in title_ids]
        
        # Try to add other fields that exist both in item_feat and field2id_token
        for field in item_feat:
            if field != "item_id" and field != "movie_title" and field in dataset.field2id_token:
                field_ids = item_feat[field]
                field_tokens = dataset.field2id_token[field]
                
                # Handle sequence fields (like genres)
                try:
                    field_values = []
                    for id_val in field_ids:
                        if hasattr(id_val, '__iter__') and not isinstance(id_val, str):
                            # It's a sequence of IDs
                            tokens = [str(field_tokens[int(id)]) for id in id_val if id > 0]
                            field_values.append(' '.join(tokens))
                        else:
                            # It's a single ID
                            field_values.append(str(field_tokens[int(id_val)]))
                    data[field] = field_values
                except Exception as e:
                    print(f"Could not process field {field}: {e}")
        
        # Convert to pandas DataFrame
        df = pd.DataFrame(data)
        
        # Save to CSV
        output_path = os.path.join(output_dir, 'item_details.csv')
        df.to_csv(output_path, index=False)
        
        print(f"Saved item details to {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error in save_item_details: {e}")
        import traceback
        traceback.print_exc()
        return None

def run(
    model,
    dataset,
    exp_name, 
    config_file_list=None,
    config_dict=None,
    saved=True,
    nproc=1,
    world_size=-1,
    ip="localhost",
    port="5678",
    group_offset=0,
):
    if nproc == 1 and world_size <= 0:
        res = run_recbole(
            model=model,
            dataset=dataset,
            exp_name=exp_name, 
            config_file_list=config_file_list,
            config_dict=config_dict,
            saved=saved,
        )
    else:
        if world_size == -1:
            world_size = nproc
        import torch.multiprocessing as mp

        # Refer to https://discuss.pytorch.org/t/problems-with-torch-multiprocess-spawn-and-simplequeue/69674/2
        # https://discuss.pytorch.org/t/return-from-mp-spawn/94302/2
        queue = mp.get_context("spawn").SimpleQueue()

        config_dict = config_dict or {}
        config_dict.update(
            {
                "world_size": world_size,
                "ip": ip,
                "port": port,
                "nproc": nproc,
                "offset": group_offset,
            }
        )
        kwargs = {
            "config_dict": config_dict,
            "queue": queue,
        }
        mp.spawn(
            run_recboles,
            args=(model, dataset, exp_name, config_file_list, kwargs),
            nprocs=nproc,
            join=True,
        )
        # Normally, there should be only one item in the queue
        res = None if queue.empty() else queue.get()
    return res


def run_recbole(
    model=None,
    dataset=None,
    exp_name=None, 
    config_file_list=None,
    config_dict=None,
    saved=True,
    queue=None,
):
    r"""A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str, optional): Model name. Defaults to ``None``.
        dataset (str, optional): Dataset name. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
        queue (torch.multiprocessing.Queue, optional): The queue used to pass the result to the main process. Defaults to ``None``.
    """
    # configurations initialization
    config = Config(
        model=model,
        dataset=dataset,
        exp_name=exp_name, 
        config_file_list=config_file_list,
        config_dict=config_dict,
    )
    init_seed(config["seed"], config["reproducibility"])
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(sys.argv)
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)
    logger.info("Saving interaction data and item details to CSV...")
    
    # Save item details
    output_dir = os.path.join("/data/user_data/chunings/RecSys-Benchmark/RecBole/data_split", config["dataset"], exp_name)
    item_csv_path = save_item_details(dataset, output_dir)
    logger.info(f"Item details saved to {item_csv_path}")
    
    # Save train interactions
    train_dict = train_data.dataset.inter_feat.numpy()
    train_csv_path = save_interaction_data(dataset, train_dict, output_dir, "train_interactions.csv")
    logger.info(f"Train interactions saved to {train_csv_path}")
    
    # Save validation interactions
    valid_dict = valid_data.dataset.inter_feat.numpy()
    valid_csv_path = save_interaction_data(dataset, valid_dict, output_dir, "valid_interactions.csv")
    logger.info(f"Validation interactions saved to {valid_csv_path}")
    
    # Save test interactions
    test_dict = test_data.dataset.inter_feat.numpy()
    test_csv_path = save_interaction_data(dataset, test_dict, output_dir, "test_interactions.csv")
    logger.info(f"Test interactions saved to {test_csv_path}")
    
    
    # model loading and initialization
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    logger.info(model)

    transform = construct_transform(config)
    flops = get_flops(model, dataset, config["device"], logger, transform)
    logger.info(set_color("FLOPs", "blue") + f": {flops}")

    # trainer loading and initialization
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    
    #trainer.resume_checkpoint('/data/group_data/cx_group/REC/checkpoints/s3rec_amzn-sports/S3Rec-amzn-sports-1.pth')
    
    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=saved, show_progress=config["show_progress"]
    )

    # model evaluation
    test_result = trainer.evaluate(
        test_data, load_best_model=saved, show_progress=config["show_progress"]
    )

    environment_tb = get_environment(config)
    logger.info(
        "The running environment of this training is as follows:\n"
        + environment_tb.draw()
    )

    logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
    logger.info(set_color("test result", "yellow") + f": {test_result}")

    result = {
        "best_valid_score": best_valid_score,
        "valid_score_bigger": config["valid_metric_bigger"],
        "best_valid_result": best_valid_result,
        "test_result": test_result,
    }

    if not config["single_spec"]:
        dist.destroy_process_group()

    if config["local_rank"] == 0 and queue is not None:
        queue.put(result)  # for multiprocessing, e.g., mp.spawn

    return result  # for the single process


def run_recboles(rank, *args):
    kwargs = args[-1]
    if not isinstance(kwargs, MutableMapping):
        raise ValueError(
            f"The last argument of run_recboles should be a dict, but got {type(kwargs)}"
        )
    kwargs["config_dict"] = kwargs.get("config_dict", {})
    kwargs["config_dict"]["local_rank"] = rank
    run_recbole(
        *args[:4],
        **kwargs,
    )


def objective_function(config_dict=None, config_file_list=None, saved=True):
    r"""The default objective_function used in HyperTuning

    Args:
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """

    config = Config(config_dict=config_dict, config_file_list=config_file_list)
    init_seed(config["seed"], config["reproducibility"])
    logger = getLogger()
    for hdlr in logger.handlers[:]:  # remove all old handlers
        logger.removeHandler(hdlr)
    init_logger(config)
    logging.basicConfig(level=logging.ERROR)
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    init_seed(config["seed"], config["reproducibility"])
    model_name = config["model"]
    model = get_model(model_name)(config, train_data._dataset).to(config["device"])
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, verbose=False, saved=saved
    )
    test_result = trainer.evaluate(test_data, load_best_model=saved)

    tune.report(**test_result)
    return {
        "model": model_name,
        "best_valid_score": best_valid_score,
        "valid_score_bigger": config["valid_metric_bigger"],
        "best_valid_result": best_valid_result,
        "test_result": test_result,
    }


def load_data_and_model(model_file):
    r"""Load filtered dataset, split dataloaders and saved model.

    Args:
        model_file (str): The path of saved model file.

    Returns:
        tuple:
            - config (Config): An instance object of Config, which record parameter information in :attr:`model_file`.
            - model (AbstractRecommender): The model load from :attr:`model_file`.
            - dataset (Dataset): The filtered dataset.
            - train_data (AbstractDataLoader): The dataloader for training.
            - valid_data (AbstractDataLoader): The dataloader for validation.
            - test_data (AbstractDataLoader): The dataloader for testing.
    """
    import torch

    checkpoint = torch.load(model_file)
    config = checkpoint["config"]
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    init_seed(config["seed"], config["reproducibility"])
    model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    model.load_state_dict(checkpoint["state_dict"])
    model.load_other_parameter(checkpoint.get("other_parameter"))

    return config, model, dataset, train_data, valid_data, test_data
