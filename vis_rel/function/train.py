# all the imports go here
# -------------------------------
# Torch Related Imports
# -------------------------------
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import os
import time
import wandb
import copy
import pprint
import numpy as np
from vis_rel.modules.frcnn_classifier import Net
from vis_rel.data.dataloader import DatasetLoader
import torch.distributed as dist
# other imports for distributed training
try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as Apex_DDP
except ImportError:
    pass


# the train engine here

"""/
def train_engine(args, config):
    wandb.init(project="intern", name=config.VERSION, config=config)

    # print the args and the config
    pprint.pprint(args)
    pprint.pprint(config)

    # manually set random seed
    if config.RNG_SEED > -1:
        np.random.seed(config.RNG_SEED)
        torch.random.manual_seed(config.RNG_SEED)

    # cudnn
    torch.backends.cudnn.benchmark = True
    if args.cudnn_off:
        torch.backends.cudnn.enabled = False

    # Choose device - CPU
    device = torch.device("cpu")  # Make sure to use CPU

    # If distributed, initialize the model but for CPU
    # If distributed training, initialize process group and model
    if args.dist:
        #dist.init_process_group(backend='gloo', init_method='tcp://localhost:12345', world_size=1, rank=0)
        model = Net(config)
        model = model.to(device)  # Ensure model is on CPU
        total_gpus = 1  # Use 1 device (CPU)

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(),
                                lr=5 * 1e-5,
                                betas=(0.9, 0.999),
                                eps=1e-6,
                                weight_decay=1e-4,
                                )
    else:
        # Using single CPU - No need for GPU setup
        model = eval(config.MODULE)(config)
        model = model.to(device)  # Ensure model is on CPU
        total_gpus = 1  # Only using 1 device (CPU)

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(),
                                lr=2 * 1e-6,
                                betas=(0.9, 0.999),
                                eps=1e-6,
                                weight_decay=1e-4,
                                )

    # Log all layers in the model using wandb
    wandb.watch(model, log='all')

    # Apex: AMP fp16 mixed-precision training (not needed without GPU)
    if config.TRAIN.FP16:
        # Skip AMP initialization since no GPU is used
        pass

    # Create the dataloaders
    train_dataset = DatasetLoader(config)
    data_size = len(train_dataset)

    # eval after every epoch either on the test or validation set
    if config.RUN_MODE == 'train+val':
        config_test = copy.deepcopy(config)
        config_test.RUN_MODE = 'test'
        val_dataset = DatasetLoader(config_test)
        val_data_size = len(val_dataset)

    elif config.EVAL_EVERY_EPOCH:
        config_val = copy.deepcopy(config)
        config_val.RUN_MODE = 'val'
        val_dataset = DatasetLoader(config_val)
        val_data_size = len(val_dataset)

    # If distributed training, create a train sampler
    if args.dist:
        train_sampler = Data.distributed.DistributedSampler(train_dataset)
        if config.EVAL_EVERY_EPOCH:
            val_sampler = Data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = None  # No distributed sampler for single CPU usage
        val_sampler = None  # No distributed sampler for single CPU usage

    # Create DataLoader for training and validation
    train_dataloader = Data.DataLoader(
        train_dataset,
        config.BATCH_SIZE,
        shuffle=(train_sampler is None),
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        sampler=train_sampler  # Use sampler for distributed training, None for CPU
    )

    val_dataloader = Data.DataLoader(
        val_dataset,
        config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        sampler=val_sampler  # Use sampler for distributed training, None for CPU
    )

    # Training the net
    for epoch in range(config.MAX_EPOCH):

        time_start = time.time()

        # iterations
        for step, (inputs, labels) in enumerate(train_dataloader):

            optimizer.zero_grad()

            # Make the tensors for CPU
            for key in inputs.keys():
                inputs[key] = inputs[key].to(device)  # Ensure inputs are on CPU

            labels = labels.to(device)  # Ensure labels are on CPU

            # Obtain the predictions
            pred = model(inputs)

            # Loss calculation
            loss = criterion(pred, labels)

            # Backpropagation
            loss.backward()

            # Optimize
            optimizer.step()

            # Print progress
            print("\r[Epoch: %2d][Step: %d/%d][Loss: %.4f]" % (
                epoch + 1,
                step,
                int(data_size / config.BATCH_SIZE),
                loss
            ), end='    ')

            # Log loss to wandb
            wandb.log({'Loss': loss})

        time_end = time.time()
        time_taken = time_end - time_start
        print('Finished in {}s'.format(int(time_taken)))

        # Save the model checkpoints
        state = {
            'state_dict': model.state_dict(),
            'epoch': epoch + 1
        }

        torch.save(
            state,
            config.OUTPUT_PATH +
            '/ckpt_' + config.VERSION +
            '_epoch' + str(epoch + 1) +
            '.pkl'
        )

        # Calling the validation function
        if val_dataset is not None:
            validate(
                config,
                val_dataloader,
                val_data_size,
                model,
                local_rank=None  # No need for local rank when using CPU
            )
/"""

def train_engine(args, config):
    wandb.init(project="intern", name=config.VERSION, config=config)

    # print the args and the config
    pprint.pprint(args)
    pprint.pprint(config)

    # manually set random seed
    if config.RNG_SEED > -1:
        np.random.seed(config.RNG_SEED)
        torch.random.manual_seed(config.RNG_SEED)

    # cudnn
    torch.backends.cudnn.benchmark = True
    if args.cudnn_off:
        torch.backends.cudnn.enabled = False

    # Choose device - CPU
    device = torch.device("cpu")  # Ensure only CPU is used

    # Initialize the model (no distributed training)
    #model = eval(config.MODULE)(config)  # Use the module specified in config
    try:
        print(f"Attempting to evaluate: {config.MODULE}")
        model = eval(config.MODULE)(config)
    except SyntaxError as e:
        print(f"Syntax error in {config.MODULE}: {e}")
    except Exception as e:
        print(f"Error while evaluating module: {e}")
        model = None
    try:
        from vis_rel.modules.frcnn_classifier import Net  # Kiểm tra xem module có import được không
        model = Net(config)  # Thử khởi tạo model thủ công
    except Exception as e:
        print(f"Error importing or initializing model: {e}")
    model = model.to(device)  # Ensure the model is on CPU

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(),
                            lr=2 * 1e-6,
                            betas=(0.9, 0.999),
                            eps=1e-6,
                            weight_decay=1e-4,
                            )

    # Log all layers in the model using wandb
    wandb.watch(model, log='all')

    # Apex: AMP fp16 mixed-precision training (not needed without GPU)
    if config.TRAIN.FP16:
        # Skip AMP initialization since no GPU is used
        pass

    # Create the dataloaders
    train_dataset = DatasetLoader(config)
    data_size = len(train_dataset)

    # eval after every epoch either on the test or validation set
    if config.RUN_MODE == 'train+val':
        config_test = copy.deepcopy(config)
        config_test.RUN_MODE = 'test'
        val_dataset = DatasetLoader(config_test)
        val_data_size = len(val_dataset)

    elif config.EVAL_EVERY_EPOCH:
        config_val = copy.deepcopy(config)
        config_val.RUN_MODE = 'val'
        val_dataset = DatasetLoader(config_val)
        val_data_size = len(val_dataset)

    # Create DataLoader for training and validation (no distributed sampler)
    train_dataloader = Data.DataLoader(
        train_dataset,
        config.BATCH_SIZE,
        shuffle=True,  # Shuffle data for training
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    val_dataloader = Data.DataLoader(
        val_dataset,
        config.BATCH_SIZE,
        shuffle=False,  # No shuffle for validation
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    # Training the net
    for epoch in range(config.MAX_EPOCH):

        time_start = time.time()

        # iterations
        for step, (inputs, labels) in enumerate(train_dataloader):

            optimizer.zero_grad()

            # Make the tensors for CPU
            for key in inputs.keys():
                inputs[key] = inputs[key].to(device)  # Ensure inputs are on CPU

            labels = labels.to(device)  # Ensure labels are on CPU

            # Obtain the predictions
            pred = model(inputs)

            # Loss calculation
            loss = criterion(pred, labels)

            # Backpropagation
            loss.backward()

            # Optimize
            optimizer.step()

            # Print progress
            print("\r[Epoch: %2d][Step: %d/%d][Loss: %.4f]" % (
                epoch + 1,
                step,
                int(data_size / config.BATCH_SIZE),
                loss
            ), end='    ')

            # Log loss to wandb
            wandb.log({'Loss': loss})

        time_end = time.time()
        time_taken = time_end - time_start
        print('Finished in {}s'.format(int(time_taken)))

        # Save the model checkpoints
        state = {
            'state_dict': model.state_dict(),
            'epoch': epoch + 1
        }

        torch.save(
            state,
            config.OUTPUT_PATH +
            '/ckpt_' + config.VERSION +
            '_epoch' + str(epoch + 1) +
            '.pkl'
        )

        # Calling the validation function
        if val_dataset is not None:
            validate(
                config,
                val_dataloader,
                val_data_size,
                model,
                local_rank=None  # No need for local rank when using CPU
            )
