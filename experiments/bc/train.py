import os
import wandb
import logging
import argparse

from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import optim

from bmoca.utils import set_random_seed

from bmoca.agent.custom.bc.bc_policy import BCContPolicy, BCDiscPolicy
from bmoca.agent.custom.bc.dataset import BCContReplayBuffer, BCDiscReplayBuffer

from bmoca.agent.custom.utils import dist, to_torch
from bmoca.agent.custom.bc.scheduler import CosineAnnealingWarmUpRestarts


_WORK_PATH = os.environ['BMOCA_HOME']

# tasks to learn BC for
_TASK_NAME_LIST = [
    "create_alarm_at_10:30_am",
    "create_alarm_at_10:30_am_on_every_weekday",
    "go_to_the_'add_a_language'_page_in_setting",
    "input_'cos(180)'_in_Calculator"
    "call_the_white_house_(202-456-1111)",
    "disable_the_top_2_and_'randomizer'_topics_in_the_feed_customization_settings_on_Wikipedia_and_go_back_to_the_feed",
    "call_911",
]


def run(args):
    ## Replay Buffer
    if args.action_space == 'continuous':
        replay_buffer = BCContReplayBuffer(
                            replay_dir=args.demo_dir, 
                            task_instructions=_TASK_NAME_LIST,
                            num_env=args.env_num
                        )
    elif args.action_space == 'discrete':
        replay_buffer = BCDiscReplayBuffer(
                            replay_dir=args.demo_dir, 
                            task_instructions=_TASK_NAME_LIST,
                            num_env=args.env_num
                        )
    
    replay_buffer._load() 

    replay_loader = torch.utils.data.DataLoader(
                                replay_buffer,
                                batch_size=args.batch_size,
                                pin_memory=True,
                                num_workers=4,
                                prefetch_factor=2,
                                persistent_workers=True,
                            )
    replay_iter = iter(replay_loader)
    
    ## Model
    if args.action_space == 'continuous':
        model = BCContPolicy()
    elif args.action_space == 'discrete':
        print("model with concat version building...")
        model = BCDiscPolicy(action_shape=[385])
        print("model with concat version built!!!")


    model = model.to(args.device)

    optimizer = optim.Adam(model.get_train_parameters(), lr=0)
    # learning rate is controlled by consine annealing
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=args.T_0, T_mult=args.T_mult, eta_max=args.eta_max,  T_up=args.T_up, gamma=args.gamma)

    use_data_parallel=False

    # wandb
    if args.log_wandb:
        wandb_project = 'bmoca'
        
        wandb_run_name = args.wandb_name
        wandb_run_name += f"_seed_{args.seed}"
        if args.output_message is not None: wandb_run_name += f"_{args.output_message}"

        run_id = wandb.util.generate_id()
        run_name = f'{wandb_run_name}.{run_id}'
        logging.info(f"wandb running with {run_name}")

        wandb.init(project=wandb_project, 
                config=args)
        wandb.run.name = run_name
        wandb.run.save()

    # main loop
    print("START OF MAIN LOOP")

    tap_acc = 0
    for e in tqdm(range(args.train_steps), desc='Training'):
        train_loss = 0

        # sample batch
        batch = next(replay_iter)

        # train
        model.train()
        model.zero_grad()

        instruction_input_ids, instruction_attention_mask,\
            observation, label_gesture = to_torch(batch, args.device)
        
        if args.action_space == 'continuous':
            label_gesture = label_gesture.float()
            pred_gesture = model(instruction_input_ids, instruction_attention_mask, observation)
            loss = F.mse_loss(pred_gesture, label_gesture)
        elif args.action_space == 'discrete':
            target_actions = label_gesture.squeeze(1)
            pred_actions = model(instruction_input_ids, instruction_attention_mask, observation)
            loss = F.cross_entropy(pred_actions, target_actions)        
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # stats
        batch_num = label_gesture.shape[0]
        train_loss += loss.detach().item() * batch_num
        
        scheduler.step()
        if (e + 1) % args.log_period == 0:
            metrics = {'train_loss': train_loss, 'lr': scheduler.get_lr()[0], 'tap_acc': tap_acc}
            if args.log_wandb:
                wandb.log(metrics)
        
        if (e + 1) % args.eval_period == 0:
            model.eval()
            print(f"train_loss: {train_loss}, lr: {scheduler.get_lr()[0]}")
            # get tap accuracy
            tap_num = 0
            threshold = 0.1
            
            if args.action_space == 'continuous':
                label_gesture_cpu = label_gesture.cpu().detach().numpy()
                pred_gesture_cpu = pred_gesture.cpu().detach().numpy()
            elif args.action_space == 'discrete':
                pred_actions_cpu = pred_actions.cpu().detach().numpy().argmax(axis=1)
                target_actions_cpu = target_actions.cpu().detach().numpy()
                
            for i in range(batch_num):
                if args.action_space == 'continuous':
                    if dist(label_gesture_cpu[i, :2], pred_gesture_cpu[i, :2]) < threshold \
                        and dist(label_gesture_cpu[i, 2:], pred_gesture_cpu[i, 2:]) < threshold:
                        tap_num += 1
                elif args.action_space == 'discrete':
                    if pred_actions_cpu[i] == target_actions_cpu[i]:
                        tap_num += 1
                    
            tap_acc = tap_num / batch_num
            if args.log_wandb:
                metrics.update({'tap_acc': tap_acc})
                wandb.log(metrics)
            print(f"tap_acc: {tap_acc}, train_loss: {train_loss}, lr: {scheduler.get_lr()[0]}")

        # save model
        if (e + 1) % args.save_period == 0:
            if use_data_parallel:
                model_ = model.module
            else:
                model_ = model
            # model_.save(f'{args.output_dir}/bc_policy_{e + 1}_env_{args.env_num:02d}_seed_{args.seed}.pt')
            model_.save(f'{args.output_dir}/bc_policy_latest_env_{args.env_num:02d}_seed_{args.seed}.pt')
    
    # last save when the training is over
    if use_data_parallel:
        model_ = model.module
    else:
        model_ = model
    model_.save(f'{args.output_dir}/bc_policy_latest_env_{args.env_num:02d}_seed_{args.seed}.pt')


def parse_args():
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--text_encoder', type=str, 
                        default=f"{_WORK_PATH}/asset/agent/Auto-UI-Base")
    parser.add_argument('--input_len', type=int, default=32)
    parser.add_argument('--action_history', default=False, action='store_true')
    parser.add_argument('--load_dir', type=str, default=None)
    # training
    parser.add_argument('--seed', type=int, default=125, help='random seed')
    parser.add_argument('--train_with_eval', default=False, 
                        action='store_true')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--train_steps', type=int, default=10000)
    parser.add_argument('--log_period', type=int, default=1, 
                        help = 'every log_period loss is computed and logged in wandb')
    parser.add_argument('--eval_period', type=int, default=10, 
                        help= 'every eval_period tap_acc is computed and logged in wandb')
    parser.add_argument('--save_period', type=int, default=2000, 
                        help='every save_period model is saved')
    parser.add_argument('--action_space', type=str, default='discrete', 
                        choices=['continuous', 'discrete'], help='action space to be used')
    # scheduler
    parser.add_argument('--T_0', type=int, default=2000)
    parser.add_argument('--T_mult', type=int, default=1)
    parser.add_argument('--eta_max', type=float, default=3e-4, 
                        help='max learning rate for 1st T_0')
    parser.add_argument('--T_up',type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.8)
    # output
    parser.add_argument('--demo_dir', type=str, 
                        default=f"{_WORK_PATH}/asset/dataset/demonstrations_dual_gesture")
    parser.add_argument('--env_num', type=int, default=7)
    parser.add_argument('--output_dir', type=str, default=f"{_WORK_PATH}/logs/bc")
    parser.add_argument('--output_message', type=str, default=None)
    parser.add_argument('--debug', default=False, action='store_true')
    # misc.
    parser.add_argument('--log_wandb', default=True, action='store_true')
    parser.add_argument('--wandb_name', type=str, default='bc')
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    set_random_seed(args.seed)
    
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    ## Log directory
    if not os.path.isdir(args.output_dir): os.makedirs(args.output_dir, exist_ok=True)
    print("output_dir:", args.output_dir)
    
    ## main
    run(args)
