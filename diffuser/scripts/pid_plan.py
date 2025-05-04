import json
import numpy as np
from os.path import join
import pdb

from diffuser.guides.policies import Policy
from diffuser.guides.functions import n_step_guided_p_sample
import diffuser.datasets as datasets
import diffuser.utils as utils


class Parser(utils.Parser):
    dataset: str = 'pointmaze-umaze-v2'
    config: str = 'config.maze2d'
    env_name: str = 'PointMaze_UMaze-v3'

#---------------------------------- setup ----------------------------------#

args = Parser().parse_args('plan')

# logger = utils.Logger(args)
## load diffusion model and value function from disk
diffusion_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.diffusion_loadpath,
    epoch=args.diffusion_epoch, 
)
value_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.value_loadpath,
    epoch=args.value_epoch, 
)

## ensure that the diffusion model and value function are compatible with each other
utils.check_compatibility(diffusion_experiment, value_experiment)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

## initialize value guide
value_function = value_experiment.ema

guide_config = utils.Config(args.guide, model=value_function, verbose=False)
guide = guide_config()


logger_config = utils.Config(
    utils.Logger,
    renderer=renderer,
    logpath=args.savepath,
    vis_freq=args.vis_freq,
    max_render=args.max_render,
)

## policies are wrappers around an unconditional diffusion model and a value guide
policy_config = utils.Config(
    args.policy,
    guide=guide,
    scale=args.scale,
    diffusion_model=diffusion,
    normalizer=dataset.normalizer,
    preprocess_fns=args.preprocess_fns,
    ## sampling kwargs
    sample_fn=n_step_guided_p_sample,
    n_guide_steps=args.n_guide_steps,
    t_stopgrad=args.t_stopgrad,
    scale_grad_by_std=args.scale_grad_by_std,
    verbose=False,
)

logger = logger_config()
policy = policy_config()

env = datasets.load_environment(args.env_name, reset_target=True)
#---------------------------------- main loop ----------------------------------#
observation, _ = env.reset(options={'reset_cell': np.array([1, 1]), 'goal_cell': np.array([2, 3])})
# observation, _ = env.reset()

if args.conditional:
    print('Resetting target')
    env.set_target()

## set conditioning xy position to be the goal
target = np.array(observation['desired_goal'])
print("Target")
print(target)
cond = {
    diffusion.horizon - 1: np.array([*target, 0, 0]),
}
# cond = { 
#     diffusion.horizon - 1: np.array([1, 1, 0, 0]),
# }
## observations for rendering
rollout = [observation['observation']]

total_reward = 0

pid_controler = utils.PIDPositionVelocityController(12, 0, 0.8)
for t in range(env.max_episode_steps):
    state = observation['observation']

    ## can replan if desired, but the open-loop plans are good enough for maze2d
    ## that we really only need to plan once
    if t == 0:
        cond[0] = observation['observation']
        print("Condition")
        print(cond)
        print(policy)
        action, samples = policy(cond, batch_size=args.batch_size)
        # actions = samples.actions[0]
        sequence = samples.observations[0]

    # pdb.set_trace()

    # ####
    if t < len(sequence) - 1:
        next_waypoint = sequence[t+1]
    else:
        next_waypoint = sequence[-1].copy()
        next_waypoint[2:] = 0
        # pdb.set_trace()

    ## can use actions or define a simple controller based on state predictions
    # action = next_waypoint[:2] - state[:2]
    action = pid_controler.compute_action(next_waypoint, state, 0.05)
    # action += np.array([0.15, -0.15])
    print("Action")
    print(action)
    # pdb.set_trace()
    ####

    # else:
    #     actions = actions[1:]
    #     if len(actions) > 1:
    #         action = actions[0]
    #     else:
    #         # action = np.zeros(2)
    #         action = -state[2:]
    #         pdb.set_trace()


    next_observation, reward, terminal, _, _ = env.step(action)
    total_reward += reward
    # score = env.get_normalized_score(total_reward)
    score = total_reward
    print(
        f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | '
        f'{action}'
    )

    if 'pointmaze' in args.dataset:
        xy = next_observation['observation'][:2]
        goal = next_observation['desired_goal']
        print(
            f'maze | pos: {xy} | goal: {goal}'
        )

    ## update rollout observations
    rollout.append(next_observation['observation'])

    # logger.log(score=score, step=t)

    if t % args.vis_freq == 0 or terminal or t==299:
        fullpath = join(args.savepath, f'{t}.png')

        # if t == 0: 
            # renderer.composite(fullpath, samples.observations, ncol=1)



        # renderer.render_plan(join(args.savepath, f'{t}_plan.mp4'), samples.actions, samples.observations, state)

        ## save rollout thus far
        if t==299:
            
            # renderer.composite(join(args.savepath, 'rollout.png'), np.array(rollout)[None], ncol=1)
            renderer.composite_overlap(join(args.savepath, 'overlap.png'), np.array(rollout)[None], samples.observations, ncol=1)
            renderer.render_rollout_composite(join(args.savepath, 'overlap.mp4'), np.array(rollout)[None], samples.observations)
            # mujoco_renderer = utils.MuJoCoRenderer(args.env_name)

            # renderer.render_rollout(join(args.savepath, f'rollout.mp4'), rollout, fps=80)

        # logger.video(rollout=join(args.savepath, f'rollout.mp4'), plan=join(args.savepath, f'{t}_plan.mp4'), step=t)

    if terminal:
        break

    observation = next_observation

# logger.finish(t, env.max_episode_steps, score=score, value=0)

## save result as a json file
json_path = join(args.savepath, 'rollout.json')
json_data = {'score': score, 'step': t, 'return': total_reward, 'term': terminal,
    'epoch_diffusion': diffusion_experiment.epoch}
json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)