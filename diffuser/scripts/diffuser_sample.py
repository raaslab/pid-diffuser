import diffuser.utils as utils

class Args:
  loadpath = '/fs/nexus-scratch/ktorsh/diffuser/logs/pointmaze-umaze-v2/diffusion/H128_T64'
  diffusion_epoch = 'latest'
  n_samples = 1
  device = 'cuda:0'
    
args = Args()

diffusion_experiment = utils.load_diffusion(
    args.loadpath, epoch=args.diffusion_epoch)

dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer
model = diffusion_experiment.trainer.ema_model

env = dataset.env
obs = env.reset()

observations = utils.colab.run_diffusion(
    model, dataset, obs, args.n_samples, args.device)
print(observations.shape)

sample = observations[-1]
utils.colab.show_sample(renderer, sample)

utils.colab.show_diffusion(renderer, observations[:,:1], substep=1)

