import os

# Must be set before numpy is imported (it reads these at BLAS init time).
# Training already parallelizes across -cpus worker processes; each one
# separately spinning up its own OpenBLAS thread pool (sized to the full
# core count by default) causes massive thread oversubscription, and on
# this machine's CPU that thread pool has also been observed to corrupt
# unrelated heap memory under heavy allocation churn (surfaces much later
# as nonsensical crashes deep in yaml parsing or elsewhere -- see
# docs/environment_setup_and_fixes.md). Pinning to 1 thread per process is
# also just standard practice for this outer-process-parallel + inner-BLAS
# combination regardless.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import gym, torch, logging

import argparse, itertools, random
import json, time

# Default to offline wandb mode so runs never block on the interactive
# "Create a W&B account / Use existing / Don't visualize" prompt -- explicit
# `WANDB_MODE=...` in the environment still overrides this.
os.environ.setdefault("WANDB_MODE", "offline")

import wandb

from vec_env.subproc_vec_env import SubprocVecEnv
from tqdm import tqdm

from config import config
from nasim_problem import NASimRRL as Problem
from training_lock import TrainingLockError, acquire_training_lock

# ----------------------------------------------------------------------------------------
def _print_curriculum_status(env, current_epoch):
	"""Print current curriculum stage based on epoch.
	
	Fetches actual curriculum settings from the environment and displays them.
	"""
	try:
		# Get curriculum info from first environment in the vectorized wrapper
		info = env.env_method('get_curriculum_info', indices=[0])[0]
		if not info:
			return  # No curriculum active
		
		params = info.get('realism_params', {})
		
		print(f"\n{'='*80}")
		print(f"[CURRICULUM] Epoch {current_epoch} | Stage: {info['name']}")
		print(f"  Epoch Range: {info['start_epoch']}-{info['end_epoch']}")
		print(f"{'='*80}")
		
		# IDS
		ids_config = params.get('ids_config', {})
		if ids_config.get('enabled'):
			print(f"  IDS: Enabled (decay={ids_config.get('detection_decay', 0):.2f}, "
			      f"threshold={ids_config.get('base_thresholds', [])})")
		else:
			print(f"  IDS: Disabled")
		
		# Scan Noise
		scan_noise = params.get('scan_noise', {})
		if 'service_scan' in scan_noise:
			ss = scan_noise['service_scan']
			print(f"  Scan Noise: service FP={ss.get('false_positive_rate', 0):.3f}, "
			      f"FN={ss.get('false_negative_rate', 0):.3f}")
		else:
			print(f"  Scan Noise: Disabled")
		
		# Network Reliability
		net_rel = params.get('network_reliability', {})
		if 'timeout_probability' in net_rel:
			print(f"  Network Reliability: timeout={net_rel['timeout_probability']:.3f}")
		else:
			print(f"  Network Reliability: Perfect (no timeouts)")
		
		# Service Dynamics
		svc_dyn = params.get('service_dynamics', {})
		if 'churn_probability' in svc_dyn:
			print(f"  Service Dynamics: churn={svc_dyn['churn_probability']:.3f}")
		else:
			print(f"  Service Dynamics: Disabled")
		
		print(f"{'='*80}\n")
	except Exception as e:
		# Silently skip if curriculum not available
		pass

# ----------------------------------------------------------------------------------------
def decay_time(step, start, min, factor, rate):
	exp = step / rate * factor
	value = (start - min) / (1 + exp) + min

	return value

def decay_exp(step, start, min, factor, rate):
	exp = step / rate
	value = (start - min) * (factor ** exp) + min

	return value

def scheduled_value_at_step(step, start, minimum, factor, rate, decay_fn):
	"""Return the value that was active immediately after ``step``.

	Schedules are only applied on exact ``rate`` boundaries in the training
	loop. A legacy weights-only checkpoint therefore needs the most recent
	boundary value, not ``decay_fn(step, ...)`` at an arbitrary interrupted
	step.
	"""
	if step <= 0:
		return start
	boundary = (step // rate) * rate
	if boundary <= 0:
		return start
	return decay_fn(boundary, start, minimum, factor, rate)

def capture_rng_state():
	np_name, np_keys, np_pos, np_has_gauss, np_cached_gaussian = np.random.get_state()
	state = {
		'python': random.getstate(),
		'numpy': {
			'bit_generator': np_name,
			'keys': torch.from_numpy(np_keys.copy()),
			'position': int(np_pos),
			'has_gauss': int(np_has_gauss),
			'cached_gaussian': float(np_cached_gaussian),
		},
		'torch': torch.get_rng_state(),
	}
	if torch.cuda.is_available():
		state['torch_cuda'] = torch.cuda.get_rng_state_all()
	return state

def restore_rng_state(state):
	if not state:
		return
	random.setstate(state['python'])
	np_state = state['numpy']
	np.random.set_state((
		np_state['bit_generator'],
		np_state['keys'].cpu().numpy(),
		np_state['position'],
		np_state['has_gauss'],
		np_state['cached_gaussian'],
	))
	torch.set_rng_state(state['torch'])
	if 'torch_cuda' in state and torch.cuda.is_available():
		torch.cuda.set_rng_state_all(state['torch_cuda'])

def make_training_state(step, env_steps_total, episodes_completed, best_value,
						net, target_net, args, config):
	"""Build the restart metadata embedded in newly saved checkpoints."""
	return {
		'format_version': 1,
		'step': int(step),
		'env_steps_total': int(env_steps_total),
		'episodes_completed': int(episodes_completed),
		'best_value': float(best_value),
		'best_split': config.save_best_split,
		'best_metric': config.save_best_metric,
		'lr': float(net.lr),
		'alpha_h': float(net.alpha_h),
		'optimizer_state_dict': net.opt.state_dict(),
		'target_state_dict': target_net.state_dict(),
		'rng_state': capture_rng_state(),
		'run_config': {
			'scenario': args.scenario,
			'test_scenario': args.test_scenario,
			'net_class': args.net_class,
			'batch': config.batch,
			'epoch': config.epoch,
			'max_epochs': config.max_epochs,
			'seed': config.seed,
			'opt_lr': config.opt_lr,
			'initial_alpha_h': config.alpha_h,
			'episode_step_limit': config.step_limit,
			'use_a_t': config.use_a_t,
			'observation_format': config.observation_format,
			'sched_lr_rate': config.sched_lr_rate,
			'sched_lr_factor': config.sched_lr_factor,
			'sched_lr_min': config.sched_lr_min,
			'sched_alpha_h_rate': config.sched_alpha_h_rate,
			'sched_alpha_h_factor': config.sched_alpha_h_factor,
			'sched_alpha_h_min': config.sched_alpha_h_min,
		},
	}

def validate_resume_run_config(saved, args, config):
	"""Refuse silent scientific-configuration drift on full-state resumes."""
	if not saved:
		return
	current = {
		'scenario': args.scenario,
		'test_scenario': args.test_scenario,
		'net_class': args.net_class,
		'batch': config.batch,
		'epoch': config.epoch,
		'seed': config.seed,
		'opt_lr': config.opt_lr,
		'initial_alpha_h': config.alpha_h,
		'episode_step_limit': config.step_limit,
		'use_a_t': config.use_a_t,
		'observation_format': config.observation_format,
		'sched_lr_rate': config.sched_lr_rate,
		'sched_lr_factor': config.sched_lr_factor,
		'sched_lr_min': config.sched_lr_min,
		'sched_alpha_h_rate': config.sched_alpha_h_rate,
		'sched_alpha_h_factor': config.sched_alpha_h_factor,
		'sched_alpha_h_min': config.sched_alpha_h_min,
	}
	mismatches = [
		f"{key}: checkpoint={saved[key]!r}, command={value!r}"
		for key, value in current.items()
		if key in saved and saved[key] != value
	]
	if mismatches:
		raise SystemExit(
			"Resume command does not match the checkpoint's training configuration:\n  "
			+ "\n  ".join(mismatches)
		)

	saved_max_epochs = saved.get('max_epochs')
	if saved_max_epochs is not None and config.max_epochs is not None and config.max_epochs < saved_max_epochs:
		raise SystemExit(
			f"Resume command shortens -max_epochs from {saved_max_epochs} to {config.max_epochs}. "
			"Keep the original endpoint or extend it."
		)

def init_seed(seed):
	np.random.seed(seed)
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)

def command_starts_training(args):
	"""Return whether this invocation enters the mutating training loop."""
	return not any(bool(getattr(args, flag, False)) for flag in (
		'calc_baseline', 'trace', 'eval', 'debug',
	))

def get_args(problem_config):
	cuda_devices = [f'cuda:{i}' for i in range(torch.cuda.device_count())]

	# optimal cpu=2, device=cuda (rate 3.5)
	parser = argparse.ArgumentParser()
	parser.add_argument('-device', type=str, choices=['auto', 'cpu', 'cuda'] + cuda_devices, default='cpu', help="Which device to use")
	parser.add_argument('-cpus', type=str, default='2', help="How many CPUs to use")
	parser.add_argument('-batch', type=int, default=128, help="Number of parallel environments")
	parser.add_argument('-seed', type=int, default=None, help="Random seed (each of the -batch parallel envs gets seed+i; see NASimEmuEnv.__init__)")
	parser.add_argument('-load_model', type=str, default=None, help="Load model from this file")
	parser.add_argument('--resume', action='store_true', help="Resume training progress from -load_model. New checkpoints restore embedded trainer state automatically; legacy weights-only checkpoints also require --resume_step.")
	parser.add_argument('-resume_step', '--resume_step', type=int, default=None, help="Global completed training step for a legacy weights-only checkpoint (for example 11600). Ignored when embedded trainer state is available.")
	parser.add_argument('-resume_best_value', '--resume_best_value', type=float, default=None, help="Best metric achieved before a legacy checkpoint interruption. Preserves best-checkpoint tracking across the recovered run.")
	parser.add_argument('-save_best_split', choices=['eval_trn', 'eval_tst'], default='eval_tst', help="Which eval split's metric to track for the best-checkpoint save (falls back to eval_trn if no -test_scenario is configured)")
	parser.add_argument('-save_best_metric', choices=['reward_avg', 'reward_avg_episodes', 'eplen_avg', 'captured_avg'], default='captured_avg', help="Metric to track for the best-checkpoint save; higher is always better for all four (eplen_avg is included for completeness, not recommended as the optimization target)")
	parser.add_argument('-epoch', type=int, default=1000, help="Epoch length")
	parser.add_argument('-max_epochs', type=int, default=None, help="Terminate after this many epochs")

	parser.add_argument('-mp_iterations', type=int, default=3, help="Number of message passes")
	parser.add_argument('-emb_dim', type=int, default=64, help="Embedding size")

	parser.add_argument('-force_continue_epochs', type=int, default=0, help="Disable force continue after this epochs (0=disable immediately; -1=never disable)")

	parser.add_argument('-lr', type=float, default=3e-3, help="Initial learning rate")
	parser.add_argument('-alpha_h', type=float, default=0.3, help="Initial entropy regularization constant")
	parser.add_argument('-max_norm', type=float, default=3., help="Maximal gradient norm")

	# Learning rate / entropy decay schedules
	parser.add_argument('--sched_lr_rate', type=int, default=None, help="Steps between LR decay updates")
	parser.add_argument('--sched_lr_factor', type=float, default=None, help="Exponential LR decay factor")
	parser.add_argument('--sched_lr_min', type=float, default=None, help="Minimum learning rate")
	parser.add_argument('--sched_alpha_h_rate', type=int, default=None, help="Steps between entropy coeff. decay updates")
	parser.add_argument('--sched_alpha_h_factor', type=float, default=None, help="Time-decay factor for entropy coeff.")
	parser.add_argument('--sched_alpha_h_min', type=float, default=None, help="Minimum entropy coefficient")

	parser.add_argument('--trace', action='store_const', const=True, help="Show trace of the agent")
	parser.add_argument('--eval', action='store_const', const=True, help="Evaluate the agent")
	parser.add_argument('--debug', action='store_const', const=True, help="Debug the agent")
	parser.add_argument('--calc_baseline', action='store_const', const=True, help="Calculate required steps of a baseline agent")
	
	parser.add_argument('--no_debug', action='store_const', const=True, help="Do not debug the agent")

	# delegate argparse to problem config
	problem_config.update_argparse(parser)

	cmd_args = parser.parse_args()
	return cmd_args

# ----------------------------------------------------------------------------------------
if __name__ == '__main__':
	# logging.basicConfig(level=logging.INFO)
	logging.basicConfig(level=logging.DEBUG)
	logging.getLogger('urllib3').setLevel(logging.INFO)
	logging.getLogger('numba').setLevel(logging.INFO)

	problem = Problem()
	problem_config = problem.make_config()

	np.set_printoptions(threshold=np.inf, precision=4, suppress=True)

	args = get_args(problem_config)
	if args.resume and not args.load_model:
		raise SystemExit("--resume requires -load_model CHECKPOINT")
	if args.resume_step is not None and not args.resume:
		raise SystemExit("--resume_step only applies with --resume")
	if args.resume_best_value is not None and not args.resume:
		raise SystemExit("--resume_best_value only applies with --resume")
	if args.resume_step is not None and args.resume_step < 0:
		raise SystemExit("--resume_step must be >= 0")

	# A second independent trainer would compete for the same CPU set and both
	# processes would append to the fixed training_data/latest/latest.json path.
	# Hold a kernel-backed, crash-safe lock before creating any environments or
	# touching run logs. Read-only/debug commands deliberately bypass the guard.
	training_run_lock = None
	if command_starts_training(args):
		try:
			training_run_lock = acquire_training_lock()
		except TrainingLockError as exc:
			raise SystemExit(str(exc))
		print(f"[run-lock] Acquired exclusive training lock: {training_run_lock.path}")

	config.init(args)
	problem_config.update_config(config, args) # update config with problem specific settings

	print(f"Config: {config}")

	if config.seed:
		init_seed(config.seed)

	torch.set_num_threads(config.cpus)	

	problem.register_gym()
	problem_debug = problem.make_debug()
	
	if args.calc_baseline:
		problem_debug.calc_baseline()
		exit(0)

	net = problem.make_net()
	target_net = problem.make_net()
	print(net)
	print(f"Number of parameters: {net.get_param_count()}")

	checkpoint_training_state = None
	if config.load_model:
		checkpoint_training_state = net.load(config.load_model)
		target_net.load(config.load_model)

		print(f"Model loaded: {config.load_model}")

	resume_step = 0
	resume_env_steps = 0
	resume_episodes_completed = 0
	resume_best_value = float('-inf')
	resume_rng_state = None
	if args.resume:
		if checkpoint_training_state is not None:
			format_version = checkpoint_training_state.get('format_version')
			if format_version != 1:
				raise SystemExit(f"Unsupported checkpoint training_state format_version={format_version!r}")
			validate_resume_run_config(checkpoint_training_state.get('run_config'), args, config)
			resume_step = int(checkpoint_training_state['step'])
			resume_env_steps = int(checkpoint_training_state.get('env_steps_total', resume_step * config.batch))
			resume_episodes_completed = int(checkpoint_training_state.get('episodes_completed', 0))
			resume_best_value = float(checkpoint_training_state.get('best_value', float('-inf')))
			resume_rng_state = checkpoint_training_state.get('rng_state')

			saved_split = checkpoint_training_state.get('best_split')
			saved_metric = checkpoint_training_state.get('best_metric')
			if saved_split and saved_split != config.save_best_split:
				raise SystemExit(
					f"Resume checkpoint tracked best split {saved_split!r}, but this run uses "
					f"{config.save_best_split!r}. Pass -save_best_split {saved_split}."
				)
			if saved_metric and saved_metric != config.save_best_metric:
				raise SystemExit(
					f"Resume checkpoint tracked best metric {saved_metric!r}, but this run uses "
					f"{config.save_best_metric!r}. Pass -save_best_metric {saved_metric}."
				)

			net.opt.load_state_dict(checkpoint_training_state['optimizer_state_dict'])
			target_net.load_state_dict(checkpoint_training_state['target_state_dict'])
			net.set_lr(float(checkpoint_training_state['lr']))
			net.set_alpha_h(float(checkpoint_training_state['alpha_h']))
			print(
				f"[resume] Restored embedded trainer state at step={resume_step}, "
				f"env_steps={resume_env_steps}, lr={net.lr:.8g}, alpha_h={net.alpha_h:.8g}, "
				f"best={resume_best_value:.6g}."
			)
		else:
			if args.resume_step is None:
				raise SystemExit(
					"This checkpoint contains weights only. Supply --resume_step with the last "
					"completed global training step."
				)
			resume_step = int(args.resume_step)
			resume_env_steps = resume_step * config.batch
			if args.resume_best_value is not None:
				resume_best_value = float(args.resume_best_value)

			resume_lr = scheduled_value_at_step(
				resume_step, config.opt_lr, config.sched_lr_min,
				config.sched_lr_factor, config.sched_lr_rate, decay_exp,
			)
			resume_alpha_h = scheduled_value_at_step(
				resume_step, config.alpha_h, config.sched_alpha_h_min,
				config.sched_alpha_h_factor, config.sched_alpha_h_rate, decay_time,
			)
			net.set_lr(resume_lr)
			net.set_alpha_h(resume_alpha_h)
			print(
				f"[resume] Legacy weights-only recovery at step={resume_step}: derived "
				f"env_steps={resume_env_steps}, lr={net.lr:.8g}, alpha_h={net.alpha_h:.8g}, "
				f"best={resume_best_value:.6g}. Optimizer/target/RNG/environment state "
				f"was not present and cannot be recovered retroactively."
			)

		if config.max_epochs:
			total_steps = config.max_epochs * config.log_rate
			if resume_step >= total_steps:
				raise SystemExit(
					f"Resume step {resume_step} has already reached the configured run end "
					f"({config.max_epochs} epochs x {config.log_rate} steps = {total_steps})."
				)

	if args.trace:
		problem_debug.trace(net, config.load_model)
		exit(0)

	if args.eval:
		import pprint
		eval_res = problem_debug.evaluate(net)
		# print(f"Avg. reward: {r_avg}, Avg. solved per step: {s_ps_avg}, Avg. solved: {s_avg}, Tot. finished: {s_tot}")
		pprint.pp(eval_res)
		exit(0)

	if args.debug:
		problem_debug.debug(net, show=True)
		exit(0)

	# Each parallel env gets its own deterministic-but-distinct seed derived from
	# config.seed (only when the user actually requested one) -- `i=i` binds the
	# loop variable at lambda-creation time, since these lambdas aren't called
	# until later inside the forked worker processes (late-binding closure).
	# See NASimEmuEnv.__init__ (env.py) for why this is required for -seed to
	# actually reproduce the same per-env scenario sequence across runs, instead
	# of every worker silently reseeding itself from fresh OS entropy.
	env = SubprocVecEnv([
		lambda i=i: gym.make(problem.get_gym_name(), seed=(config.seed + i) if config.seed else None)
		for i in range(config.batch)
	], in_series=(config.batch // config.cpus), context='fork')

	wandb.init(project=problem.get_project_name(), name=problem.get_run_name(), config=config)
	wandb.watch(net, log='all')

	# ---------------------------
	# Local per-episode JSON logger
	# ---------------------------
	log_dir = os.environ.get(
		'NASIMEMU_RUN_LOG_DIR',
		os.path.join(os.path.dirname(__file__), 'training_data', 'runs'),
	)
	os.makedirs(log_dir, exist_ok=True)
	jsonl_path = os.path.join(log_dir, f'{wandb.run.id}.json')

	# "latest" convenience mirror: same records, always at a fixed path, so you
	# don't need to look up the wandb run id to tail the current run. Truncated
	# fresh at the start of THIS run -- the per-run file above (keyed by run id)
	# remains the collision-safe source of truth if multiple runs ever overlap.
	latest_dir = os.environ.get(
		'NASIMEMU_LATEST_DIR',
		os.path.join(os.path.dirname(__file__), 'training_data', 'latest'),
	)
	os.makedirs(latest_dir, exist_ok=True)
	latest_path = os.path.join(latest_dir, 'latest.json')
	open(latest_path, 'w').close()

	def _append_jsonl(path, obj):
		try:
			with open(path, 'a') as f:
				f.write(json.dumps(obj) + "\n")
		except Exception as e:
			logging.getLogger(__name__).warning(f"Failed to write JSON log: {e}")

	tot_env_steps = resume_env_steps
	best_val = resume_best_value
	norm_log = []
	entropy_log = []

	if config.force_continue_steps >= 0 and resume_step >= config.force_continue_steps:
		print("Disabling force_continue")
		net.set_force_continue(False)
	else:
		print("Enabling force_continue")
		net.set_force_continue(True)

	resume_epoch = resume_step // config.log_rate
	total_training_steps = config.max_epochs * config.log_rate if config.max_epochs else None
	tqdm_main = tqdm(desc='Training', unit=' steps', initial=resume_step, total=total_training_steps)
	s = env.reset()
	if resume_step:
		# NASimEmuEnv creates its inner NASim environment on the first reset,
		# so position the curriculum only after that initialization, then reset
		# once more to begin the fresh episode at the restored difficulty.
		env.env_method('set_epoch', resume_epoch)
		s = env.reset()
		print(f"[resume] Curriculum positioned at epoch={resume_epoch}; next global step={resume_step + 1}.")
	# Environment subprocesses cannot be checkpointed and start fresh episodes.
	# Restore the parent RNG only after construction/reset has consumed its own
	# random values so future trainer-side sampling continues from the saved
	# stream whenever a full training-state checkpoint is available.
	restore_rng_state(resume_rng_state)
	
	# Track total episodes completed across all parallel envs for curriculum
	total_episodes_completed = resume_episodes_completed
	
	# Get curriculum stage transition epochs dynamically from scenario
	try:
		curriculum_transition_epochs = env.env_method('get_stage_transition_epochs', indices=[0])[0]
	except:
		curriculum_transition_epochs = []

	for step in itertools.count(start=resume_step + 1):
		trace = []

		hidden_s0 = problem.make_net()		# save internal (recurrent) network state at s_0 and s_last
		hidden_s0.clone_state(net)

		for step_trace in range(config.ppo_t):
			s_orig = s
			
			a, v, pi, raw_a = net(s)
			a = np.array(a, dtype=object)
			s, r, d, i = env.step(a)
			net.reset_state(d)

			a_cnt = [0 if a_action == -1 else 1 for (a_node, a_action) in a] # action_q - 0 = terminate / 1 = continue

			s_true = [x['s_true'] for x in i]
			d_true = [x['d_true'] for x in i] # note: currently d == d_true (dependency in v_target, q_target computations and reccurency in ppo

			# Track episode completions for monitoring
			episodes_done_this_step = sum(d_true)
			total_episodes_completed += episodes_done_this_step

			trace.append( (s_orig, raw_a, a_cnt, r, s_true, d_true) )

		# update network
		# loss, entropy, norm, pi_deviations = net.update(s_orig, raw_a, r, s_true, d_true, target_net)
		target_net.clone_state(net)
		loss, entropy, norm, pi_deviations = net.update(trace, target_net, hidden_s0)
		target_net.copy_weights(net, rho=config.target_rho)

		# print([x.item() for x in pi_deviations])

		# save step stats
		tot_env_steps += config.batch
		tqdm_main.update()

		norm_log.append(norm)
		entropy_log.append(entropy)

		if step % config.sched_lr_rate == 0:
			lr = decay_exp(step, config.opt_lr, config.sched_lr_min, config.sched_lr_factor, config.sched_lr_rate)
			net.set_lr(lr)

		if step % config.sched_alpha_h_rate == 0:
			alpha_h = decay_time(step, config.alpha_h, config.sched_alpha_h_min, config.sched_alpha_h_factor, config.sched_alpha_h_rate)
			net.set_alpha_h(alpha_h)

		if step % config.log_rate == 0:
			log_step = step // config.log_rate
			current_epoch = log_step
			
			# Update curriculum based on current epoch
			env.env_method('set_epoch', current_epoch)

			# r_avg, s_ps_avg, s_avg, _ = problem_debug.evaluate(net)
			# r_avg_trn, s_ps_avg_trn, s_avg_trn, _ = problem_debug.evaluate(net, split='train', subset=config.subset)

			eval_perf = problem_debug.evaluate(net)
			# log_trn_eval = problem_debug.evaluate(net, split='train', subset=config.subset)

			if args.no_debug:
				log_debug = None
			else:
				log_debug = problem_debug.debug(net)
				# print(log_debug['value'], log_debug['q_val'])
		
			log = {
				'env_steps': tot_env_steps,
				'episodes_completed': total_episodes_completed,  # Track curriculum progress
				# 'el_env_steps': tot_el_env_steps,
				'rate': tqdm_main.format_dict['rate'],

				'loss': loss,
				# 'loss_pi': loss_pi,
				# 'loss_v': loss_v,
				# 'loss_h': loss_h,

				'pi_deviations': wandb.Histogram(pi_deviations),

				'grad_mean': np.mean(norm_log),
				'grad_min': np.min(norm_log),
				'grad_max': np.max(norm_log),

				'entropy_mean': np.mean(entropy_log),
				'entropy_min': np.min(entropy_log),
				'entropy_max': np.max(entropy_log),

				'lr': net.lr,
				'alpha_h': net.alpha_h,

				'eval_perf': eval_perf,

				'debug': log_debug,
			}

			# Print curriculum status at meaningful intervals
			# Print at: early epochs (0-5), every 10 epochs, and at stage transitions from scenario
			should_print = (
				current_epoch <= 5 or  # Early training
				current_epoch % 10 == 0 or  # Every 10 epochs
				current_epoch in curriculum_transition_epochs  # Stage transitions from scenario
			)
			if should_print:
				_print_curriculum_status(env, current_epoch)

			norm_log = []
			entropy_log = []

			print(log)
			wandb.log(log)

			# Update the global best before building checkpoint metadata so both
			# model.pt and a newly improved model_best.pt carry the same value.
			split = config.save_best_split
			metric_name = config.save_best_metric
			split_perf = eval_perf.get(split) or eval_perf.get('eval_trn')
			cur_val = split_perf.get(metric_name) if split_perf else None
			is_new_best = cur_val is not None and cur_val > best_val
			if is_new_best:
				best_val = cur_val

			# Write one JSON record per logging interval (epoch-like)
			def _to_serializable(x):
				try:
					import numpy as _np
					if isinstance(x, (_np.floating, _np.integer)):
						return x.item()
				except Exception:
					pass
				return x

			log_json = {
				'run_id': wandb.run.id,
				'trainer_pid': os.getpid(),
				'timestamp': time.time(),
				'train_step': int(step),
				'env_steps_total': int(tot_env_steps),
				'resume_start_step': int(resume_step),
				'best_value': float(best_val),
				'loss': float(_to_serializable(log['loss'])),
				'grad_mean': float(_to_serializable(log['grad_mean'])),
				'grad_min': float(_to_serializable(log['grad_min'])),
				'grad_max': float(_to_serializable(log['grad_max'])),
				'entropy_mean': float(_to_serializable(log['entropy_mean'])),
				'entropy_min': float(_to_serializable(log['entropy_min'])),
				'entropy_max': float(_to_serializable(log['entropy_max'])),
				'lr': float(_to_serializable(log['lr'])),
				'alpha_h': float(_to_serializable(log['alpha_h'])),
				'eval_trn': {k: _to_serializable(v) for k, v in (log['eval_perf'].get('eval_trn') or {}).items()},
				'eval_tst': {k: _to_serializable(v) for k, v in (log['eval_perf'].get('eval_tst') or {}).items()},
			}
			_append_jsonl(jsonl_path, log_json)
			_append_jsonl(latest_path, log_json)
			
			# save model to wandb
			model_file = os.path.join(wandb.run.dir, "model.pt")
			os.makedirs(os.path.dirname(model_file), exist_ok=True)
			training_state = make_training_state(
				step, tot_env_steps, total_episodes_completed, best_val,
				net, target_net, args, config,
			)
			net.save(model_file, training_state=training_state)
			wandb.save(model_file)

			if is_new_best:
				best_model_file = os.path.join(wandb.run.dir, "model_best.pt")
				net.save(best_model_file, training_state=training_state)
				wandb.save(best_model_file)
				print(f"[save_best] new best {split}/{metric_name}={cur_val:.4f} at epoch {current_epoch} -> {best_model_file}")
		

			# if per-epoch auto mode, roll scenario for next epoch
			try:
				if getattr(config, 'auto_mode', 'off') == 'per_epoch':
					# broadcast to all workers
					env.env_method('set_roll_on_next_reset', True)
			except Exception as _:
				pass
 
		# finish if max_epochs exceeded
		if config.max_epochs and (step // config.log_rate >= config.max_epochs):
			break

		if step == config.force_continue_steps:
			print("Disabling force_continue")
			net.set_force_continue(False)

	env.close()
	tqdm_main.close()
