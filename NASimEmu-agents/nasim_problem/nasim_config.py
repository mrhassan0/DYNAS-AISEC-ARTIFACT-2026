from nasimemu.env import NASimEmuEnv
from .nasim_net_feudal import FeudalGTM
from .nasim_net_base_hrl import NASimNetDHRL


from .nasim_net_mlp import NASimNetMLP # Multi-layer perceptron
from .nasim_net_mlp_lstm import NASimNetMLP_LSTM # Multi-layer perceptron + LSTM

from .nasim_net_inv import NASimNetInv # Permutation invariant + compound action (p_node * p_act)
from .nasim_net_inv_mact import NASimNetInvMAct # Inv + Matrix action
from .nasim_net_inv_mact_train_at import NASimNetInvMActTrainAT # Inv + Matrix action + trainable a_t
from .nasim_net_inv_mact_lstm import NASimNetInvMActLSTM # Inv + Matrix action + GRU

from .nasim_net_gnn import NASimNetGNN	# Graph NN
from .nasim_net_gnn_mact import NASimNetGNN_MAct # Graph NN + Matrix action
from .nasim_net_gnn_lstm import NASimNetGNN_LSTM # Graph NN with LSTM

from .nasim_net_xatt import NASimNetXAtt # Attention
from .nasim_net_xatt_mact import NASimNetXAttMAct # Attention 

from .nasim_baseline import BaselineAgent 

class NASimConfig():
	@staticmethod
	def update_config(config, args):
		# Set training mode based on --eval flag
		# training_mode=True enables curriculum learning, False uses max difficulty
		config.training_mode = not getattr(args, 'eval', False)
		
		# config.scenario_name = 'medium-gen-rgoal'
		# config.scenario_name = "nasim/scenarios/benchmark/tiny.yaml"
		config.scenario_name = args.scenario
		config.test_scenario_name = args.test_scenario
		config.eval_split = getattr(args, 'eval_split', 'both')
		
		# config.node_dim = 34
		# config.step_limit = 200
		config.step_limit = args.episode_step_limit
		config.use_a_t = args.use_a_t
		
		# config.scenario_name = 'huge-gen-rgoal'
		# config.node_dim = 43
		# config.step_limit = 400
		
		config.edge_dim = 0
		config.pos_enc_dim = 8
		
		config.fully_obs = args.fully_obs
		config.observation_format = args.observation_format
		config.augment_with_action = args.augment_with_action
		
		# auto scenario settings
		config.auto_mode = getattr(args, 'auto_mode', 'off')
		config.auto_template = getattr(args, 'auto_template', None)
		config.auto_host_range = getattr(args, 'auto_host_range', None)
		config.auto_subnet_count = getattr(args, 'auto_subnet_count', None)
		config.auto_topology = getattr(args, 'auto_topology', None)
		config.auto_sensitive_policy = getattr(args, 'auto_sensitive_policy', None)
		config.auto_seed_base = getattr(args, 'auto_seed_base', None)
		config.auto_sensitive_jitter = getattr(args, 'auto_sensitive_jitter', 0.0)
		# separate test template/mode if provided
		config.auto_template_test = getattr(args, 'auto_template_test', None)
		config.auto_mode_test = getattr(args, 'auto_mode_test', None)
			
		config.net_class = eval(args.net_class)
		
		graph_required_nets = {
			FeudalGTM,
			NASimNetGNN,
			NASimNetGNN_MAct,
			NASimNetGNN_LSTM,
			NASimNetDHRL,
		}
		graph_formats = {'graph', 'graph_v1', 'graph_v2'}
		if config.net_class in graph_required_nets and config.observation_format not in graph_formats:
			print(f"[NASimConfig] Forcing observation_format 'graph_v2' for {config.net_class.__name__} (got '{config.observation_format}')")
			config.observation_format = 'graph_v2'
			
		# config.net_class = NASimNetXAtt
		# config.net_class = NASimNetMLP
		# config.net_class = NASimNetInv
		# config.net_class = NASimNetInvMAct
		# config.net_class = NASimNetGNN
		# config.net_class = NASimNetGNN_LSTM
		
		# calculate number of actions
		env = NASimEmuEnv(
			scenario_name=config.scenario_name,
			augment_with_action=config.augment_with_action,
			training_mode=config.training_mode,  # Pass training_mode for curriculum learning
			curriculum_total_epochs=getattr(config, 'max_epochs', None),
			feature_dropout_p=getattr(args, 'feature_dropout_p', 0.0),
			dr_prob_jitter=getattr(args, 'dr_prob_jitter', 0.0),
			dr_cost_jitter=getattr(args, 'dr_cost_jitter', 0.0),
			dr_scan_cost_jitter=getattr(args, 'dr_scan_cost_jitter', 0.0),
			# auto generation plumbing (env will ignore until implemented)
			auto_mode=config.auto_mode,
			auto_template=config.auto_template,
			auto_host_range=config.auto_host_range,
			auto_subnet_count=config.auto_subnet_count,
			auto_topology=config.auto_topology,
			auto_sensitive_policy=config.auto_sensitive_policy,
			auto_seed_base=config.auto_seed_base,
			auto_sensitive_jitter=config.auto_sensitive_jitter,
		)
		s = env.reset()
		
		config.action_dim = len(env.action_list)
		config.node_dim = s.shape[1] + 1 # + 1 feature (node/subnet)
		
		# expose action metadata for masking in all nets
		config.fixed_scan_actions = 4  # ServiceScan, OSScan, SubnetScan, ProcessScan
		config.exploit_list = env.exploit_list  # list of (name, dict)
		config.privesc_list = env.privesc_list  # list of (name, dict)
		
		if config.net_class == BaselineAgent:
			BaselineAgent.action_list = [x[1]['name'] if 'name' in x[1] else None for x in env.action_list]	 # action ids
			
			BaselineAgent.exploit_list = env.exploit_list
			BaselineAgent.privesc_list = env.privesc_list
			
		# Exploit
		# PrivilegeEscalation
		# ServiceScan
		# OSScan
		# SubnetScan
		# ProcessScan
	
	@staticmethod
	def update_argparse(argparse):
		argparse.add_argument('scenario', type=str, help="Path to scenario to load. You can specify multiple scenarios with ':', just make sure that they share the same 'address_space_bounds'.")
		argparse.add_argument('-fully_obs', action='store_const', const=True, help="Use fully observable environment (default: False)")
		
		argparse.add_argument('-observation_format', type=str, default='list', help="list / graph")
		argparse.add_argument('-augment_with_action', action='store_const', const=True, help="Include the last action in observation (useful with LSTM)")
		argparse.add_argument('-net_class', type=str, default='NASimNetMLP', choices=['BaselineAgent', 'NASimNetMLP', 'NASimNetMLP_LSTM','FeudalGTM', 'NASimNetInv', 'NASimNetInvMAct', 'NASimNetInvMActTrainAT', 'NASimNetInvMActLSTM', 'NASimNetGNN', 'NASimNetGNN_MAct', 'NASimNetGNN_LSTM', 'NASimNetXAtt', 'NASimNetXAttMAct', 'NASimNetDHRL'])
		
		argparse.add_argument('-episode_step_limit', type=int, default=200, help="Force termination after number of steps")
		argparse.add_argument('-use_a_t', action='store_const', const=True, help="Enable agent to terminate the episode")
		
		argparse.add_argument('--test_scenario', type=str, help="Additional test scenarios to separately test the model (aka train/test datasets).")
		argparse.add_argument('--eval_split', choices=['both', 'train', 'test'], default='both', help="Select which evaluation split to run. 'test' requires --test_scenario and skips the positional training-scenario evaluation.")
		argparse.add_argument('--feature_dropout_p', type=float, default=0.0, help="Training-time feature dropout probability for observed service/process bits (0.0 to disable)")
		argparse.add_argument('--dr_prob_jitter', type=float, default=0.0, help="Per-episode multiplicative jitter for exploit/privesc probabilities (e.g., 0.1 -> ±10%%)")
		argparse.add_argument('--dr_cost_jitter', type=float, default=0.0, help="Per-episode multiplicative jitter for exploit/privesc/scan costs (rounded, min 1)")
		argparse.add_argument('--dr_scan_cost_jitter', type=float, default=0.0, help="Per-episode multiplicative jitter for scan costs (rounded, min 1)")
		
		# Auto scenario generation arguments (plumbing only for now)
		argparse.add_argument('--auto_mode', type=str, default='off', choices=['off', 'per_episode', 'per_epoch'], help="Auto-generate a fresh scenario per episode/epoch using a template")
		argparse.add_argument('--auto_template', type=str, default=None, help="Template YAML to copy OS/services/processes/exploits/privescs and costs")
		argparse.add_argument('--auto_host_range', type=str, default='72-96', help="Total non-internet hosts range, e.g., 72-96")
		argparse.add_argument('--auto_subnet_count', type=int, default=12, help="Number of subnets excluding internet (will always add internet subnet)")
		argparse.add_argument('--auto_topology', type=str, default='mesh', choices=['mesh', 'chain', 'random'], help="Topology generation strategy")
		argparse.add_argument('--auto_sensitive_policy', type=str, default='deep', choices=['deep', 'uniform'], help="Sensitive host placement policy")
		argparse.add_argument('--auto_seed_base', type=int, default=None, help="Base seed for auto generation (optional)")
		# Separate test-time auto controls (optional)
		argparse.add_argument('--auto_mode_test', type=str, default=None, choices=['off', 'per_episode', 'per_epoch'], help="Override auto_mode for test eval only")
		argparse.add_argument('--auto_template_test', type=str, default=None, help="Use a different template YAML for test eval only")
		# Sensitivity probability jitter
		argparse.add_argument('--auto_sensitive_jitter', type=float, default=0.0, help="Per-episode jitter for per-subnet sensitive host probabilities (e.g., 0.1 -> ±10%%)")
