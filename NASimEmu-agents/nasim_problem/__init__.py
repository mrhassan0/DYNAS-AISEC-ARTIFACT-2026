from nasimemu.env import NASimEmuEnv

from .nasim_debug import NASimDebug
from .nasim_config import NASimConfig

from config import config
import gym

class NASimRRL():
	@staticmethod
	def make_env():
		return NASimEmuEnv()

	@staticmethod
	def make_net():
		return config.net_class()

	@staticmethod
	def make_debug():
		return NASimDebug()

	@staticmethod
	def make_config():
		return NASimConfig()

	@staticmethod
	def register_gym():
		gym.envs.registration.register(
			id='NASimEmuEnv-v99',
			entry_point='nasimemu.env:NASimEmuEnv',
			kwargs={'scenario_name': config.scenario_name, 'step_limit': config.step_limit, 
				'fully_obs': config.fully_obs, 'observation_format': config.observation_format,
				'augment_with_action': config.augment_with_action, 'random_init': True, 'verbose': False,
				'training_mode': getattr(config, 'training_mode', True),  # Curriculum: True=training, False=evaluation
				'curriculum_total_epochs': getattr(config, 'max_epochs', None),  # lets curriculum stages ramp via start_frac/end_frac
				# auto scenario kwargs (env must accept them; no behavior yet)
				'auto_mode': getattr(config, 'auto_mode', 'off'),
				'auto_template': getattr(config, 'auto_template', None),
				'auto_host_range': getattr(config, 'auto_host_range', None),
				'auto_subnet_count': getattr(config, 'auto_subnet_count', None),
				'auto_topology': getattr(config, 'auto_topology', None),
				'auto_sensitive_policy': getattr(config, 'auto_sensitive_policy', None),
				'auto_seed_base': getattr(config, 'auto_seed_base', None),
				'auto_sensitive_jitter': getattr(config, 'auto_sensitive_jitter', 0.0),
			}
		)

	@staticmethod
	def get_gym_name():
		return 'NASimEmuEnv-v99'

	@staticmethod
	def get_project_name():
		return 'rrl-nasim'

	@staticmethod
	def get_run_name():
		return None
