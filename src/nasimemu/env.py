import gym, random, copy
import numpy as np
from nasimemu import nasim, env_utils

from nasimemu.nasim.envs.action import Exploit, PrivilegeEscalation, ServiceScan, OSScan, SubnetScan, ProcessScan, NoOp
import nasimemu.nasim.scenarios.benchmark as benchmark

from nasimemu.nasim.envs import NASimEnv
from nasimemu.nasim.envs.host_vector import HostVector
from nasimemu.nasim.envs.utils import AccessLevel
import traceback 
import nasimemu.nasim.scenarios.utils as u
from nasimemu.nasim.scenarios import Scenario
from nasimemu.nasim.scenarios.loader_v2 import ScenarioLoaderV2
from nasimemu.nasim.scenarios.host import Host

class TerminalAction():
    pass

# with deterministic exploits & privescs
class NASimScenarioGenerator(nasim.scenarios.generator.ScenarioGenerator):
    def _generate_exploits(self, num_exploits, exploit_cost, exploit_probs):
        rng = np.random.get_state()

        np.random.seed(12345)
        exploits = super()._generate_exploits(num_exploits, exploit_cost, exploit_probs)
        np.random.set_state(rng)

        return exploits

    def _generate_privescs(self, num_privesc, privesc_cost, privesc_probs):
        rng = np.random.get_state()

        np.random.seed(12346)
        privescs = super()._generate_privescs(num_privesc, privesc_cost, privesc_probs)
        np.random.set_state(rng)

        return privescs

class PartiallyObservableWrapper():
    def reset(self, s):
        self.__obs = dict()
        obs = self.__update_obs(s)
        return obs

    def step(self, s):
        obs = self.__update_obs(s)
        return obs

    def __update_obs(self, s):
        for host_data in s[:-1]:
            address = HostVector(host_data).address

            if address == (0, 0): # skip invalid entries
                continue

            if address in self.__obs:
                new_fields = host_data != 0
                self.__obs[address][new_fields] = host_data[new_fields]
            else:
                self.__obs[address] = host_data

        # construct and return new observation
        action_result = s[-1]
        obs = np.vstack([list(self.__obs.values()), action_result])

        return obs

# observation_format in ['matrix', 'graph']
class NASimEmuEnv(gym.Env):
    def __init__(self, scenario_name, step_limit=None, random_init=False, observation_format='matrix', fully_obs=False, augment_with_action=False, verbose=False, feature_dropout_p=0.0, dr_prob_jitter=0.0, dr_cost_jitter=0.0, dr_scan_cost_jitter=0.0,
                 training_mode=True,  # Curriculum learning: True=training, False=evaluation
                 curriculum_total_epochs=None,  # total run length, lets curriculum stages use start_frac/end_frac
                 # auto scenario generation (plumbing only; no behavior yet)
                 auto_mode='off', auto_template=None, auto_host_range=None, auto_subnet_count=None, auto_topology=None, auto_sensitive_policy=None, auto_seed_base=None, auto_sensitive_jitter=0.0,
                 seed=None):
        # Each SubprocVecEnv worker process constructs its envs after fork, so
        # without an explicit seed we must reseed from fresh OS entropy here --
        # otherwise every forked process inherits identical `random`/`np.random`
        # state and produces correlated/duplicate scenario draws across workers.
        # When an explicit seed IS given (deterministic-repro runs, one distinct
        # value per parallel env index), it must win outright: a bare no-arg
        # reseed() here would silently discard the caller's requested
        # determinism, which is exactly what the old "seed in multiprocessing is
        # not implemented" gap was.
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed % (2**32))
        else:
            random.seed()
            np.random.seed()

        self.step_limit = step_limit
        self.verbose = verbose
        self.fully_obs = fully_obs
        self.augment_with_action = augment_with_action
        self.observation_format = observation_format
        self.feature_dropout_p = float(feature_dropout_p or 0.0)
        self.dr_prob_jitter = float(dr_prob_jitter or 0.0)
        self.dr_cost_jitter = float(dr_cost_jitter or 0.0)
        self.dr_scan_cost_jitter = float(dr_scan_cost_jitter or 0.0)
        self.training_mode = training_mode  # For curriculum learning
        self.curriculum_total_epochs = curriculum_total_epochs  # for fractional curriculum stage bounds

        self.scenario_name = scenario_name
        self.random_init = random_init

        # store auto-generation params (unused for now)
        self.auto_mode = auto_mode or 'off'
        self.auto_template = auto_template
        self.auto_host_range = auto_host_range
        self.auto_subnet_count = auto_subnet_count
        self.auto_topology = auto_topology
        self.auto_sensitive_policy = auto_sensitive_policy
        self.auto_seed_base = auto_seed_base
        self.auto_sensitive_jitter = float(auto_sensitive_jitter or 0.0)

        # cache for template-derived fixed action space and metadata
        self._auto_cache = None
        self._epoch_scenario = None
        self._roll_on_next_reset = True
        self._action_signature = None

    # allow trainer to request a new scenario for next episode in per-epoch mode
    def set_roll_on_next_reset(self, flag=True):
        self._roll_on_next_reset = bool(flag)

    def _parse_range(self, s, fallback_min, fallback_max):
        try:
            a, b = str(s).split('-')
            return int(a), int(b)
        except Exception:
            return int(fallback_min), int(fallback_max)

    def _sample_subnet_sizes(self, count, per_min=6, per_max=8, total_min=72, total_max=96, max_tries=100):
        for _ in range(max_tries):
            sizes = [np.random.randint(per_min, per_max+1) for _ in range(count)]
            tot = sum(sizes)
            if total_min <= tot <= total_max:
                return sizes
        # fallback: clamp to closest total by adjusting first entries
        sizes = [per_min for _ in range(count)]
        tot = sum(sizes)
        idx = 0
        while tot < total_min and idx < count:
            inc = min(per_max - sizes[idx], total_min - tot)
            sizes[idx] += inc
            tot += inc
            idx += 1
        return sizes

    def _build_topology(self, num_subnets, style='mesh'):
        # num_subnets includes internet (index 0)
        topo = np.zeros((num_subnets, num_subnets), dtype=int)
        # connect each subnet to itself
        for i in range(num_subnets):
            topo[i][i] = 1
        # basic chain among 1..N-1
        for i in range(1, num_subnets-1):
            topo[i][i+1] = 1
            topo[i+1][i] = 1
        # internet connects to 1
        if num_subnets > 2:
            topo[0][1] = 1
            topo[1][0] = 1
        if style == 'mesh':
            # add cross links every 3rd node
            for i in range(1, num_subnets-3):
                if (i % 3) == 1:
                    topo[i][i+2] = 1
                    topo[i+2][i] = 1
            # connect last to a mid node
            topo[num_subnets-1][max(1, (num_subnets//2))] = 1
            topo[max(1, (num_subnets//2))][num_subnets-1] = 1
        elif style == 'random':
            # sprinkle a few random extra edges
            extra = max(2, (num_subnets//3))
            for _ in range(extra):
                i = np.random.randint(1, num_subnets)
                j = np.random.randint(1, num_subnets)
                if i != j:
                    topo[i][j] = 1
                    topo[j][i] = 1
        # leave as-is for 'chain'
        return topo.tolist()

    def _ensure_sensitive_service(self, services_map, sensitive_services, all_services):
        # if none of sensitive_services is enabled, force one that exists in catalog
        if not any(services_map.get(s, False) for s in sensitive_services):
            candidates = [s for s in sensitive_services if s in all_services]
            if candidates:
                s = random.choice(candidates)
                services_map[s] = True

    def _generate_firewall(self, subnets, hosts, services, exploits, restrictiveness=5, topology=None):
        num_subnets = len(subnets)
        firewall = {}
        # find services running on each subnet that are vulnerable
        subnet_services = {}
        subnet_services[u.INTERNET] = set()
        # helper: determine if host is vulnerable to exploit
        def host_vulnerable_to_e(host, e_def):
            e_srv = e_def[u.EXPLOIT_SERVICE]
            e_os = e_def[u.EXPLOIT_OS]
            if not host.services.get(e_srv, False):
                return False
            return (e_os is None) or host.os.get(e_os, False)
        for host_addr, host in hosts.items():
            subnet = host_addr[0]
            if subnet not in subnet_services:
                subnet_services[subnet] = set()
            for e_def in exploits.values():
                if host_vulnerable_to_e(host, e_def):
                    subnet_services[subnet].add(e_def[u.EXPLOIT_SERVICE])
        for src in range(num_subnets):
            for dest in range(num_subnets):
                if src == dest or (topology and not topology[src][dest]):
                    continue
                elif src > 2 and dest > 2:
                    # allow all services between user subnets
                    firewall[(src, dest)] = set(services)
                    continue
                dest_avail = subnet_services.get(dest, set()).copy()
                if len(dest_avail) <= restrictiveness:
                    firewall[(src, dest)] = dest_avail.copy()
                    continue
                # ensure at least one allowed, then sample up to restrictiveness
                allowed = set()
                if dest_avail:
                    first = random.choice(list(dest_avail))
                    allowed.add(first)
                    dest_avail.discard(first)
                while len(allowed) < restrictiveness and dest_avail:
                    s = random.choice(list(dest_avail))
                    allowed.add(s)
                    dest_avail.discard(s)
                firewall[(src, dest)] = allowed
        return {k: v for k, v in firewall.items()}

    def _generate_auto_from_template(self, template_path):
        # cache template-derived fixed parts
        if (self._auto_cache is None) or (self._auto_cache.get('template_path') != template_path):
            loader = ScenarioLoaderV2()
            sc_t = loader.load(template_path)
            yaml = loader.yaml_dict
            self._auto_cache = {
                'template_path': template_path,
                'os': list(sc_t.os),
                'services': list(sc_t.services),
                'processes': list(sc_t.processes),
                'exploits': copy.deepcopy(sc_t.exploits),
                'privescs': copy.deepcopy(sc_t.privescs),
                'service_scan_cost': sc_t.service_scan_cost,
                'os_scan_cost': sc_t.os_scan_cost,
                'subnet_scan_cost': sc_t.subnet_scan_cost,
                'process_scan_cost': sc_t.process_scan_cost,
                'address_space_bounds': sc_t.scenario_dict.get('address_space_bounds', None),
                'service_probabilities': yaml.get('service_probabilities', {}),
                'process_probabilities': yaml.get('process_probabilities', {}),
                'sensitive_services': yaml.get('sensitive_services', []),
                'sensitive_hosts_probs': yaml.get('sensitive_hosts', {}),
                'step_limit': self.step_limit,
                'scan_noise': yaml.get('scan_noise', {}),
                'service_dynamics': yaml.get('service_dynamics', {}),
                'network_reliability': yaml.get('network_reliability', {}),
                'curriculum': yaml.get('curriculum', {}),
                'intrusion_detection': yaml.get('intrusion_detection', {}),
            }

        T = self._auto_cache
        # determine host range and subnet count
        total_min, total_max = self._parse_range(self.auto_host_range or '72-96', 72, 96)
        subnet_count = int(self.auto_subnet_count or 12)
        # sample per-subnet sizes ~6-8 like corp/test, and ensure total within range
        sizes = self._sample_subnet_sizes(subnet_count, per_min=6, per_max=8, total_min=total_min, total_max=total_max)
        # include internet subnet at index 0
        subnets = [1] + sizes
        num_subnets = 1 + subnet_count
        # build topology
        style = (self.auto_topology or 'mesh')
        topology = self._build_topology(num_subnets, style=style)
        # sensitive host placement: use template probabilities per subnet (1..subnet_count)
        sensitive_hosts = {}
        for s_id in range(1, num_subnets):
            p0 = float(T['sensitive_hosts_probs'].get(s_id, T['sensitive_hosts_probs'].get(str(s_id), 0.0)))
            if self.auto_sensitive_jitter > 0.0:
                eps = np.random.uniform(-self.auto_sensitive_jitter, self.auto_sensitive_jitter)
                p = float(np.clip(p0 * (1.0 + eps), 0.0, 1.0))
            else:
                p = p0
            for h in range(subnets[s_id]):
                if np.random.rand() < p:
                    sensitive_hosts[(s_id, h)] = 10.0 if s_id == 2 else 10.0
        # generate hosts with probability maps
        hosts = {}
        services = T['services']
        processes = T['processes']
        os_list = T['os']
        srv_prob = {k: float(v) for k, v in T['service_probabilities'].items()}
        proc_prob = {k: float(v) for k, v in T['process_probabilities'].items()}
        sensitive_services = T['sensitive_services']
        for subnet, size in enumerate(subnets):
            if subnet == u.INTERNET:
                continue
            for h in range(size):
                os_choice = random.choice(os_list)
                os_map = {osn: (osn == os_choice) for osn in os_list}
                # sample services
                srv_map = {}
                for s_name in services:
                    p = srv_prob.get(s_name, 0.5)
                    srv_map[s_name] = (np.random.rand() < p)
                if not any(srv_map.values()):
                    # ensure at least one service
                    s_pick = random.choice(services)
                    srv_map[s_pick] = True
                # sample processes
                proc_map = {}
                for p_name in processes:
                    p = proc_prob.get(p_name, 0.2)
                    proc_map[p_name] = (np.random.rand() < p)
                # ensure at least one process
                if not any(proc_map.values()):
                    p_pick = random.choice(processes)
                    proc_map[p_pick] = True
                # ensure sensitive service for sensitive hosts
                if (subnet, h) in sensitive_hosts and sensitive_services:
                    self._ensure_sensitive_service(srv_map, sensitive_services, services)
                addr = (subnet, h)
                value = float(sensitive_hosts.get(addr, 1.0))
                host = Host(address=addr, os=os_map.copy(), services=srv_map.copy(), processes=proc_map.copy(), firewall={}, value=value, discovery_value=1)
                hosts[addr] = host
        # firewall
        fw = self._generate_firewall(subnets, hosts, services, T['exploits'], restrictiveness=5, topology=topology)
        # assemble scenario dict
        scenario_dict = {
            u.SUBNETS: subnets,
            u.TOPOLOGY: topology,
            u.OS: os_list,
            u.SERVICES: services,
            u.PROCESSES: processes,
            u.SENSITIVE_HOSTS: sensitive_hosts,
            u.EXPLOITS: copy.deepcopy(T['exploits']),
            u.PRIVESCS: copy.deepcopy(T['privescs']),
            u.OS_SCAN_COST: T['os_scan_cost'],
            u.SERVICE_SCAN_COST: T['service_scan_cost'],
            u.SUBNET_SCAN_COST: T['subnet_scan_cost'],
            u.PROCESS_SCAN_COST: T['process_scan_cost'],
            u.FIREWALL: fw,
            u.HOSTS: hosts,
            u.STEP_LIMIT: self.step_limit,
        }
        if T['address_space_bounds'] is not None:
            scenario_dict['address_space_bounds'] = T['address_space_bounds']
        # Add realism configurations from template
        if T['scan_noise']:
            scenario_dict['scan_noise'] = T['scan_noise']
        if T['service_dynamics']:
            scenario_dict['service_dynamics'] = T['service_dynamics']
        if T['network_reliability']:
            scenario_dict['network_reliability'] = T['network_reliability']
        if T['curriculum']:
            scenario_dict['curriculum'] = T['curriculum']
        if T['intrusion_detection']:
            scenario_dict['intrusion_detection'] = T['intrusion_detection']
        sc = Scenario(scenario_dict, name='auto_from_template', generated=True)
        return sc

    def _generate_env(self):
        if (self.auto_mode and self.auto_mode != 'off') and (self.auto_template and self.auto_template.endswith('.yaml')):
            if self.auto_mode == 'per_epoch':
                if (self._epoch_scenario is None) or self._roll_on_next_reset:
                    scenario = self._generate_auto_from_template(self.auto_template)
                    self._epoch_scenario = scenario
                    self._roll_on_next_reset = False
                else:
                    scenario = self._epoch_scenario
            else: # per_episode
                scenario = self._generate_auto_from_template(self.auto_template)
        else:
            if ':' in self.scenario_name: # there are multiple possible scenarios
                scenarios = self.scenario_name.split(':')
                scenario = random.choice(scenarios)
            else:
                scenario = self.scenario_name

            if scenario.endswith(".yaml"):        # static scenario
                # Scenarios with ranged/dynamic subnet sizes (e.g. "6-8")
                # rebuild a fresh random host-count instance on every load;
                # under heavy allocation churn this has been observed to hit
                # rare, non-reproducible corruption inside PyYAML's C-level
                # parsing state on this class of machine (manifests as
                # nonsensical exceptions -- AttributeError/UnboundLocalError/
                # TypeError -- deep in yaml/composer.py or yaml/scanner.py,
                # never on fixed-size scenarios). It doesn't reproduce twice
                # in a row, so retrying the load is a cheap, effective
                # mitigation; see docs/environment_setup_and_fixes.md.
                scenario_path = scenario
                for _attempt in range(3):
                    try:
                        scenario = nasim.load_scenario(scenario_path)
                        break
                    except Exception:
                        if _attempt == 2:
                            raise

            else:   # generated scenario
                scenario_params = benchmark.AVAIL_GEN_BENCHMARKS[scenario]
                scenario_params['step_limit'] = None

                generator = NASimScenarioGenerator()
                scenario = generator.generate(randomize_subnet_sizes=True, **scenario_params)

        # apply light per-episode jitter to improve robustness (training-time domain randomization)
        def _jitter_prob(p):
            if self.dr_prob_jitter <= 0.0:
                return p
            eps = np.random.uniform(-self.dr_prob_jitter, self.dr_prob_jitter)
            return float(np.clip(p * (1.0 + eps), 0.0, 1.0))
        def _jitter_cost(c):
            if self.dr_cost_jitter <= 0.0:
                return c
            eps = np.random.uniform(-self.dr_cost_jitter, self.dr_cost_jitter)
            c2 = int(max(1, round(c * (1.0 + eps))))
            return c2
        if self.dr_prob_jitter > 0.0 or self.dr_cost_jitter > 0.0:
            # exploits
            for e in scenario.exploits.values():
                e['prob'] = _jitter_prob(e['prob'])
                e['cost'] = _jitter_cost(e['cost'])
            # privesc
            for pe in scenario.privescs.values():
                pe['prob'] = _jitter_prob(pe['prob'])
                pe['cost'] = _jitter_cost(pe['cost'])
        if self.dr_scan_cost_jitter > 0.0:
            scenario.scenario_dict['service_scan_cost'] = _jitter_cost(scenario.scenario_dict['service_scan_cost'])
            scenario.scenario_dict['subnet_scan_cost'] = _jitter_cost(scenario.scenario_dict['subnet_scan_cost'])
            scenario.scenario_dict['process_scan_cost'] = _jitter_cost(scenario.scenario_dict['process_scan_cost'])
            scenario.scenario_dict['os_scan_cost'] = _jitter_cost(scenario.scenario_dict['os_scan_cost'])

        # Only create new env if it doesn't exist, otherwise just update scenario
        # This preserves curriculum state across resets
        if not hasattr(self, 'env') or self.env is None:
            self.env = NASimEnv(scenario, fully_obs=self.fully_obs, flat_actions=False, flat_obs=False,
                                training_mode=self.training_mode,
                                curriculum_total_epochs=self.curriculum_total_epochs)
        else:
            # Update scenario on existing env without recreating (preserves curriculum state)
            self.env.scenario = scenario
            # Reinitialize network and state with new scenario
            from nasimemu.nasim.envs.network import Network
            from nasimemu.nasim.envs.state import State
            from nasimemu.nasim.envs.action import FlatActionSpace, ParameterisedActionSpace
            self.env.network = Network(scenario)
            self.env.current_state = State.generate_initial_state(self.env.network)
            # Recreate action space for new scenario
            if self.env.flat_actions:
                self.env.action_space = FlatActionSpace(scenario)
            else:
                self.env.action_space = ParameterisedActionSpace(scenario)

        # Initialize realism parameters ONLY if curriculum is not managing them
        # If curriculum is active, it will handle these settings dynamically
        curriculum_active = (
            hasattr(self.env, 'curriculum_manager') and 
            self.env.curriculum_manager is not None and 
            self.env.curriculum_manager.is_enabled()
        )
        
        if not curriculum_active:
            # No curriculum - use static settings from scenario
            if hasattr(scenario, 'scan_noise'):
                from nasimemu.nasim.envs.host_vector import HostVector
                HostVector.set_scan_noise(scenario.scan_noise)

            if hasattr(scenario, 'service_dynamics'):
                from nasimemu.nasim.envs.host_vector import HostVector
                HostVector.set_churn_config(scenario.service_dynamics)

            if hasattr(scenario, 'network_reliability'):
                self.env.network.timeout_config = scenario.network_reliability
            
            if hasattr(scenario, 'intrusion_detection'):
                from nasimemu.nasim.envs.host_vector import HostVector
                HostVector.set_ids_config(scenario.intrusion_detection)
        # else: curriculum is active and will manage these settings via _apply_curriculum_settings()

        if not self.fully_obs:
            self.env_po_wrapper = PartiallyObservableWrapper()

        # self.edge_index = self._gen_edge_index()
        self.exploit_list, self.privesc_list, self.action_list = self._create_action_lists()
        self.action_cls = [x[0] for x  in self.action_list]

        # enforce fixed action space across resets for auto mode
        if self.auto_mode and self.auto_mode != 'off':
            exploit_names = [name for name, _ in self.exploit_list]
            privesc_names = [name for name, _ in self.privesc_list]
            signature = (tuple(exploit_names), tuple(privesc_names))
            if self._action_signature is None:
                self._action_signature = signature
            else:
                assert signature == self._action_signature, "Auto-generated scenario changed action space; ensure template OS/services/exploits/privescs remain constant."

        host_num_map = self.env.scenario.host_num_map
        self.host_index = np.array( sorted(host_num_map, key=host_num_map.get) )# fixed order node index
        self.subnet_index = np.array( [(x, -1) for x in range(len(self.env.scenario.subnets))] )

        self.discovered_subnets = set()
        self.subnet_graph = set() # (from, to)

    def _create_action_lists(self):
        exploit_list = sorted(self.env.scenario.exploits.items())
        privesc_list = sorted(self.env.scenario.privescs.items())

        action_list = [
                (ServiceScan, {'cost': self.env.scenario.service_scan_cost}),
                (OSScan, {'cost': self.env.scenario.os_scan_cost}),
                (SubnetScan, {'cost': self.env.scenario.subnet_scan_cost}),
                (ProcessScan, {'cost': self.env.scenario.process_scan_cost}),
        ]

        for exploit_name, exploit_params in exploit_list:
            action_list.append( (Exploit, {'name': exploit_name, **exploit_params}))

        for privesc_name, privesc_params in privesc_list:
            action_list.append( (PrivilegeEscalation, {'name': privesc_name, **privesc_params}))

        return exploit_list, privesc_list, action_list

    def _translate_action(self, action):
        target, action_id = action

        if action_id == -1: # terminal action
            return TerminalAction()

        a_class, a_params = self.action_list[action_id]
        # print(a_class, a_params, target, action_id)
        a = a_class(target=tuple(target), **a_params)

        assert a in self.env.action_space.actions, "Failed to execute " + str(a)

        return a

    def _get_subnets(self, s):
        return {HostVector(x).address[0] for x in s[:-1]}

    def _augment_with_action(self, s, action):
        action_matrix = np.zeros((len(s), len(self.action_list)), dtype=np.float32)

        if action is not None:
            target, action_id = action

            host_addresses = [HostVector(host_data).address for host_data in s[:-1]]
            host_index = host_addresses.index(tuple(target))

            action_matrix[host_index, action_id] = 1.

        s = np.concatenate((s, action_matrix), axis=1)

        return s

    def _apply_feature_dropout(self, s):
        if self.feature_dropout_p <= 0.0:
            return s
        # apply dropout only to host rows (exclude last action-result row)
        out = s.copy()
        srv_slice = HostVector._service_idx_slice()
        proc_slice = HostVector._process_idx_slice()
        # Bernoulli masks per row and per feature
        for i in range(len(out)-1):
            # Dropout services
            if srv_slice.stop <= out.shape[1]:
                mask_srv = (np.random.rand(srv_slice.stop - srv_slice.start) >= self.feature_dropout_p).astype(out.dtype)
                out[i, srv_slice] *= mask_srv
            # Dropout processes
            if proc_slice.stop <= out.shape[1]:
                mask_proc = (np.random.rand(proc_slice.stop - proc_slice.start) >= self.feature_dropout_p).astype(out.dtype)
                out[i, proc_slice] *= mask_proc
        return out

    # action = ((subnet, host), action_id)
    def step(self, action):
        if type(action) in [np.ndarray, list, tuple]:
            a = self._translate_action(action)
        else:
            a = action

        if isinstance(a, TerminalAction):
            s, r, d, i = self.env.step(NoOp())
            r = 0.
            d = True

        else:
            s, r, d, i = self.env.step(a)
            d = False # ignore done flag from the environment, the agent has to choose to terminate

        r /= 10. # reward scaling
        self.r_tot += r

        # Record each distinct sensitive host captured during the episode:
        # count a capture only on the state transition that first grants ROOT access on a
        # *distinct* sensitive host, not on any step with r > 0 (which also fires for
        # discovery-value rewards etc. and could double-count). self._captured_addrs is
        # reset per-episode in reset().
        for addr in self.env.network.sensitive_hosts.keys():
            if addr not in self._captured_addrs and self.env.current_state.host_has_access(addr, AccessLevel.ROOT):
                self._captured_addrs.add(addr)
                self.captured += 1

        if not self.fully_obs:
            s = self.env_po_wrapper.step(s)

        # optionally apply feature dropout on observed service/process bits (training-time robustness)
        s = self._apply_feature_dropout(s)

        # track newly discovered subnets and remember the origin
        if isinstance(a, SubnetScan):
            s_subnets = self._get_subnets(s)
            new_subnets = s_subnets - self.discovered_subnets
            origin_subnet = a.target[0]

            for new_subnet in new_subnets:
                self.subnet_graph.add( (origin_subnet, new_subnet) )

            self.discovered_subnets = s_subnets

        self.step_idx += 1

        if self.verbose:
            print("===========================================")
            print(f"Step: {self.step_idx}")
            # print(f"Raw state: \n {s}")
            print(f"Action: {a}")
            print(f"R: {r} D: {d}")
            print(f"Info: {i}")

            self.env.render(obs=s)
            # self.env.render() # show observation

        self.s_raw = s # before any augmentation / transformation (e.g., into graph)

        i['s_raw'] = s
        i['a_raw'] = a

        # optionally, include the last performed action
        if self.augment_with_action:
            s = self._augment_with_action(s, action)

        # optionally, convert the observation into a graph
        if self.observation_format in ['graph', 'graph_v1']:
            s = env_utils.convert_to_graph(s, self.subnet_graph, version=1)

        if self.observation_format == 'graph_v2':
            s = env_utils.convert_to_graph(s, self.subnet_graph, version=2)

        i['s_true'] = s
        i['d_true'] = d
        i['step_idx'] = self.step_idx
        i['subnet_graph'] = self.subnet_graph
        i['r_tot'] = self.r_tot
        i['captured'] = self.captured

        if (self.step_limit is not None) and (self.step_idx >= self.step_limit):
            d = True

        if d:
            s = self.reset()

        i['done'] = d
        i['d_true'] = d # fix: this will disable the difference between true termination and step_limit exceedance - both are treated the same

        return s, r, d, i

    def reset(self):
        self.step_idx = 0
        self.r_tot = 0.
        self.captured = 0
        self._captured_addrs = set()

        self._generate_env() # generate new env

        s = self.env.reset()

        if not self.fully_obs:
            s = self.env_po_wrapper.reset(s)

        # apply feature dropout at reset as well
        s = self._apply_feature_dropout(s)

        self.s_raw = s

        self.discovered_subnets = self._get_subnets(s)
        self.subnet_graph = set()

        # optionally, include the last performed action (zeros in reset())
        if self.augment_with_action:
            s = self._augment_with_action(s, None)


        if self.observation_format in ['graph', 'graph_v1']:
            s = env_utils.convert_to_graph(s, self.subnet_graph, version=1)

        if self.observation_format == 'graph_v2':
            s = env_utils.convert_to_graph(s, self.subnet_graph, version=2)

        # break the tie with random offset
        if self.random_init:
            self.step_idx = np.random.randint(self.step_limit)
            self.random_init = False

        if self.verbose:
            print()
            print("-------")
            print("reset()")
            print("-------")

            self.env.render_state()
            self.env.render(obs=s)

        return s
    
    def set_epoch(self, epoch_num):
        """Set current epoch number for curriculum learning.
        
        This should be called at each epoch to update the curriculum stage.
        It will immediately apply the new curriculum settings if the stage changes.
        
        Parameters
        ----------
        epoch_num : int
            The current epoch number
        """
        if hasattr(self.env, 'update_curriculum_epoch'):
            self.env.update_curriculum_epoch(epoch_num)
    
    def get_curriculum_info(self):
        """Get curriculum information from the underlying environment.
        
        Returns
        -------
        dict or None
            Dictionary with curriculum stage info and realism parameters,
            or None if curriculum is not enabled.
        """
        if hasattr(self.env, 'get_curriculum_info'):
            stage_info = self.env.get_curriculum_info()
            if stage_info:
                # Also get the realism parameters
                if hasattr(self.env, 'curriculum_manager'):
                    params = self.env.curriculum_manager.get_realism_params()
                    stage_info['realism_params'] = params
                return stage_info
        return None
    
    def get_stage_transition_epochs(self):
        """Get list of epochs where curriculum stages transition.
        
        Returns
        -------
        list of int
            Sorted list of epoch numbers where stage transitions occur,
            or empty list if curriculum is not enabled.
        """
        if hasattr(self.env, 'curriculum_manager') and self.env.curriculum_manager is not None:
            return self.env.curriculum_manager.get_stage_transition_epochs()
        return []
    
    def get_actual_realism_params(self):
        """Get ACTUAL realism parameters currently set in the environment.
        
        Queries the actual class variables (HostVector, Network) to get
        what's currently active, not just what's in the YAML config.
        
        Returns
        -------
        dict
            Dictionary with actual settings for IDS, scan noise, network reliability,
            and service dynamics.
        """
        try:
            from nasimemu.nasim.envs.host_vector import HostVector
            
            # Get actual values from HostVector class variables
            return {
                'ids_config': getattr(HostVector, 'ids_config', None),
                'scan_noise': getattr(HostVector, 'scan_noise_config', None),
                'service_dynamics': getattr(HostVector, 'churn_config', None),
                'network_reliability': getattr(self.env.network, 'timeout_config', None) if hasattr(self.env, 'network') else None
            }
        except Exception as e:
            return {}

    def render(self, s):
        self.env.render(obs=s)

    def render_state(self):
        self.env.render_state()
