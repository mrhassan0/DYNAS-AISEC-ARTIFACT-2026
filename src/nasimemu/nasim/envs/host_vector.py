""" This module contains the HostVector class.

This is the main class for storing and updating the state of a single host
in the NASim environment.
"""

import numpy as np

from .utils import AccessLevel
from .action import ActionResult


class HostVector:
    """ A Vector representation of a single host in NASim.

    Each host is represented as a vector (1D numpy array) for efficiency and to
    make it easier to use with deep learning agents. The vector is made up of
    multiple features arranged in a consistent way.

    Features in the vector, listed in order, are:

    1. subnet address - one-hot encoding with length equal to the number
                        of subnets
    2. host address - one-hot encoding with length equal to the maximum number
                      of hosts in any subnet
    3. compromised - bool
    4. reachable - bool
    5. discovered - bool
    6. value - float
    7. discovery value - float
    8. access - int
    9. OS - bool for each OS in scenario (only one OS has value of true)
    10. services running - bool for each service in scenario
    11. processes running - bool for each process in scenario

    Notes
    -----
    - The size of the vector is equal to:

        #subnets + max #hosts in any subnet + 6 + #OS + #services + #processes.

    - Where the +6 is for compromised, reachable, discovered, value,
      discovery_value, and access features
    - The vector is a float vector so True/False is actually represented as
      1.0/0.0.

    """

    # class properties that are the same for all hosts
    # these are set when calling vectorize method
    # the bounds on address space (used for one hot encoding of host address)
    address_space_bounds = None
    # number of OS in scenario
    num_os = None
    # map from OS name to its index in host vector
    os_idx_map = {}
    # number of services in scenario
    num_services = None
    # map from service name to its index in host vector
    service_idx_map = {}
    # number of processes in scenario
    num_processes = None
    # map from process name to its index in host vector
    process_idx_map = {}
    # size of state for host vector (i.e. len of vector)
    state_size = None
    
    # scan noise configuration
    scan_noise = {
        'service_scan': {'false_positive_rate': 0.0, 'false_negative_rate': 0.0},
        'os_scan': {'false_positive_rate': 0.0, 'false_negative_rate': 0.0},
        'process_scan': {'false_positive_rate': 0.0, 'false_negative_rate': 0.0}
    }
    
    # service dynamics configuration
    churn_config = {
        'churn_probability': 0.0,
        'affected_services': [],
        'restart_delay': 10,
        'churn_types': [
            {
                'type': 'crash_restart',
                'probability': 1.0,
                'down_duration': [5, 15]
            }
        ]
    }
    
    # IDS (Intrusion Detection System) configuration
    ids_config = {
        'enabled': False,
        'detection_decay': 0.98,
        'base_thresholds': [0.7, 0.8],
        'response_types': {
            'quarantine': 0.2,
            'patch': 0.4,
            'monitor': 0.4
        },
        'failed_exploit_multiplier': 3.0,
        'detection_increase': {
            'subnet_scan': 0.02,
            'service_scan': 0.05,
            'os_scan': 0.03,
            'process_scan': 0.03,
            'exploit_failed': 0.15,
            'exploit_success': 0.08,
            'privesc_failed': 0.20,
            'privesc_success': 0.10,
        }
    }

    # vector position constants
    # to be initialized
    _subnet_address_idx = 0
    _host_address_idx = None
    _compromised_idx = None
    _reachable_idx = None
    _discovered_idx = None
    _value_idx = None
    _discovery_value_idx = None
    _access_idx = None
    _os_start_idx = None
    _service_start_idx = None
    _process_start_idx = None
    _detection_level_idx = None
    _detection_threshold_idx = None
    _detection_multiplier_idx = None

    def __init__(self, vector):
        self.vector = vector
        # Add service state tracking for churn
        self.service_states = {}  # service_name -> {'status': 'up'/'down', 'down_until': step}
        
        # IDS tracking (instance variables, not in vector for now)
        self.detection_level = 0.0  # 0.0 = undetected, 1.0 = fully detected
        self.last_scan_time = -1000  # Steps since last scan
        self.failed_exploit_count = 0
        # Random threshold for when detection triggers alert
        threshold_range = self.ids_config.get('base_thresholds', [0.7, 0.8])
        self.detection_threshold = np.random.uniform(threshold_range[0], threshold_range[1])
        self.patched_services = set()  # Services that have been patched after detection
        self.detection_multiplier = 1.0  # Increased monitoring after minor detection

    @classmethod
    def vectorize(cls, host, address_space_bounds, vector=None):
        if cls.address_space_bounds is None:
            cls._initialize(
                address_space_bounds, host.services, host.os, host.processes
            )

        if vector is None:
            vector = np.zeros(cls.state_size, dtype=np.float32)
        else:
            assert len(vector) == cls.state_size

        vector[cls._subnet_address_idx + host.address[0]] = 1
        vector[cls._host_address_idx + host.address[1]] = 1
        vector[cls._compromised_idx] = int(host.compromised)
        vector[cls._reachable_idx] = int(host.reachable)
        vector[cls._discovered_idx] = int(host.discovered)
        vector[cls._value_idx] = host.value
        vector[cls._discovery_value_idx] = host.discovery_value
        vector[cls._access_idx] = host.access
        for os_num, (os_key, os_val) in enumerate(host.os.items()):
            vector[cls._get_os_idx(os_num)] = int(os_val) # TODO we should determine os by os_key, not os_num
        for srv_num, (srv_key, srv_val) in enumerate(host.services.items()):
            vector[cls._get_service_idx(srv_num)] = int(srv_val) # TODO we should determine service by srv_key, not srv_num
        host_procs = host.processes.items()
        for proc_num, (proc_key, proc_val) in enumerate(host_procs):
            vector[cls._get_process_idx(proc_num)] = int(proc_val) # TODO we should determine process by proc_key, not proc_num
        detection_threshold = host.detection_threshold
        if detection_threshold is None:
            threshold_range = cls.ids_config.get('base_thresholds', [0.7, 0.8])
            detection_threshold = np.random.uniform(threshold_range[0], threshold_range[1])
        vector[cls._detection_level_idx] = host.detection_level
        vector[cls._detection_threshold_idx] = detection_threshold
        vector[cls._detection_multiplier_idx] = host.detection_multiplier
        return cls(vector)

    @classmethod
    def vectorize_random(cls, host, address_space_bounds, vector=None):
        hvec = cls.vectorize(host, vector)
        # random variables
        for srv_num in cls.service_idx_map.values():
            srv_val = np.random.randint(0, 2)
            hvec.vector[cls._get_service_idx(srv_num)] = srv_val

        chosen_os = np.random.choice(list(cls.os_idx_map.values()))
        for os_num in cls.os_idx_map.values():
            hvec.vector[cls._get_os_idx(os_num)] = int(os_num == chosen_os)

        for proc_num in cls.process_idx_map.values():
            proc_val = np.random.randint(0, 2)
            hvec.vector[cls._get_process_idx(proc_num)] = proc_val
        return hvec

    @property
    def compromised(self):
        return self.vector[self._compromised_idx]

    @compromised.setter
    def compromised(self, val):
        self.vector[self._compromised_idx] = int(val)

    @property
    def discovered(self):
        return self.vector[self._discovered_idx]

    @discovered.setter
    def discovered(self, val):
        self.vector[self._discovered_idx] = int(val)

    @property
    def reachable(self):
        return self.vector[self._reachable_idx]

    @reachable.setter
    def reachable(self, val):
        self.vector[self._reachable_idx] = int(val)

    @property
    def address(self):
        return (
            self.vector[self._subnet_address_idx_slice()].argmax(),
            self.vector[self._host_address_idx_slice()].argmax()
        )

    @property
    def value(self):
        return self.vector[self._value_idx]

    @property
    def discovery_value(self):
        return self.vector[self._discovery_value_idx]

    @property
    def access(self):
        return self.vector[self._access_idx]

    @access.setter
    def access(self, val):
        self.vector[self._access_idx] = int(val)

    @property
    def services(self):
        services = {}
        for srv, srv_num in self.service_idx_map.items():
            services[srv] = self.vector[self._get_service_idx(srv_num)]
        return services

    @property
    def os(self):
        os = {}
        for os_key, os_num in self.os_idx_map.items():
            os[os_key] = self.vector[self._get_os_idx(os_num)]
        return os

    @property
    def processes(self):
        processes = {}
        for proc, proc_num in self.process_idx_map.items():
            processes[proc] = self.vector[self._get_process_idx(proc_num)]
        return processes

    def is_running_service(self, srv):
        srv_num = self.service_idx_map[srv]
        return bool(self.vector[self._get_service_idx(srv_num)])

    def is_running_os(self, os):
        os_num = self.os_idx_map[os]
        return bool(self.vector[self._get_os_idx(os_num)])

    def is_running_process(self, proc):
        proc_num = self.process_idx_map[proc]
        return bool(self.vector[self._get_process_idx(proc_num)])

    @classmethod
    def set_scan_noise(cls, noise_config):
        """Set scan noise configuration from scenario"""
        cls.scan_noise = noise_config

    @classmethod
    def set_churn_config(cls, config):
        """Set service churn configuration"""
        cls.churn_config = config
    
    @classmethod
    def set_ids_config(cls, config):
        """Set IDS configuration from scenario"""
        cls.ids_config = config
    
    def _apply_scan_noise(self, mapping, scan_type):
        """Apply noise to scan results"""
        noise_config = self.scan_noise.get(scan_type, {})
        fp_rate = noise_config.get('false_positive_rate', 0.0)
        fn_rate = noise_config.get('false_negative_rate', 0.0)
        
        noisy = dict(mapping)  # copy
        for k, v in noisy.items():
            r = np.random.rand()
            if v and r < fn_rate:
                noisy[k] = False  # false negative
            elif (not v) and r < fp_rate:
                noisy[k] = True   # false positive
        return noisy

    def update_service_churn(self, current_step):
        """Update service states based on churn probability"""
        if not self.churn_config or self.churn_config.get('churn_probability', 0.0) == 0.0:
            return
        
        churn_prob = self.churn_config.get('churn_probability', 0.0)
        affected_services = self.churn_config.get('affected_services', [])
        
        for service in affected_services:
            if service not in self.service_idx_map:
                continue
            
            # Check if service should go down
            if (self.is_running_service(service) and  # service is currently up
                np.random.rand() < churn_prob):
                self._service_goes_down(service, current_step)
            
            # Check if service should come back up
            if (service in self.service_states and
                self.service_states[service]['status'] == 'down' and
                current_step >= self.service_states[service]['down_until']):
                self._service_comes_up(service)
    
    def _service_goes_down(self, service, current_step):
        """Mark service as down"""
        churn_types = self.churn_config.get('churn_types', [])
        if not churn_types:
            # Default: simple restart
            down_duration = self.churn_config.get('restart_delay', 10)
        else:
            # Choose churn type based on probabilities
            rand = np.random.rand()
            cum_prob = 0
            down_duration = 10  # default
            for churn_type in churn_types:
                cum_prob += churn_type['probability']
                if rand < cum_prob:
                    duration_range = churn_type['down_duration']
                    if isinstance(duration_range, list):
                        down_duration = np.random.randint(duration_range[0], duration_range[1] + 1)
                    else:
                        down_duration = duration_range
                    break
        
        self.service_states[service] = {
            'status': 'down',
            'down_until': current_step + down_duration
        }
        # Update the actual service state in the vector
        srv_num = self.service_idx_map[service]
        self.vector[self._get_service_idx(srv_num)] = 0.0
    
    def _service_comes_up(self, service):
        """Mark service as up"""
        if service in self.service_states:
            del self.service_states[service]
        # Update the actual service state in the vector
        srv_num = self.service_idx_map[service]
        self.vector[self._get_service_idx(srv_num)] = 1.0

    def update_detection(self, action, success, current_step):
        """Update IDS detection level based on action
        
        Arguments
        ---------
        action : Action
            the action that was performed
        success : bool
            whether the action was successful
        current_step : int
            current environment step number
            
        Returns
        -------
        tuple (str, dict or None)
            ('DETECTED', detection_response) if detected, ('SAFE', None) otherwise
        """
        if not self.ids_config.get('enabled', False):
            return 'SAFE', None
        
        detection_increase_map = self.ids_config.get('detection_increase', {})
        
        # Determine action type and base detection increase
        increase = 0.0
        
        if action.is_scan():
            if action.is_subnet_scan():
                increase = detection_increase_map.get('subnet_scan', 0.02)
            elif action.is_service_scan():
                increase = detection_increase_map.get('service_scan', 0.05)
                # Rapid scanning is more suspicious
                time_since_last = current_step - self.last_scan_time
                if time_since_last < 5:
                    increase *= 2.0
                self.last_scan_time = current_step
            elif action.is_os_scan():
                increase = detection_increase_map.get('os_scan', 0.03)
                time_since_last = current_step - self.last_scan_time
                if time_since_last < 5:
                    increase *= 2.0
                self.last_scan_time = current_step
            elif action.is_process_scan():
                increase = detection_increase_map.get('process_scan', 0.03)
                time_since_last = current_step - self.last_scan_time
                if time_since_last < 5:
                    increase *= 2.0
                self.last_scan_time = current_step
        elif action.is_exploit():
            if not success:
                # Failed exploits are VERY noisy
                self.failed_exploit_count += 1
                increase = detection_increase_map.get('exploit_failed', 0.15)
                # Multiple failed attempts trigger IDS faster
                multiplier = self.ids_config.get('failed_exploit_multiplier', 3.0)
                increase *= (1 + 0.3 * self.failed_exploit_count * multiplier)
            else:
                increase = detection_increase_map.get('exploit_success', 0.08)
        elif action.is_privilege_escalation():
            if not success:
                increase = detection_increase_map.get('privesc_failed', 0.20)
                self.failed_exploit_count += 1
            else:
                increase = detection_increase_map.get('privesc_success', 0.10)
        
        # Apply detection multiplier from previous monitoring
        increase *= self.detection_multiplier
        
        # Update detection level
        self.detection_level += increase
        
        # Natural decay (IDS logs rotate, alerts fade)
        decay_rate = self.ids_config.get('detection_decay', 0.98)
        self.detection_level *= decay_rate
        self.vector[self._detection_level_idx] = self.detection_level

        # Check if host triggers alert
        if self.detection_level > self.detection_threshold:
            return 'DETECTED', self._handle_detection()
        
        return 'SAFE', None
    
    def _handle_detection(self):
        """Handle what happens when IDS detects intrusion
        
        Returns
        -------
        dict
            Dictionary containing detection response information
        """
        response_types = self.ids_config.get('response_types', {
            'quarantine': 0.2,
            'patch': 0.4,
            'monitor': 0.4
        })
        
        response_roll = np.random.random()
        
        # Determine response based on probabilities
        quarantine_prob = response_types.get('quarantine', 0.2)
        patch_prob = response_types.get('patch', 0.4)
        
        if response_roll < quarantine_prob:
            # Severe: Host quarantined, all access lost
            return {
                'type': 'quarantine',
                'penalty': -50,
                'message': 'Host has been quarantined by IDS'
            }
        elif response_roll < quarantine_prob + patch_prob:
            # Moderate: Service patched, exploit blocked
            vulnerable_services = []
            for srv_name, srv_num in self.service_idx_map.items():
                if self.vector[self._get_service_idx(srv_num)] and srv_name not in self.patched_services:
                    vulnerable_services.append(srv_name)
            
            patched = []
            if vulnerable_services:
                # Patch 1-2 random vulnerable services
                num_to_patch = min(np.random.randint(1, 3), len(vulnerable_services))
                patched = np.random.choice(vulnerable_services, num_to_patch, replace=False).tolist()
                for srv in patched:
                    self.patched_services.add(srv)
            
            return {
                'type': 'patch',
                'penalty': -20,
                'patched_services': patched,
                'message': f'IDS patched services: {patched}'
            }
        else:
            # Minor: Increased monitoring (future actions riskier)
            self.detection_multiplier = 2.0
            self.vector[self._detection_multiplier_idx] = self.detection_multiplier
            return {
                'type': 'monitor',
                'penalty': -5,
                'detection_multiplier': 2.0,
                'message': 'Host under increased IDS monitoring'
            }

    def perform_action(self, action):
        """Perform given action against this host

        Arguments
        ---------
        action : Action
            the action to perform

        Returns
        -------
        HostVector
            the resulting state of host after action
        ActionObservation
            the result from the action
        """
        next_state = self.copy()
        if action.is_service_scan():
            noisy_services = self._apply_scan_noise(self.services, 'service_scan')
            result = ActionResult(True, 0, services=noisy_services)
            return next_state, result

        if action.is_os_scan():
            noisy_os = self._apply_scan_noise(self.os, 'os_scan')
            return next_state, ActionResult(True, 0, os=noisy_os)

        if action.is_exploit():
            # Check if service has been patched by IDS
            if action.service in self.patched_services:
                result = ActionResult(False, 0, undefined_error=True)
                return next_state, result
            
            if self.is_running_service(action.service) and \
               (action.os is None or self.is_running_os(action.os)):
                # service and os is present so exploit is successful
                value = 0
                next_state.compromised = True
                if not self.access == AccessLevel.ROOT:
                    # ensure a machine is not rewarded twice
                    # and access doesn't decrease
                    next_state.access = action.access
                    if action.access == AccessLevel.ROOT:
                        value = self.value

                result = ActionResult(
                    True,
                    value=value,
                    services=self.services,
                    os=self.os,
                    access=action.access
                )
                return next_state, result

        # following actions are on host so require correct access
        if not self.compromised and action.req_access <= self.access:
            result = ActionResult(False, 0, permission_error=True)
            return next_state, result

        if action.is_process_scan():
            noisy_processes = self._apply_scan_noise(self.processes, 'process_scan')
            result = ActionResult(
                True, 0, access=self.access, processes=noisy_processes
            )
            return next_state, result

        if action.is_privilege_escalation():
            has_proc = (
                action.process is None
                or self.is_running_process(action.process)
            )
            has_os = (
                action.os is None or self.is_running_os(action.os)
            )
            if has_proc and has_os:
                # host compromised and proc and os is present
                # so privesc is successful
                value = 0.0
                if not self.access == AccessLevel.ROOT:
                    # ensure a machine is not rewarded twice
                    # and access doesn't decrease
                    next_state.access = action.access
                    if action.access == AccessLevel.ROOT:
                        value = self.value
                result = ActionResult(
                    True,
                    value=value,
                    processes=self.processes,
                    os=self.os,
                    access=action.access
                )
                return next_state, result

        # action failed due to host config not meeting preconditions
        return next_state, ActionResult(False, 0)

    def observe(self,
                address=False,
                compromised=False,
                reachable=False,
                discovered=False,
                access=False,
                value=False,
                discovery_value=False,
                services=False,
                processes=False,
                os=False):
        obs = np.zeros(self.state_size, dtype=np.float32)
        if address:
            subnet_slice = self._subnet_address_idx_slice()
            host_slice = self._host_address_idx_slice()
            obs[subnet_slice] = self.vector[subnet_slice]
            obs[host_slice] = self.vector[host_slice]
        if compromised:
            obs[self._compromised_idx] = self.vector[self._compromised_idx]
        if reachable:
            obs[self._reachable_idx] = self.vector[self._reachable_idx]
        if discovered:
            obs[self._discovered_idx] = self.vector[self._discovered_idx]
        if value:
            obs[self._value_idx] = self.vector[self._value_idx]
        if discovery_value:
            v = self.vector[self._discovery_value_idx]
            obs[self._discovery_value_idx] = v
        if access:
            obs[self._access_idx] = self.vector[self._access_idx]
        if os:
            idxs = self._os_idx_slice()
            obs[idxs] = self.vector[idxs]
        if services:
            idxs = self._service_idx_slice()
            obs[idxs] = self.vector[idxs]
        if processes:
            idxs = self._process_idx_slice()
            obs[idxs] = self.vector[idxs]
        # IDS state: whenever a host is observed at all, reveal how "hot" it is
        # (accumulated detection level) and whether it is under increased
        # monitoring. Without this the agent is penalised by the IDS through the
        # reward but has NO observable signal for its own detection footprint,
        # so it cannot learn a stealth policy (it only ever sees these columns
        # as 0). The hidden per-host detection *threshold* is deliberately NOT
        # exposed -- that uncertainty is what makes stealth non-trivial. These
        # columns already exist in the state vector, so exposing them does not
        # change the observation dimensionality (existing checkpoints stay
        # compatible; the columns are simply no longer forced to 0).
        if self._detection_level_idx is not None:
            obs[self._detection_level_idx] = self.vector[self._detection_level_idx]
        if self._detection_multiplier_idx is not None:
            obs[self._detection_multiplier_idx] = self.vector[self._detection_multiplier_idx]
        return obs

    def readable(self):
        return self.get_readable(self.vector)

    def copy(self):
        vector_copy = np.copy(self.vector)
        new_host = HostVector(vector_copy)
        # Preserve service state tracking
        new_host.service_states = dict(self.service_states)
        # Preserve IDS state
        new_host.detection_level = self.detection_level
        new_host.last_scan_time = self.last_scan_time
        new_host.failed_exploit_count = self.failed_exploit_count
        new_host.detection_threshold = self.detection_threshold
        new_host.patched_services = set(self.patched_services)
        new_host.detection_multiplier = self.detection_multiplier
        return new_host

    def numpy(self):
        return self.vector

    @classmethod
    def _initialize(cls, address_space_bounds, services, os_info, processes):
        cls.os_idx_map = {}
        cls.service_idx_map = {}
        cls.process_idx_map = {}
        cls.address_space_bounds = address_space_bounds
        cls.num_os = len(os_info)
        cls.num_services = len(services)
        cls.num_processes = len(processes)
        cls._update_vector_idxs()
        for os_num, (os_key, os_val) in enumerate(os_info.items()):
            cls.os_idx_map[os_key] = os_num
        for srv_num, (srv_key, srv_val) in enumerate(services.items()):
            cls.service_idx_map[srv_key] = srv_num
        for proc_num, (proc_key, proc_val) in enumerate(processes.items()):
            cls.process_idx_map[proc_key] = proc_num

    @classmethod
    def _update_vector_idxs(cls):
        cls._subnet_address_idx = 0
        cls._host_address_idx = cls.address_space_bounds[0]
        cls._compromised_idx = (
            cls._host_address_idx + cls.address_space_bounds[1]
        )
        cls._reachable_idx = cls._compromised_idx + 1
        cls._discovered_idx = cls._reachable_idx + 1
        cls._value_idx = cls._discovered_idx + 1
        cls._discovery_value_idx = cls._value_idx + 1
        cls._access_idx = cls._discovery_value_idx + 1
        cls._os_start_idx = cls._access_idx + 1
        cls._service_start_idx = cls._os_start_idx + cls.num_os
        cls._process_start_idx = cls._service_start_idx + cls.num_services
        cls._detection_level_idx = cls._process_start_idx + cls.num_processes
        cls._detection_threshold_idx = cls._detection_level_idx + 1
        cls._detection_multiplier_idx = cls._detection_threshold_idx + 1
        cls.state_size = cls._detection_multiplier_idx + 1

    @classmethod
    def _subnet_address_idx_slice(cls):
        return slice(cls._subnet_address_idx, cls._host_address_idx)

    @classmethod
    def _host_address_idx_slice(cls):
        return slice(cls._host_address_idx, cls._compromised_idx)

    @classmethod
    def _get_service_idx(cls, srv_num):
        return cls._service_start_idx+srv_num

    @classmethod
    def _service_idx_slice(cls):
        return slice(cls._service_start_idx, cls._process_start_idx)

    @classmethod
    def _get_os_idx(cls, os_num):
        return cls._os_start_idx+os_num

    @classmethod
    def _os_idx_slice(cls):
        return slice(cls._os_start_idx, cls._service_start_idx)

    @classmethod
    def _get_process_idx(cls, proc_num):
        return cls._process_start_idx+proc_num

    @classmethod
    def _process_idx_slice(cls):
        return slice(cls._process_start_idx, cls.state_size)

    @classmethod
    def get_readable(cls, vector):
        readable_dict = dict()
        hvec = cls(vector)
        readable_dict["Address"] = hvec.address
        readable_dict["Compromised"] = bool(hvec.compromised)
        readable_dict["Reachable"] = bool(hvec.reachable)
        readable_dict["Discovered"] = bool(hvec.discovered)
        readable_dict["Value"] = hvec.value
        readable_dict["Discovery Value"] = hvec.discovery_value
        readable_dict["Access"] = hvec.access
        for os_name in cls.os_idx_map:
            readable_dict[f"{os_name}"] = hvec.is_running_os(os_name)
        for srv_name in cls.service_idx_map:
            readable_dict[f"{srv_name}"] = hvec.is_running_service(srv_name)
        for proc_name in cls.process_idx_map:
            readable_dict[f"{proc_name}"] = hvec.is_running_process(proc_name)

        readable_dict["Detection Level"] = hvec.detection_level
        readable_dict["Detection Threshold"] = hvec.detection_threshold
        readable_dict["Detection Multiplier"] = hvec.detection_multiplier

        return readable_dict

    @classmethod
    def reset(cls):
        """Resets any class variables.

        This is used to avoid errors when changing scenarios within a single
        python session
        """
        cls.address_space_bounds = None

    def __repr__(self):
        return f"Host: {self.address}"

    def __hash__(self):
        return hash(str(self.vector))

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, HostVector):
            return False
        return np.array_equal(self.vector, other.vector)
