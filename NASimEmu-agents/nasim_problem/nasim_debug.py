from vec_env.subproc_vec_env import SubprocVecEnv
from tqdm import tqdm
import gym, numpy as np, torch
import itertools
import networkx as nx
import plotly.graph_objects as go

from config import config

from nasimemu.nasim.envs.action import Exploit, PrivilegeEscalation, ServiceScan, OSScan, SubnetScan, ProcessScan
from nasimemu.nasim.envs.host_vector import HostVector
from nasimemu.env_utils import convert_to_graph, plot_network, _make_graph, _plot

import plotly.io as pio  

ACTION_MAP = {
    ServiceScan: 'ServScan',
    OSScan: 'OSScan',
    SubnetScan: 'SubnetScan',
    ProcessScan: 'ProcScan',
    Exploit: 'Exploit',
    PrivilegeEscalation: 'PrivEsc'
}

class NASimDebug():
    def calc_baseline(self):
        # calculate the mean success try for probability p
        def get_expected_success(p, l=100):
            assert p >= 0.1, 'Computation not precise for p < 0.1; increase l'

            n_0 = np.arange(l)
            n_1 = n_0 + 1

            p_not = np.full(l, 1-p) ** n_0
            p_mul = p_not * p

            p_mean = np.sum( p_mul * n_1 )
            return p_mean

        test_env = gym.make('NASimEmuEnv-v99', random_init=False)
        env_raw = test_env.env
        test_env.reset()

        # useful constants
        p_e = np.mean([e_data['prob'] for e_name, e_data in env_raw.exploit_list])
        p_pe = np.mean([pe_data['prob'] for pe_name, pe_data in env_raw.privesc_list])

        exp_e = get_expected_success(p_e) # we assume exploit success probability = 0.8
        exp_pe = get_expected_success(p_pe) # 

        # stats
        req_actions_list = []
        reward_list = []

        reward_per_action = -0.1
        reward_per_sensitive_host = 10.

        sensitive_hosts = 0

        # need to iterate and average, because there is randomness in the generated scenario
        for i in range(config.eval_problems):
            s = test_env.reset()
            network = env_raw.env.network

            req_actions = ( len(network.hosts) +                          # service scan each host
                            exp_e * len(network.hosts) +                  # exploit each host   
                            exp_pe * len(network.get_sensitive_hosts()) + # privesc each sensitive host
                            len(network.subnets) )                        # scan in each of the subnets once to discover whole network

            if len(env_raw.env.scenario.privescs) > 1:            # only if there are multiple privescs
                req_actions += len(network.get_sensitive_hosts()) # process scan for each sensitive host

            reward = reward_per_action * req_actions + \
                     reward_per_sensitive_host * len(network.get_sensitive_hosts())

            sensitive_hosts += len(network.get_sensitive_hosts())

            req_actions_list.append(req_actions)
            reward_list.append(reward)

        req_actions_mean = np.mean(req_actions_list)
        reward_mean = np.mean(reward_list)

        print(f"This scenario contains {sensitive_hosts / config.eval_problems} sensitive hosts on average.")
        print(f"{test_env.env.scenario_name};{reward_mean:.2f};{req_actions_mean:.2f}")

    def evaluate(self, net):
        eval_split = getattr(config, 'eval_split', 'both')
        if eval_split == 'test' and not config.test_scenario_name:
            raise SystemExit("--eval_split test requires --test_scenario.")

        results = {}
        if eval_split in ('both', 'train'):
            results['eval_trn'] = self._eval(net, config.scenario_name)
        if eval_split in ('both', 'test'):
            results['eval_tst'] = (
                self._eval(net, config.test_scenario_name)
                if config.test_scenario_name else None
            )

        return results

    def _eval(self, net, scenario_name):
        # choose auto overrides depending on whether this is test or train
        is_test = (config.test_scenario_name is not None) and (scenario_name == config.test_scenario_name)
        auto_mode_use = getattr(config, 'auto_mode_test', None) if is_test and getattr(config, 'auto_mode_test', None) is not None else getattr(config, 'auto_mode', 'off')
        auto_template_use = getattr(config, 'auto_template_test', None) if is_test and getattr(config, 'auto_template_test', None) is not None else getattr(config, 'auto_template', None)

        def make_one(i):
            # Force training_mode=False so the curriculum manager reports its
            # FINAL (hardest) stage during evaluation, regardless of the training
            # epoch. Without this, eval envs inherit the training-time
            # training_mode=True and, since set_epoch is never called on them,
            # sit at epoch 0 = the baseline stage (IDS off) -- so the reported
            # eval metrics and the save_best checkpoint would track IDS-OFF
            # performance while the real test (standalone --eval) uses full IDS.
            # This aligns training-time eval with the real test difficulty.
            return gym.make('NASimEmuEnv-v99', random_init=False, scenario_name=scenario_name,
                training_mode=False,
                seed=(config.seed + i) if config.seed is not None else None,
                auto_mode=auto_mode_use, auto_template=auto_template_use,
                auto_host_range=getattr(config, 'auto_host_range', None),
                auto_subnet_count=getattr(config, 'auto_subnet_count', None),
                auto_topology=getattr(config, 'auto_topology', None),
                auto_sensitive_policy=getattr(config, 'auto_sensitive_policy', None),
                auto_seed_base=getattr(config, 'auto_seed_base', None),
                auto_sensitive_jitter=getattr(config, 'auto_sensitive_jitter', 0.0)
            )

        test_env = SubprocVecEnv(
            [lambda i=i: make_one(i) for i in range(config.eval_batch)],
            in_series=(config.eval_batch // config.cpus), context='fork',
        )
        tqdm_val = tqdm(desc='Validating', unit=' problems')

        saved_state = net.__class__() # create a fresh instance
        saved_state.clone_state(net)
        
        with torch.no_grad():
            net.eval()
            net.reset_state()

            r_tot = 0.
            r_episodes = 0.
            problems_solved = 0
            problems_finished = 0
            episode_lengths = 0
            captured = 0
            steps = 0
            terminated = np.zeros(config.eval_batch, dtype=bool)

            s = test_env.reset()

            while True:
                steps += np.sum(~terminated)

                a, v, pi, _ = net(s)
                a = np.array(a, dtype=object)

                s, r_, d_, i_ = test_env.step(a)
                net.reset_state(d_)

                r = r_[~terminated]
                d = d_[~terminated]
                i = list(itertools.compress(i_, ~terminated))

                # print(r)
                r_tot += np.sum(r)
                r_episodes += sum(x['r_tot'] for x in i if x['done'] == True)

                problems_solved   += sum('d_true' in x and x['d_true'] == True for x in i)
                problems_finished += np.sum(d)
                
                episode_lengths += sum(x['step_idx'] for x in i if x['done'] == True)
                captured += sum(x['captured'] for x in i if x['done'] == True)

                tqdm_val.update(np.sum(d))

                if problems_finished >= config.eval_problems:
                    terminated |= d_
                    if np.all(terminated):
                        break

            r_avg = r_tot / steps # average reward per step <- obsolete! this is a wrong metric to track in case there are differences in episode lengths
            r_avg_episodes = r_episodes / problems_finished

            problems_solved_ps  = problems_solved / steps
            problems_solved_avg = problems_solved / problems_finished

            episode_lengths_avg = episode_lengths / problems_finished
            captured_avg = captured / problems_finished

            net.train()

        net.clone_state(saved_state)

        tqdm_val.close()
        test_env.close()

        log = {
            'reward_avg': r_avg,
            'reward_avg_episodes': r_avg_episodes,
            'eplen_avg': episode_lengths_avg,
            'captured_avg': captured_avg
            # 'solved_per_step': problems_solved_ps,
            # 'solved_avg': problems_solved_avg,
        }

        return log

    def debug(self, net, show=False):
        """
        Debug method with comprehensive error handling to prevent workflow failures.
        
        Args:
            net: Neural network model
            show: Whether to display the visualization
            
        Returns:
            dict: Log dictionary with 'value', 'q_val', and 'figure' keys
        """
        try:
            test_env = gym.make('NASimEmuEnv-v99', random_init=False)
            s = test_env.reset()
            
            saved_state = net.__class__() # create a fresh instance
            saved_state.clone_state(net)
            
            # Initialize default values in case of neural network failure
            node_softmax = None
            action_softmax = None
            value = torch.tensor(0.0)
            q_val = torch.tensor(0.0)
            
            try:
                with torch.no_grad():
                    net.eval()
                    net.reset_state()
                    node_softmax, action_softmax, value, q_val = net([s], complete=True)
                    net.train()
            except Exception as e:
                print(f"Warning: Neural network forward pass failed: {e}")
                print("Continuing with default values for visualization...")
                
            net.clone_state(saved_state)

            # Create graph with error handling
            try:
                G = self._make_graph(s, node_softmax, action_softmax)
            except Exception as e:
                print(f"Warning: Graph creation failed: {e}")
                print("Creating minimal fallback graph...")
                G = self._create_fallback_graph()

            # Create plot with error handling
            try:
                value_scalar = value.item() if hasattr(value, 'item') else float(value)
                q_val_scalar = q_val.item() if hasattr(q_val, 'item') else float(q_val)
                fig = self._plot(G, value_scalar, q_val_scalar, test_env)
            except Exception as e:
                print(f"Warning: Plot creation failed: {e}")
                print("Creating fallback visualization...")
                fig = self._create_fallback_plot(e)

            # Safe figure display
            if show:
                try:
                    fig.show()
                except Exception as e:
                    print(f"Warning: Could not display figure: {e}")

            log = {
                'value': value, 
                'q_val': q_val,
                'figure': fig
            }

            return log
            
        except Exception as e:
            print(f"Error: Debug method failed completely: {e}")
            # Return minimal log to prevent complete failure
            return {
                'value': torch.tensor(0.0), 
                'q_val': torch.tensor(0.0),
                'figure': self._create_error_plot(str(e))
            }

    def trace(self, net, net_name):
        with torch.no_grad():
            test_env = gym.make('NASimEmuEnv-v99', verbose=False, random_init=False)
            s = test_env.reset()
            i = {'subnet_graph': test_env.subnet_graph.copy()}

            # self._plot_network(test_env, net, s)

            net.reset_state()

            print("Note: This is simulator state and is not exposed to the policy.")
            test_env.env.env.render_state()

            test_env.env.env.render(obs=test_env.s_raw)
            # input()
        
            pio.kaleido.scope.mathjax = None

            for step in range(1, 200):
                print(f"\nSTEP {step}")
                a, v, pi, _ = net([s])
                # a = np.array(a, dtype=object) # todo: test when error occurs

                s_raw_orig = test_env.s_raw
                orig_subnet_graph = i['subnet_graph'].copy()

                s, r, d, i = test_env.step(a[0])
                net.reset_state([d])

                print()
                print(f"a: {i['a_raw']}, r: {r}, d: {d}")
                # print(f"V(s)={v.item():.2f}, Q(s, a_cnt)={q.item():.2f}")
                if v is not None:
                    print(f"V(s)={v.item():.2f}")

                fig = plot_network(s_raw_orig, orig_subnet_graph, i['a_raw'])
                # fig.show()

                fig.update_xaxes(range=[-1.35, 1.3])
                fig.update_yaxes(range=[-1.2, 1.3])
                # fig.write_image(f"out/trace-{step}.pdf", width=1200, height=600, scale=1.0)  # Disabled PDF generation

                if i['success']:
                    test_env.env.env.render(obs=i['s_raw'])

                if d:
                    print("-------------FINISHED----------------")
                    exit()

                    # input()

    def _make_graph(self, s, node_softmax, action_softmax):
        """
        Create a NetworkX graph with neural network probability data for visualization.
        Includes comprehensive input validation and error handling.
        
        Args:
            s: Environment state (raw observation)
            node_softmax: Node attention probabilities from neural network
            action_softmax: Action probabilities from neural network
            
        Returns:
            NetworkX graph with node attributes for visualization
        """
        try:
            # Validate input state
            if s is None:
                raise ValueError("Environment state 's' cannot be None")
            
            # Convert state to graph format using existing utility with error handling
            try:
                # Try to get subnet_graph from the environment if available
                subnet_graph = getattr(self, '_last_subnet_graph', [])
            except:
                subnet_graph = []
                
            try:
                node_feats, edge_index, node_index, pos_index = convert_to_graph(s, subnet_graph, version=2)
            except Exception as e:
                print(f"Warning: convert_to_graph failed: {e}")
                # Create minimal fallback data
                node_feats = np.array([[1, 0, 0, 0, 0]])  # Single node
                edge_index = np.array([[0], [0]])  # Self-loop
                node_index = np.array([[0, 0]])  # Single host node
                pos_index = np.array([0])
            
            # Validate graph data
            if edge_index is None or len(edge_index) == 0:
                print("Warning: Empty edge_index, creating single node graph")
                edge_index = np.array([[0], [0]])  # Self-loop for single node
                
            # Create NetworkX graph from edge index with error handling
            try:
                G = nx.Graph()
                if edge_index.ndim == 2 and edge_index.shape[0] >= 2:
                    G.add_edges_from(edge_index.T)
                else:
                    # Fallback: create single node
                    G.add_node(0)
            except Exception as e:
                print(f"Warning: Graph creation failed: {e}")
                G = nx.Graph()
                G.add_node(0)  # Minimal single-node graph
            
            # Ensure graph has at least one node
            if len(G.nodes) == 0:
                G.add_node(0)
            
            # Generate layout positions with error handling
            try:
                pos = nx.kamada_kawai_layout(G)
            except Exception as e:
                print(f"Warning: Layout generation failed: {e}")
                # Fallback to simple positions
                pos = {i: (i * 0.1, 0) for i in G.nodes}
            
            # Validate and process neural network probability data
            node_probs = self._validate_probability_data(node_softmax, len(G.nodes), "node_softmax")
            action_probs = self._validate_probability_data(action_softmax, len(G.nodes), "action_softmax")

            # Helper functions for node attributes with error handling
            def get_node_label(i):
                try:
                    if node_index is None or i >= len(node_index):
                        return f"Node {i}"
                    node_idx = node_index[i]
                    if len(node_idx) < 2:
                        return f"Node {i}"
                    if node_idx[1] == -1:  # Subnet node
                        return f"Subnet {node_idx[0]}"
                    else:  # Host node
                        return f"Host {node_idx}"
                except Exception:
                    return f"Node {i}"
                    
            def get_node_type(i):
                try:
                    if node_index is None or i >= len(node_index):
                        return 'node'
                    return 'subnet' if node_index[i][1] == -1 else 'node'
                except Exception:
                    return 'node'
                    
            def get_node_color(i):
                try:
                    if node_index is None or i >= len(node_index):
                        return 'lightblue'
                    
                    node_idx = node_index[i]
                    if len(node_idx) < 2 or node_idx[1] == -1:  # Subnet node
                        return 'grey'
                    else:  # Host node
                        # Check if host is sensitive (has value > 0)
                        if node_feats is not None and i < len(node_feats):
                            try:
                                host_vec = HostVector(node_feats[i][1:])  # Skip node type indicator
                                return 'red' if host_vec.value > 0 else 'lightblue'
                            except Exception:
                                return 'lightblue'
                        return 'lightblue'
                except Exception:
                    return 'lightblue'
                    
            def get_node_symbol(i):
                try:
                    return 'triangle-up' if get_node_type(i) == 'subnet' else 'circle'
                except Exception:
                    return 'circle'

            # Set node attributes with error handling
            try:
                node_labels = {i: get_node_label(i) for i in G.nodes}
                node_types = {i: get_node_type(i) for i in G.nodes}
                node_colors = {i: get_node_color(i) for i in G.nodes}
                node_symbols = {i: get_node_symbol(i) for i in G.nodes}
                
                # Add probability data as node attributes
                node_n_probs = {i: float(node_probs[i]) if i < len(node_probs) else 0.0 for i in G.nodes}
                node_a_probs = {i: float(action_probs[i]) if i < len(action_probs) else 0.0 for i in G.nodes}
                
                # Set all attributes
                nx.set_node_attributes(G, pos, 'pos')
                nx.set_node_attributes(G, node_labels, 'label')
                nx.set_node_attributes(G, node_types, 'type')
                nx.set_node_attributes(G, node_colors, 'color')
                nx.set_node_attributes(G, node_symbols, 'symbol')
                nx.set_node_attributes(G, node_n_probs, 'n_prob')
                nx.set_node_attributes(G, node_a_probs, 'a_prob')
                
                # Set default line width
                node_line_widths = {i: 1.0 for i in G.nodes}
                nx.set_node_attributes(G, node_line_widths, 'line_width')
                
            except Exception as e:
                print(f"Warning: Node attribute setting failed: {e}")
                # Set minimal attributes
                for node in G.nodes:
                    G.nodes[node]['pos'] = pos.get(node, (0, 0))
                    G.nodes[node]['label'] = f"Node {node}"
                    G.nodes[node]['type'] = 'node'
                    G.nodes[node]['color'] = 'lightblue'
                    G.nodes[node]['symbol'] = 'circle'
                    G.nodes[node]['n_prob'] = 0.0
                    G.nodes[node]['a_prob'] = 0.0
                    G.nodes[node]['line_width'] = 1.0
            
            return G
            
        except Exception as e:
            print(f"Error: _make_graph failed completely: {e}")
            # Return minimal fallback graph
            return self._create_fallback_graph()
    
    def _validate_probability_data(self, prob_data, expected_length, data_name):
        """
        Validate and process neural network probability data with comprehensive error handling.
        
        Args:
            prob_data: Raw probability data from neural network
            expected_length: Expected length to match number of nodes
            data_name: Name for error reporting
            
        Returns:
            numpy.ndarray: Validated probability array
        """
        try:
            if prob_data is None:
                return np.zeros(expected_length)
            
            # Convert tensor to numpy if needed
            if hasattr(prob_data, 'detach'):
                try:
                    prob_data = prob_data.detach().cpu().numpy()
                except Exception as e:
                    print(f"Warning: Failed to convert {data_name} tensor to numpy: {e}")
                    return np.zeros(expected_length)
            
            # Flatten if needed
            if hasattr(prob_data, 'flatten'):
                prob_data = prob_data.flatten()
            elif hasattr(prob_data, 'ravel'):
                prob_data = prob_data.ravel()
            
            # Convert to numpy array if not already
            if not isinstance(prob_data, np.ndarray):
                try:
                    prob_data = np.array(prob_data)
                except Exception as e:
                    print(f"Warning: Failed to convert {data_name} to numpy array: {e}")
                    return np.zeros(expected_length)
            
            # Validate data type and values
            if not np.issubdtype(prob_data.dtype, np.number):
                print(f"Warning: {data_name} contains non-numeric data")
                return np.zeros(expected_length)
            
            # Check for invalid values (NaN, inf)
            if np.any(np.isnan(prob_data)) or np.any(np.isinf(prob_data)):
                print(f"Warning: {data_name} contains NaN or infinite values")
                prob_data = np.nan_to_num(prob_data, nan=0.0, posinf=1.0, neginf=0.0)
            
            # Ensure values are in valid probability range [0, 1]
            prob_data = np.clip(prob_data, 0.0, 1.0)
            
            # Handle length mismatch
            if len(prob_data) != expected_length:
                print(f"Warning: {data_name} length ({len(prob_data)}) doesn't match expected ({expected_length})")
                result = np.zeros(expected_length)
                min_len = min(len(prob_data), expected_length)
                result[:min_len] = prob_data[:min_len]
                return result
            
            return prob_data
            
        except Exception as e:
            print(f"Error: Failed to validate {data_name}: {e}")
            return np.zeros(expected_length)

    def _plot(self, G, value, q_val, test_env):
        """
        Create Plotly visualization with neural network values and probability information.
        Includes comprehensive error handling to prevent visualization failures.
        
        Args:
            G: NetworkX graph with node attributes including probabilities
            value: State value from neural network
            q_val: Q-value from neural network
            test_env: Test environment for action mapping
            
        Returns:
            plotly.graph_objects.Figure: Interactive visualization with NN debugging info
        """
        try:
            # Validate inputs
            if G is None:
                raise ValueError("Graph G cannot be None")
            
            if len(G.nodes) == 0:
                raise ValueError("Graph has no nodes")
            
            # Validate and sanitize neural network values
            try:
                value = float(value) if value is not None else 0.0
                q_val = float(q_val) if q_val is not None else 0.0
                
                # Handle invalid values
                if np.isnan(value) or np.isinf(value):
                    print("Warning: Invalid value detected, using 0.0")
                    value = 0.0
                if np.isnan(q_val) or np.isinf(q_val):
                    print("Warning: Invalid q_val detected, using 0.0")
                    q_val = 0.0
                    
            except (TypeError, ValueError) as e:
                print(f"Warning: Failed to convert neural network values: {e}")
                value, q_val = 0.0, 0.0

            def get_edges(edge_type):
                """Get edge coordinates for different edge types with error handling"""
                edge_x = []
                edge_y = []

                try:
                    for edge in G.edges():
                        try:
                            node_in = G.nodes[edge[0]]
                            node_out = G.nodes[edge[1]]

                            # Validate node attributes exist
                            if 'type' not in node_in or 'type' not in node_out:
                                continue
                            if 'pos' not in node_in or 'pos' not in node_out:
                                continue

                            if edge_type == 'subnet':  # both have to be subnets
                                if not (node_in['type'] == 'subnet' and node_out['type'] == 'subnet'):
                                    continue
                            elif edge_type == 'node':  # both can't be subnets
                                if node_in['type'] == 'subnet' and node_out['type'] == 'subnet':
                                    continue

                            x0, y0 = node_in['pos']
                            x1, y1 = node_out['pos']
                            
                            # Validate coordinates
                            if any(not isinstance(coord, (int, float)) or np.isnan(coord) or np.isinf(coord) 
                                   for coord in [x0, y0, x1, y1]):
                                continue
                                
                            edge_x.extend([x0, x1, None])
                            edge_y.extend([y0, y1, None])
                            
                        except Exception as e:
                            print(f"Warning: Failed to process edge {edge}: {e}")
                            continue
                            
                except Exception as e:
                    print(f"Warning: Failed to process edges: {e}")

                return edge_x, edge_y

            # Create edge traces with error handling
            try:
                node_edges = get_edges('node')
                subnet_edges = get_edges('subnet')

                edge_trace = go.Scatter(
                    x=node_edges[0], y=node_edges[1],
                    line=dict(width=1, color='black', dash='dash'),
                    hoverinfo='none',
                    mode='lines',
                    name='Node Connections')

                subnet_trace = go.Scatter(
                    x=subnet_edges[0], y=subnet_edges[1],
                    line=dict(width=1, color='black'),
                    hoverinfo='none',
                    mode='lines',
                    name='Subnet Connections')
                    
            except Exception as e:
                print(f"Warning: Failed to create edge traces: {e}")
                # Create empty edge traces
                edge_trace = go.Scatter(x=[], y=[], mode='lines', name='Node Connections')
                subnet_trace = go.Scatter(x=[], y=[], mode='lines', name='Subnet Connections')

            # Prepare node data with comprehensive error handling
            node_x = []
            node_y = []
            node_text = []
            node_color = []
            node_symbols = []
            node_line_widths = []
            node_hover_text = []

            try:
                for node_id, node in G.nodes.items():
                    try:
                        # Get position with fallback
                        pos = node.get('pos', (0, 0))
                        if not isinstance(pos, (tuple, list)) or len(pos) < 2:
                            pos = (0, 0)
                        x, y = pos
                        
                        # Validate coordinates
                        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                            x, y = 0, 0
                        if np.isnan(x) or np.isinf(x):
                            x = 0
                        if np.isnan(y) or np.isinf(y):
                            y = 0
                            
                        node_x.append(x)
                        node_y.append(y)

                        # Get node attributes with fallbacks
                        base_label = node.get('label', f'Node {node_id}')
                        node_type = node.get('type', 'node')
                        n_prob = node.get('n_prob', 0.0)
                        a_prob = node.get('a_prob', 0.0)
                        
                        # Validate probabilities
                        try:
                            n_prob = float(n_prob)
                            a_prob = float(a_prob)
                            if np.isnan(n_prob) or np.isinf(n_prob):
                                n_prob = 0.0
                            if np.isnan(a_prob) or np.isinf(a_prob):
                                a_prob = 0.0
                        except (TypeError, ValueError):
                            n_prob = a_prob = 0.0
                        
                        # Enhanced node text with formatted probability information
                        if node_type == 'node':  # Only show probabilities for host nodes
                            # Format probabilities with better precision and highlighting
                            n_prob_str = f"{n_prob:.3f}" if n_prob < 0.1 else f"**{n_prob:.3f}**"
                            a_prob_str = f"{a_prob:.3f}" if a_prob < 0.1 else f"**{a_prob:.3f}**"
                            node_text.append(f"{base_label}<br>N:{n_prob_str} A:{a_prob_str}")
                        else:
                            node_text.append(str(base_label))

                        # Enhanced detailed hover text with action probability breakdown
                        hover_info = [f"🎯 Node: {base_label}"]
                        hover_info.append(f"📊 Type: {node_type}")
                        hover_info.append(f"🧠 Node Attention: {n_prob:.4f} ({n_prob*100:.2f}%)")
                        hover_info.append(f"⚡ Action Probability: {a_prob:.4f} ({a_prob*100:.2f}%)")
                        
                        # Add probability ranking indicators
                        if n_prob > 0.5:
                            hover_info.append("🔥 VERY HIGH Node Attention")
                        elif n_prob > 0.2:
                            hover_info.append("🔶 HIGH Node Attention")
                        elif n_prob > 0.1:
                            hover_info.append("🔸 Medium Node Attention")
                        
                        if a_prob > 0.5:
                            hover_info.append("⭐ VERY HIGH Action Probability")
                        elif a_prob > 0.2:
                            hover_info.append("🟡 HIGH Action Probability")
                        elif a_prob > 0.1:
                            hover_info.append("🟠 Medium Action Probability")
                        
                        # Add combined attention score
                        combined_score = (n_prob + a_prob) / 2
                        hover_info.append(f"📈 Combined Score: {combined_score:.4f}")
                        
                        # Most likely action indicator
                        if a_prob > 0.1:
                            hover_info.append("🎯 Likely Target for Action")
                            
                        node_hover_text.append("<br>".join(hover_info))

                        # Enhanced color coding based on probabilities
                        base_color = node.get('color', 'lightblue')
                        
                        # Apply probability-based color enhancement
                        if node_type == 'node':  # Only enhance host nodes
                            max_prob = max(n_prob, a_prob)
                            if max_prob > 0.5:
                                # Very high probability - bright red
                                enhanced_color = 'crimson'
                            elif max_prob > 0.2:
                                # High probability - orange-red
                                enhanced_color = 'orangered'
                            elif max_prob > 0.1:
                                # Medium probability - orange
                                enhanced_color = 'orange'
                            elif max_prob > 0.05:
                                # Low probability - yellow
                                enhanced_color = 'gold'
                            else:
                                # Very low probability - keep base color but make it lighter
                                if base_color == 'red':
                                    enhanced_color = 'lightcoral'
                                elif base_color == 'lightblue':
                                    enhanced_color = 'lightsteelblue'
                                else:
                                    enhanced_color = base_color
                        else:
                            enhanced_color = base_color
                            
                        node_color.append(enhanced_color)
                        
                        # Enhanced symbol selection based on attention
                        base_symbol = node.get('symbol', 'circle')
                        if node_type == 'node' and max(n_prob, a_prob) > 0.2:
                            # High attention nodes get star symbol
                            enhanced_symbol = 'star'
                        elif node_type == 'node' and max(n_prob, a_prob) > 0.1:
                            # Medium attention nodes get diamond symbol
                            enhanced_symbol = 'diamond'
                        else:
                            enhanced_symbol = base_symbol
                            
                        node_symbols.append(enhanced_symbol)
                        
                        # Enhanced line width calculation with better scaling
                        base_width = node.get('line_width', 1.0)
                        try:
                            base_width = float(base_width)
                            if np.isnan(base_width) or np.isinf(base_width):
                                base_width = 1.0
                        except (TypeError, ValueError):
                            base_width = 1.0
                        
                        # More sophisticated probability enhancement
                        max_prob = max(n_prob, a_prob)
                        if max_prob > 0.5:
                            prob_enhancement = 8.0  # Very thick border for high attention
                        elif max_prob > 0.2:
                            prob_enhancement = 5.0  # Thick border for medium-high attention
                        elif max_prob > 0.1:
                            prob_enhancement = 3.0  # Medium border for medium attention
                        elif max_prob > 0.05:
                            prob_enhancement = 1.5  # Slight enhancement for low attention
                        else:
                            prob_enhancement = 0.0  # No enhancement for very low attention
                            
                        enhanced_width = base_width + prob_enhancement
                        node_line_widths.append(enhanced_width)
                        
                    except Exception as e:
                        print(f"Warning: Failed to process node {node_id}: {e}")
                        # Add fallback node data
                        node_x.append(0)
                        node_y.append(0)
                        node_text.append(f"Node {node_id}")
                        node_color.append('lightblue')
                        node_symbols.append('circle')
                        node_line_widths.append(1.0)
                        node_hover_text.append(f"Node: {node_id}<br>Error in processing")
                        
            except Exception as e:
                print(f"Warning: Failed to process nodes: {e}")
                # Create minimal node data
                node_x = [0]
                node_y = [0]
                node_text = ["Error Node"]
                node_color = ['red']
                node_symbols = ['circle']
                node_line_widths = [1.0]
                node_hover_text = ["Error processing nodes"]

            # Calculate enhanced node sizes based on probabilities
            node_sizes = []
            try:
                for node_id, node in G.nodes.items():
                    try:
                        n_prob = node.get('n_prob', 0.0)
                        a_prob = node.get('a_prob', 0.0)
                        node_type = node.get('type', 'node')
                        
                        # Base size depends on node type
                        base_size = 50 if node_type == 'subnet' else 40
                        
                        # Calculate size enhancement based on probabilities
                        max_prob = max(n_prob, a_prob)
                        if max_prob > 0.5:
                            size_enhancement = 30  # Very large for high attention
                        elif max_prob > 0.2:
                            size_enhancement = 20  # Large for medium-high attention
                        elif max_prob > 0.1:
                            size_enhancement = 10  # Medium for medium attention
                        elif max_prob > 0.05:
                            size_enhancement = 5   # Slight increase for low attention
                        else:
                            size_enhancement = 0   # No enhancement for very low attention
                            
                        enhanced_size = base_size + size_enhancement
                        node_sizes.append(enhanced_size)
                        
                    except Exception as e:
                        print(f"Warning: Failed to calculate size for node {node_id}: {e}")
                        node_sizes.append(40)  # Default size
                        
            except Exception as e:
                print(f"Warning: Failed to calculate node sizes: {e}")
                node_sizes = [40] * len(node_x)  # Default sizes

            # Create node trace with error handling
            try:
                node_trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers+text',
                    hoverinfo='text',
                    hovertext=node_hover_text,
                    marker=dict(
                        showscale=False, 
                        color=node_color, 
                        symbol=node_symbols, 
                        size=node_sizes,  # Enhanced sizes based on probabilities
                        line_width=node_line_widths, 
                        line_color="black",
                        opacity=0.8  # Slight transparency for better visual appeal
                    ),
                    text=node_text,
                    textposition="top center",
                    textfont=dict(
                        size=10,
                        color="black"
                    ),
                    name='Network Nodes')
                    
            except Exception as e:
                print(f"Warning: Failed to create node trace: {e}")
                # Create minimal node trace
                node_trace = go.Scatter(
                    x=[0], y=[0],
                    mode='markers+text',
                    text=["Error"],
                    marker=dict(color='red', size=40),
                    name='Error Node')

            # Calculate summary statistics for enhanced annotations
            try:
                all_node_probs = [node.get('n_prob', 0.0) for node in G.nodes.values()]
                all_action_probs = [node.get('a_prob', 0.0) for node in G.nodes.values()]
                
                max_node_prob = max(all_node_probs) if all_node_probs else 0.0
                max_action_prob = max(all_action_probs) if all_action_probs else 0.0
                avg_node_prob = np.mean(all_node_probs) if all_node_probs else 0.0
                avg_action_prob = np.mean(all_action_probs) if all_action_probs else 0.0
                
                # Count high attention nodes
                high_attention_nodes = sum(1 for p in all_node_probs if p > 0.2)
                high_action_nodes = sum(1 for p in all_action_probs if p > 0.2)
                
            except Exception as e:
                print(f"Warning: Failed to calculate summary statistics: {e}")
                max_node_prob = max_action_prob = avg_node_prob = avg_action_prob = 0.0
                high_attention_nodes = high_action_nodes = 0

            # Create the figure with error handling
            try:
                # Enhanced title with value interpretation
                value_interpretation = ""
                if value > 0.5:
                    value_interpretation = " (Promising State)"
                elif value > 0.0:
                    value_interpretation = " (Positive State)"
                elif value < -0.5:
                    value_interpretation = " (Poor State)"
                else:
                    value_interpretation = " (Neutral State)"
                
                q_val_interpretation = ""
                if q_val > 0.5:
                    q_val_interpretation = " (Good Action)"
                elif q_val > 0.0:
                    q_val_interpretation = " (Positive Action)"
                elif q_val < -0.5:
                    q_val_interpretation = " (Poor Action)"
                else:
                    q_val_interpretation = " (Neutral Action)"

                fig = go.Figure(
                    data=[edge_trace, subnet_trace, node_trace],
                    layout=go.Layout(
                        title=dict(
                            text=f"🧠 Neural Network Debug View<br>📊 State Value: {value:.4f}{value_interpretation} | ⚡ Q-Value: {q_val:.4f}{q_val_interpretation}",
                            x=0.5,
                            font=dict(size=16, color="darkblue")
                        ),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=20, r=20, t=100),
                        annotations=[
                            # Main neural network values annotation
                            dict(
                                text=f"🎯 <b>Neural Network Values</b><br>" +
                                     f"📈 State Value: <b>{value:.4f}</b>{value_interpretation}<br>" +
                                     f"⚡ Q-Value: <b>{q_val:.4f}</b>{q_val_interpretation}<br>" +
                                     f"🔥 Max Node Attention: <b>{max_node_prob:.3f}</b><br>" +
                                     f"🎯 Max Action Prob: <b>{max_action_prob:.3f}</b>",
                                showarrow=False,
                                xref="paper", yref="paper",
                                x=0.02, y=0.98,
                                xanchor='left', yanchor='top',
                                bgcolor="rgba(240,248,255,0.95)",
                                bordercolor="darkblue",
                                borderwidth=2,
                                font=dict(size=11, color="darkblue")
                            ),
                            # Summary statistics annotation
                            dict(
                                text=f"📊 <b>Attention Summary</b><br>" +
                                     f"🧠 Avg Node Attention: <b>{avg_node_prob:.3f}</b><br>" +
                                     f"⚡ Avg Action Prob: <b>{avg_action_prob:.3f}</b><br>" +
                                     f"🔥 High Attention Nodes: <b>{high_attention_nodes}</b><br>" +
                                     f"🎯 High Action Nodes: <b>{high_action_nodes}</b>",
                                showarrow=False,
                                xref="paper", yref="paper",
                                x=0.98, y=0.98,
                                xanchor='right', yanchor='top',
                                bgcolor="rgba(255,248,240,0.95)",
                                bordercolor="darkorange",
                                borderwidth=2,
                                font=dict(size=10, color="darkorange")
                            ),
                            # Legend annotation
                            dict(
                                text="🎨 <b>Visual Legend</b><br>" +
                                     "🔴 High Probability (>0.2)<br>" +
                                     "🟠 Medium Probability (0.1-0.2)<br>" +
                                     "🟡 Low Probability (0.05-0.1)<br>" +
                                     "⭐ Star = Very High Attention<br>" +
                                     "💎 Diamond = High Attention<br>" +
                                     "⚫ Circle = Normal/Low Attention",
                                showarrow=False,
                                xref="paper", yref="paper",
                                x=0.02, y=0.02,
                                xanchor='left', yanchor='bottom',
                                bgcolor="rgba(248,248,255,0.95)",
                                bordercolor="purple",
                                borderwidth=2,
                                font=dict(size=9, color="purple")
                            )
                        ],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    )
                )

                # Apply consistent styling with error handling
                try:
                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(
                            size=12,
                            color="black"
                        )
                    )
                except Exception as e:
                    print(f"Warning: Failed to apply styling: {e}")

                return fig
                
            except Exception as e:
                print(f"Warning: Failed to create figure: {e}")
                return self._create_fallback_plot(str(e))
                
        except Exception as e:
            print(f"Error: _plot failed completely: {e}")
            return self._create_fallback_plot(str(e))

    # def _plot_new(self, graph, node_data):
    #     edge_x = []
    #     edge_y = []
    #     for edge in graph.edges():
    #         x0, y0 = node_data['x'][edge[0]], node_data['y'][edge[0]]
    #         x1, y1 = node_data['x'][edge[1]], node_data['y'][edge[1]]
    #         edge_x.append(x0)
    #         edge_x.append(x1)
    #         edge_x.append(None)
    #         edge_y.append(y0)
    #         edge_y.append(y1)
    #         edge_y.append(None)

    #         print(edge)
    #         print(x0, y0, x1, y1)

    #     edge_trace = go.Scatter(
    #         x=edge_x, y=edge_y,
    #         line=dict(width=0.5, color='#444'),
    #         hoverinfo='none',
    #         mode='lines')

    #     node_trace = go.Scatter(
    #         node_data,
    #         mode='markers+text',
    #         # hoverinfo='text',
    #         # marker=dict(showscale=False, size=15,),
    #         textposition="top center")

    #     fig = go.Figure(data=[edge_trace, node_trace],
    #                     layout=go.Layout(
    #                         showlegend=False,
    #                         hovermode='closest',
    #                         margin=dict(b=0,l=0,r=0,t=0),
    #                         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    #                         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)),
    #                     )

    #     fig.update_layout(
    #         paper_bgcolor="rgba(0,0,0,0)",
    #         plot_bgcolor="rgba(0,0,0,0)"
    #     )

    #     return fig


    # def _make_graph_new(self, s):
    #     node_feats, edge_index, node_index = s

    #     graph = nx.Graph() 
    #     graph.add_edges_from(edge_index.T) 
    #     node_positions = nx.kamada_kawai_layout(graph)
    #     print(graph.nodes)
    #     print(node_positions)
    #     node_positions = np.stack([(node_positions[i]) for i in range(len(graph.nodes))])

    #     node_data = {}
    #     node_data['x'] = node_positions[:, 0]
    #     node_data['y'] = node_positions[:, 1]

    #     node_color = lambda node_id: f'rgb({scenario["sensitive_hosts"][node_id] * 191 + 64}, 64, 64)'

    #     node_data['text']  = [f"Subnet {node_index[i][0]}" if node_index[i][1] == -1 else f"{node_index[i]}" for i in graph.nodes]
    #     node_data['marker'] = dict(opacity=1.0, size=15, color=['seagreen' if node_index[i][1] == -1 else 'skyblue' for i in graph.nodes])

    #     return graph, node_data
    def _create_fallback_graph(self):
        """
        Create a minimal fallback graph when graph creation fails.
        
        Returns:
            NetworkX.Graph: Minimal single-node graph with basic attributes
        """
        try:
            G = nx.Graph()
            G.add_node(0)
            
            # Set minimal node attributes
            G.nodes[0]['pos'] = (0, 0)
            G.nodes[0]['label'] = "Fallback Node"
            G.nodes[0]['type'] = 'node'
            G.nodes[0]['color'] = 'orange'
            G.nodes[0]['symbol'] = 'circle'
            G.nodes[0]['n_prob'] = 0.0
            G.nodes[0]['a_prob'] = 0.0
            G.nodes[0]['line_width'] = 2.0
            
            return G
            
        except Exception as e:
            print(f"Error: Even fallback graph creation failed: {e}")
            # Return absolute minimal graph
            G = nx.Graph()
            G.add_node(0)
            return G
    
    def _create_fallback_plot(self, error_msg):
        """
        Create a fallback plot when normal plotting fails.
        
        Args:
            error_msg: Error message to display
            
        Returns:
            plotly.graph_objects.Figure: Simple error visualization
        """
        try:
            fig = go.Figure()
            
            # Add a simple error node
            fig.add_trace(go.Scatter(
                x=[0], y=[0],
                mode='markers+text',
                marker=dict(size=60, color='orange', symbol='circle'),
                text=["Fallback<br>Visualization"],
                textposition="middle center",
                name='Fallback Node'
            ))
            
            fig.update_layout(
                title=dict(
                    text=f"Fallback Debug View<br>Error: {str(error_msg)[:100]}...",
                    x=0.5,
                    font=dict(size=14)
                ),
                showlegend=False,
                xaxis=dict(range=[-1, 1], showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(range=[-1, 1], showgrid=False, zeroline=False, showticklabels=False),
                margin=dict(b=20, l=20, r=20, t=80),
                annotations=[
                    dict(
                        text="Visualization failed<br>Using fallback display",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.02, y=0.98,
                        xanchor='left', yanchor='top',
                        bgcolor="rgba(255,200,200,0.8)",
                        bordercolor="red",
                        borderwidth=1,
                        font=dict(size=10, color="red")
                    )
                ]
            )
            
            return fig
            
        except Exception as e:
            print(f"Error: Even fallback plot creation failed: {e}")
            return self._create_error_plot(f"Multiple failures: {error_msg}, {e}")
    
    def _create_error_plot(self, error_msg):
        """
        Create a minimal error plot as last resort.
        
        Args:
            error_msg: Error message to display
            
        Returns:
            plotly.graph_objects.Figure: Minimal error figure
        """
        try:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=[0], y=[0],
                mode='markers+text',
                marker=dict(size=80, color='red', symbol='x'),
                text=["ERROR"],
                textposition="middle center",
                name='Error'
            ))
            
            fig.update_layout(
                title="Debug Visualization Error",
                showlegend=False,
                xaxis=dict(range=[-1, 1], showgrid=False, showticklabels=False),
                yaxis=dict(range=[-1, 1], showgrid=False, showticklabels=False),
                annotations=[
                    dict(
                        text=f"Error: {str(error_msg)[:200]}",
                        showarrow=False,
                        x=0, y=-0.5,
                        font=dict(size=10, color="red")
                    )
                ]
            )
            
            return fig
            
        except Exception:
            # Absolute last resort - return empty figure
            return go.Figure()
