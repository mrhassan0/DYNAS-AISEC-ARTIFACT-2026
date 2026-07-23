"""Curriculum learning manager for progressive difficulty in training.

This module implements a curriculum learning system that gradually increases
the realism and difficulty of the environment during training, while ensuring
evaluation always uses the most difficult settings.
"""

import copy


class CurriculumManager:
    """Manages difficulty progression during training via curriculum stages.
    
    The curriculum manager controls the realism parameters (IDS, scan noise,
    network reliability, service dynamics) based on the current training epoch.
    During evaluation mode, it always returns the most difficult stage settings.
    
    Attributes
    ----------
    config : dict
        The curriculum configuration from the scenario
    training_mode : bool
        Whether the environment is in training or evaluation mode
    current_epoch : int
        The current epoch number during training
    """
    
    def __init__(self, curriculum_config, training_mode=True, total_epochs=None):
        """Initialize the curriculum manager.

        Parameters
        ----------
        curriculum_config : dict
            Dictionary containing curriculum stages and settings
        training_mode : bool, optional
            Whether in training mode (True) or evaluation mode (False).
            In evaluation mode, always uses the final/hardest stage.
            (default=True)
        total_epochs : int, optional
            Total number of training epochs for the whole run. When set, stages
            may declare ``start_frac``/``end_frac`` in [0, 1] instead of (or in
            addition to) absolute ``start_epoch``/``end_epoch``, and the manager
            resolves those fractions against this total. This makes a curriculum
            robust to the run length -- the same YAML ramps correctly whether the
            run is 50 or 500 epochs -- instead of hard-coding absolute epoch
            numbers that silently never trigger on shorter/longer runs.
            (default=None -> fall back to absolute epochs)
        """
        self.config = curriculum_config
        self.training_mode = training_mode
        self.total_epochs = total_epochs
        self.current_epoch = 0

        # Validate and sort stages by their resolved start epoch
        if 'stages' in self.config:
            self.config['stages'] = sorted(
                self.config['stages'],
                key=lambda x: self._stage_bounds(x)[0]
            )

    def _stage_bounds(self, stage):
        """Resolve a stage's [start, end) epoch window.

        Prefers fractional bounds (``start_frac``/``end_frac``) scaled by
        ``total_epochs`` when both are available, otherwise falls back to
        absolute ``start_epoch``/``end_epoch``. Returns (start_epoch, end_epoch)
        as numbers (end may be float('inf')).
        """
        if self.total_epochs and 'start_frac' in stage:
            start = int(round(float(stage['start_frac']) * self.total_epochs))
        else:
            start = stage.get('start_epoch', 0)

        if self.total_epochs and 'end_frac' in stage:
            end = int(round(float(stage['end_frac']) * self.total_epochs))
        else:
            end = stage.get('end_epoch', float('inf'))

        return start, end
    
    def update_epoch(self, epoch):
        """Update the current epoch number.
        
        This should be called at the start of each epoch during training.
        
        Parameters
        ----------
        epoch : int
            The current epoch number
        """
        self.current_epoch = epoch
    
    def get_current_stage(self):
        """Determine the current curriculum stage based on epoch number.
        
        In evaluation mode, always returns the final (most difficult) stage.
        In training mode, returns the stage corresponding to current epoch.
        
        Returns
        -------
        dict
            The stage configuration dictionary
        """
        if not self.training_mode:
            return self._get_final_stage()
        
        # Find the appropriate stage for current epoch
        if 'stages' not in self.config or len(self.config['stages']) == 0:
            return self._get_default_stage()
        
        for stage in self.config['stages']:
            start, end = self._stage_bounds(stage)

            if start <= self.current_epoch < end:
                return stage

        # If no stage matches, return the last stage
        return self.config['stages'][-1]
    
    def _get_final_stage(self):
        """Get the final (most difficult) stage for evaluation.
        
        Returns
        -------
        dict
            The final stage configuration
        """
        if 'stages' not in self.config or len(self.config['stages']) == 0:
            return self._get_default_stage()
        
        return self.config['stages'][-1]
    
    def _get_default_stage(self):
        """Get a default stage with no realism enabled.
        
        Returns
        -------
        dict
            Default stage configuration
        """
        return {
            'name': 'default',
            'start_epoch': 0,
            'end_epoch': float('inf'),
            'ids': {'enabled': False},
            'scan_noise': {
                'service_scan': {'false_positive_rate': 0.0, 'false_negative_rate': 0.0},
                'os_scan': {'false_positive_rate': 0.0, 'false_negative_rate': 0.0},
                'process_scan': {'false_positive_rate': 0.0, 'false_negative_rate': 0.0}
            },
            'network_reliability': {
                'timeout_probability': 0.0,
                'affected_actions': []
            },
            'service_dynamics': {
                'churn_probability': 0.0,
                'affected_services': []
            }
        }
    
    def get_realism_params(self):
        """Get the current realism parameters based on the active stage.
        
        Returns
        -------
        dict
            Dictionary containing:
            - ids_config: IDS configuration
            - scan_noise: Scan noise configuration
            - network_reliability: Network reliability configuration
            - service_dynamics: Service dynamics configuration
        """
        stage = self.get_current_stage()
        
        return {
            'ids_config': stage.get('ids', {'enabled': False}),
            'scan_noise': stage.get('scan_noise', {}),
            'network_reliability': stage.get('network_reliability', {}),
            'service_dynamics': stage.get('service_dynamics', {})
        }
    
    def get_stage_info(self):
        """Get information about the current stage.
        
        Returns
        -------
        dict
            Dictionary with stage information:
            - name: Stage name
            - epoch: Current epoch
            - start_epoch: Stage start epoch
            - end_epoch: Stage end epoch
            - training_mode: Whether in training mode
        """
        stage = self.get_current_stage()
        start, end = self._stage_bounds(stage)

        return {
            'name': stage.get('name', 'unknown'),
            'epoch': self.current_epoch,
            'start_epoch': start,
            'end_epoch': end,
            'training_mode': self.training_mode
        }
    
    def is_enabled(self):
        """Check if curriculum learning is enabled.
        
        Returns
        -------
        bool
            True if curriculum is enabled in config
        """
        return self.config.get('enabled', False)
    
    def get_stage_transition_epochs(self):
        """Get list of epoch numbers where stage transitions occur.
        
        Returns
        -------
        list of int
            Sorted list of epoch numbers where stages transition
        """
        if 'stages' not in self.config or len(self.config['stages']) == 0:
            return []
        
        transitions = set()
        for stage in self.config['stages']:
            start, _ = self._stage_bounds(stage)
            transitions.add(start)

        return sorted(transitions)

