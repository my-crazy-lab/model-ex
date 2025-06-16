"""
State Manager for Conversational Agents

This module provides sophisticated state management for multi-step
conversational processes with transition logic and validation.
"""

from typing import Dict, Any, Optional, List, Callable, Set
from enum import Enum
from dataclasses import dataclass
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class StateTransitionError(Exception):
    """Raised when an invalid state transition is attempted"""
    pass


class StateValidationError(Exception):
    """Raised when state validation fails"""
    pass


@dataclass
class StateTransition:
    """Defines a state transition with conditions and actions"""
    from_state: Enum
    to_state: Enum
    condition: Optional[Callable] = None
    action: Optional[Callable] = None
    description: str = ""
    
    def can_transition(self, context: Any) -> bool:
        """Check if transition is allowed given current context"""
        if self.condition is None:
            return True
        try:
            return self.condition(context)
        except Exception as e:
            logger.warning(f"Transition condition failed: {e}")
            return False
    
    def execute_action(self, context: Any) -> Any:
        """Execute transition action if defined"""
        if self.action:
            try:
                return self.action(context)
            except Exception as e:
                logger.error(f"Transition action failed: {e}")
                raise StateTransitionError(f"Action execution failed: {e}")
        return None


class StateMachine:
    """
    Generic state machine for managing conversation flow
    
    Provides:
    - State transition validation
    - Conditional transitions
    - State entry/exit actions
    - Transition history tracking
    """
    
    def __init__(self, initial_state: Enum):
        self.current_state = initial_state
        self.previous_state = None
        self.state_history: List[tuple] = [(initial_state, datetime.now())]
        
        # Transition definitions
        self.transitions: Dict[tuple, StateTransition] = {}
        self.valid_transitions: Dict[Enum, Set[Enum]] = {}
        
        # State actions
        self.entry_actions: Dict[Enum, Callable] = {}
        self.exit_actions: Dict[Enum, Callable] = {}
        
        # State validators
        self.state_validators: Dict[Enum, Callable] = {}
    
    def add_transition(self, transition: StateTransition):
        """Add a state transition to the machine"""
        key = (transition.from_state, transition.to_state)
        self.transitions[key] = transition
        
        # Update valid transitions map
        if transition.from_state not in self.valid_transitions:
            self.valid_transitions[transition.from_state] = set()
        self.valid_transitions[transition.from_state].add(transition.to_state)
        
        logger.debug(f"Added transition: {transition.from_state} -> {transition.to_state}")
    
    def add_entry_action(self, state: Enum, action: Callable):
        """Add action to execute when entering a state"""
        self.entry_actions[state] = action
    
    def add_exit_action(self, state: Enum, action: Callable):
        """Add action to execute when exiting a state"""
        self.exit_actions[state] = action
    
    def add_state_validator(self, state: Enum, validator: Callable):
        """Add validator for state data"""
        self.state_validators[state] = validator
    
    def can_transition_to(self, target_state: Enum, context: Any = None) -> bool:
        """Check if transition to target state is possible"""
        # Check if transition is defined
        if target_state not in self.valid_transitions.get(self.current_state, set()):
            return False
        
        # Check transition condition
        transition = self.transitions.get((self.current_state, target_state))
        if transition and not transition.can_transition(context):
            return False
        
        return True
    
    def transition_to(self, target_state: Enum, context: Any = None) -> bool:
        """
        Attempt to transition to target state
        
        Args:
            target_state: State to transition to
            context: Context object for condition checking and actions
            
        Returns:
            bool: True if transition successful
            
        Raises:
            StateTransitionError: If transition is invalid
        """
        if not self.can_transition_to(target_state, context):
            raise StateTransitionError(
                f"Invalid transition from {self.current_state} to {target_state}"
            )
        
        # Execute exit action for current state
        if self.current_state in self.exit_actions:
            try:
                self.exit_actions[self.current_state](context)
            except Exception as e:
                logger.error(f"Exit action failed for {self.current_state}: {e}")
        
        # Execute transition action
        transition = self.transitions.get((self.current_state, target_state))
        if transition:
            try:
                transition.execute_action(context)
            except StateTransitionError:
                raise
            except Exception as e:
                raise StateTransitionError(f"Transition action failed: {e}")
        
        # Update state
        self.previous_state = self.current_state
        self.current_state = target_state
        self.state_history.append((target_state, datetime.now()))
        
        # Execute entry action for new state
        if target_state in self.entry_actions:
            try:
                self.entry_actions[target_state](context)
            except Exception as e:
                logger.error(f"Entry action failed for {target_state}: {e}")
        
        logger.info(f"State transition: {self.previous_state} -> {self.current_state}")
        return True
    
    def validate_current_state(self, context: Any) -> bool:
        """Validate current state data"""
        validator = self.state_validators.get(self.current_state)
        if validator:
            try:
                return validator(context)
            except Exception as e:
                logger.error(f"State validation failed for {self.current_state}: {e}")
                return False
        return True
    
    def get_possible_transitions(self) -> Set[Enum]:
        """Get all possible transitions from current state"""
        return self.valid_transitions.get(self.current_state, set())
    
    def get_state_history(self) -> List[tuple]:
        """Get complete state transition history"""
        return self.state_history.copy()
    
    def reset_to_initial(self, initial_state: Enum):
        """Reset state machine to initial state"""
        self.previous_state = self.current_state
        self.current_state = initial_state
        self.state_history.append((initial_state, datetime.now()))


class ConversationStateManager:
    """
    Specialized state manager for conversation flows
    
    Manages the conversation state machine with built-in support for:
    - Information collection phases
    - Service suggestion workflows
    - Booking confirmation processes
    - Error recovery mechanisms
    """
    
    def __init__(self, conversation_states: Enum, initial_state: Enum):
        self.state_machine = StateMachine(initial_state)
        self.conversation_states = conversation_states
        
        # Information collection tracking
        self.required_info: Dict[str, bool] = {}
        self.collected_info: Dict[str, Any] = {}
        self.info_collection_order: List[str] = []
        
        # Service selection tracking
        self.suggested_services: List[str] = []
        self.selected_services: List[str] = []
        
        # Booking tracking
        self.booking_data: Dict[str, Any] = {}
        self.confirmation_pending: bool = False
        
        self._setup_default_transitions()
    
    def _setup_default_transitions(self):
        """Setup default conversation flow transitions"""
        # This will be customized by specific implementations
        pass
    
    def set_required_information(self, info_fields: List[str]):
        """Define what information needs to be collected"""
        self.required_info = {field: False for field in info_fields}
        self.info_collection_order = info_fields.copy()
    
    def mark_info_collected(self, info_type: str, value: Any) -> bool:
        """Mark information as collected and store value"""
        if info_type in self.required_info:
            self.required_info[info_type] = True
            self.collected_info[info_type] = value
            logger.debug(f"Collected {info_type}: {value}")
            return True
        return False
    
    def is_info_complete(self) -> bool:
        """Check if all required information has been collected"""
        return all(self.required_info.values())
    
    def get_missing_info(self) -> List[str]:
        """Get list of missing required information"""
        return [info for info, collected in self.required_info.items() if not collected]
    
    def get_next_required_info(self) -> Optional[str]:
        """Get next information field to collect in order"""
        for info_field in self.info_collection_order:
            if not self.required_info.get(info_field, False):
                return info_field
        return None
    
    def add_suggested_service(self, service_id: str):
        """Add a service to suggestions"""
        if service_id not in self.suggested_services:
            self.suggested_services.append(service_id)
    
    def select_service(self, service_id: str) -> bool:
        """Select a suggested service"""
        if service_id in self.suggested_services:
            if service_id not in self.selected_services:
                self.selected_services.append(service_id)
            return True
        return False
    
    def deselect_service(self, service_id: str) -> bool:
        """Deselect a service"""
        if service_id in self.selected_services:
            self.selected_services.remove(service_id)
            return True
        return False
    
    def set_booking_data(self, key: str, value: Any):
        """Set booking-related data"""
        self.booking_data[key] = value
    
    def get_booking_data(self, key: str, default: Any = None) -> Any:
        """Get booking-related data"""
        return self.booking_data.get(key, default)
    
    def set_confirmation_pending(self, pending: bool = True):
        """Set confirmation pending status"""
        self.confirmation_pending = pending
    
    def is_confirmation_pending(self) -> bool:
        """Check if confirmation is pending"""
        return self.confirmation_pending
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of conversation state"""
        return {
            "current_state": self.state_machine.current_state.value,
            "previous_state": self.state_machine.previous_state.value if self.state_machine.previous_state else None,
            "collected_info": self.collected_info.copy(),
            "missing_info": self.get_missing_info(),
            "selected_services": self.selected_services.copy(),
            "booking_data": self.booking_data.copy(),
            "confirmation_pending": self.confirmation_pending,
            "state_history": [(state.value, timestamp.isoformat()) for state, timestamp in self.state_machine.state_history]
        }
    
    def reset_conversation(self, initial_state: Enum):
        """Reset conversation to initial state"""
        self.state_machine.reset_to_initial(initial_state)
        self.required_info = {field: False for field in self.required_info.keys()}
        self.collected_info.clear()
        self.suggested_services.clear()
        self.selected_services.clear()
        self.booking_data.clear()
        self.confirmation_pending = False
    
    # Delegate state machine methods
    def can_transition_to(self, target_state: Enum, context: Any = None) -> bool:
        return self.state_machine.can_transition_to(target_state, context)
    
    def transition_to(self, target_state: Enum, context: Any = None) -> bool:
        return self.state_machine.transition_to(target_state, context)
    
    def get_current_state(self) -> Enum:
        return self.state_machine.current_state
    
    def get_previous_state(self) -> Optional[Enum]:
        return self.state_machine.previous_state
    
    def add_transition(self, transition: StateTransition):
        self.state_machine.add_transition(transition)
    
    def add_entry_action(self, state: Enum, action: Callable):
        self.state_machine.add_entry_action(state, action)
    
    def add_exit_action(self, state: Enum, action: Callable):
        self.state_machine.add_exit_action(state, action)
