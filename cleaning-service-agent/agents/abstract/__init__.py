"""
Abstract Base Classes for Conversational Agents
"""

from .base_agent import (
    AbstractConversationalAgent,
    AbstractServiceBookingAgent,
    AgentResponse,
    ConversationContext,
    ConversationState
)
from .state_manager import (
    ConversationStateManager,
    StateMachine,
    StateTransition,
    StateTransitionError,
    StateValidationError
)

__all__ = [
    "AbstractConversationalAgent",
    "AbstractServiceBookingAgent", 
    "AgentResponse",
    "ConversationContext",
    "ConversationState",
    "ConversationStateManager",
    "StateMachine",
    "StateTransition",
    "StateTransitionError",
    "StateValidationError"
]
