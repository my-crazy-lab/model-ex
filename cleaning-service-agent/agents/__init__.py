"""
Cleaning Service Agent Package

This package provides intelligent conversational agents for cleaning service bookings.
"""

from .cleaning_agent.agent import CleaningServiceAgent
from .abstract.base_agent import (
    AbstractConversationalAgent,
    AbstractServiceBookingAgent,
    AgentResponse,
    ConversationContext,
    ConversationState
)

__version__ = "1.0.0"
__author__ = "Cleaning Service Team"

# Main exports
__all__ = [
    "CleaningServiceAgent",
    "AbstractConversationalAgent", 
    "AbstractServiceBookingAgent",
    "AgentResponse",
    "ConversationContext",
    "ConversationState"
]
