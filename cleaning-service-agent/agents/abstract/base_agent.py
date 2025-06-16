"""
Abstract Base Agent for Conversational Service Booking

This module provides the foundation for building conversational agents
that handle multi-step service booking processes.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
import logging
import time
from datetime import datetime

logger = logging.getLogger(__name__)


class ConversationState(Enum):
    """Defines the main conversation states"""
    INITIAL = "initial"
    COLLECTING_INFO = "collecting_info"
    INFO_COMPLETE = "info_complete"
    SUGGESTING_SERVICES = "suggesting_services"
    CALCULATING_PRICE = "calculating_price"
    CONFIRMING_BOOKING = "confirming_booking"
    BOOKING_COMPLETE = "booking_complete"
    ERROR_RECOVERY = "error_recovery"
    TERMINATED = "terminated"


class AgentResponse:
    """Structured response from the agent"""
    
    def __init__(
        self,
        message: str,
        state: ConversationState,
        data: Optional[Dict[str, Any]] = None,
        actions: Optional[List[str]] = None,
        confidence: float = 1.0
    ):
        self.message = message
        self.state = state
        self.data = data or {}
        self.actions = actions or []
        self.confidence = confidence
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary format"""
        return {
            "message": self.message,
            "state": self.state.value,
            "data": self.data,
            "actions": self.actions,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat()
        }


class ConversationContext:
    """Manages conversation context and history"""
    
    def __init__(self, session_id: str, customer_id: Optional[str] = None):
        self.session_id = session_id
        self.customer_id = customer_id
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        
        # Conversation history
        self.messages: List[Dict[str, Any]] = []
        self.extracted_entities: Dict[str, Any] = {}
        self.user_preferences: Dict[str, Any] = {}
        
        # State tracking
        self.current_state = ConversationState.INITIAL
        self.previous_state = None
        self.state_history: List[Tuple[ConversationState, datetime]] = []
        
        # Metadata
        self.metadata: Dict[str, Any] = {}
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add a message to conversation history"""
        message = {
            "role": role,  # "user" or "agent"
            "content": content,
            "timestamp": datetime.now(),
            "metadata": metadata or {}
        }
        self.messages.append(message)
        self.last_activity = datetime.now()
    
    def update_state(self, new_state: ConversationState):
        """Update conversation state"""
        self.previous_state = self.current_state
        self.current_state = new_state
        self.state_history.append((new_state, datetime.now()))
        logger.debug(f"State transition: {self.previous_state} -> {new_state}")
    
    def add_entity(self, entity_type: str, value: Any, confidence: float = 1.0):
        """Add extracted entity to context"""
        self.extracted_entities[entity_type] = {
            "value": value,
            "confidence": confidence,
            "extracted_at": datetime.now()
        }
    
    def get_entity(self, entity_type: str) -> Optional[Any]:
        """Get extracted entity value"""
        entity = self.extracted_entities.get(entity_type)
        return entity["value"] if entity else None
    
    def is_expired(self, timeout_minutes: int = 30) -> bool:
        """Check if conversation has expired"""
        elapsed = (datetime.now() - self.last_activity).total_seconds() / 60
        return elapsed > timeout_minutes


class AbstractConversationalAgent(ABC):
    """
    Abstract base class for conversational service booking agents
    
    This class provides the framework for building agents that can:
    1. Process natural language input
    2. Maintain conversation state
    3. Extract and validate information
    4. Guide users through multi-step processes
    5. Generate appropriate responses
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self.default_config()
        self.active_conversations: Dict[str, ConversationContext] = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize components
        self._initialize_components()
    
    @classmethod
    @abstractmethod
    def default_config(cls) -> Dict[str, Any]:
        """Return default configuration for the agent"""
        return {
            "max_conversation_turns": 20,
            "conversation_timeout_minutes": 30,
            "language": "vietnamese",
            "confidence_threshold": 0.7,
            "enable_logging": True,
            "enable_analytics": False
        }
    
    @abstractmethod
    def _initialize_components(self):
        """Initialize agent-specific components (NLP, pricing, etc.)"""
        pass
    
    def process_message(
        self,
        message: str,
        session_id: str,
        customer_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """
        Main entry point for processing user messages
        
        Args:
            message: User input message
            session_id: Unique session identifier
            customer_id: Optional customer identifier
            context: Additional context information
            
        Returns:
            AgentResponse: Structured response from agent
        """
        try:
            # Get or create conversation context
            conversation = self._get_or_create_conversation(session_id, customer_id)
            
            # Check if conversation has expired
            if conversation.is_expired(self.config["conversation_timeout_minutes"]):
                return self._handle_expired_conversation(conversation)
            
            # Add user message to history
            conversation.add_message("user", message, context)
            
            # Process the message
            response = self._process_message_internal(message, conversation, context)
            
            # Add agent response to history
            conversation.add_message("agent", response.message, {
                "state": response.state.value,
                "confidence": response.confidence
            })
            
            # Update conversation state
            conversation.update_state(response.state)
            
            # Log interaction
            self._log_interaction(conversation, message, response)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing message: {str(e)}")
            return self._handle_error(session_id, str(e))
    
    @abstractmethod
    def _process_message_internal(
        self,
        message: str,
        conversation: ConversationContext,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """
        Internal message processing logic - to be implemented by subclasses
        
        Args:
            message: User input message
            conversation: Current conversation context
            context: Additional context information
            
        Returns:
            AgentResponse: Processed response
        """
        pass
    
    def _get_or_create_conversation(
        self,
        session_id: str,
        customer_id: Optional[str] = None
    ) -> ConversationContext:
        """Get existing conversation or create new one"""
        if session_id not in self.active_conversations:
            self.active_conversations[session_id] = ConversationContext(
                session_id=session_id,
                customer_id=customer_id
            )
        return self.active_conversations[session_id]
    
    def _handle_expired_conversation(self, conversation: ConversationContext) -> AgentResponse:
        """Handle expired conversation"""
        conversation.update_state(ConversationState.TERMINATED)
        return AgentResponse(
            message="Phiên trò chuyện đã hết hạn. Vui lòng bắt đầu lại để đặt dịch vụ.",
            state=ConversationState.TERMINATED,
            confidence=1.0
        )
    
    def _handle_error(self, session_id: str, error_message: str) -> AgentResponse:
        """Handle processing errors"""
        self.logger.error(f"Session {session_id}: {error_message}")
        return AgentResponse(
            message="Xin lỗi, đã có lỗi xảy ra. Vui lòng thử lại hoặc liên hệ hỗ trợ.",
            state=ConversationState.ERROR_RECOVERY,
            data={"error": error_message},
            confidence=0.0
        )
    
    def _log_interaction(
        self,
        conversation: ConversationContext,
        user_message: str,
        agent_response: AgentResponse
    ):
        """Log conversation interaction"""
        if self.config.get("enable_logging", True):
            self.logger.info(
                f"Session {conversation.session_id}: "
                f"State={agent_response.state.value}, "
                f"Confidence={agent_response.confidence:.2f}"
            )
    
    def get_conversation_history(self, session_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get conversation history for a session"""
        conversation = self.active_conversations.get(session_id)
        return conversation.messages if conversation else None
    
    def end_conversation(self, session_id: str) -> bool:
        """End and cleanup conversation"""
        if session_id in self.active_conversations:
            conversation = self.active_conversations[session_id]
            conversation.update_state(ConversationState.TERMINATED)
            del self.active_conversations[session_id]
            return True
        return False
    
    def get_active_conversations_count(self) -> int:
        """Get number of active conversations"""
        return len(self.active_conversations)
    
    def cleanup_expired_conversations(self):
        """Remove expired conversations"""
        timeout_minutes = self.config["conversation_timeout_minutes"]
        expired_sessions = [
            session_id for session_id, conversation in self.active_conversations.items()
            if conversation.is_expired(timeout_minutes)
        ]
        
        for session_id in expired_sessions:
            self.end_conversation(session_id)
        
        if expired_sessions:
            self.logger.info(f"Cleaned up {len(expired_sessions)} expired conversations")


class AbstractServiceBookingAgent(AbstractConversationalAgent):
    """
    Abstract base class specifically for service booking agents
    
    Extends the conversational agent with service booking specific functionality
    """
    
    @abstractmethod
    def collect_required_information(
        self,
        conversation: ConversationContext,
        message: str
    ) -> AgentResponse:
        """Collect required information for service booking"""
        pass
    
    @abstractmethod
    def suggest_additional_services(
        self,
        conversation: ConversationContext,
        message: str
    ) -> AgentResponse:
        """Suggest additional services to customer"""
        pass
    
    @abstractmethod
    def calculate_pricing(
        self,
        conversation: ConversationContext,
        message: str
    ) -> AgentResponse:
        """Calculate total service pricing"""
        pass
    
    @abstractmethod
    def confirm_booking(
        self,
        conversation: ConversationContext,
        message: str
    ) -> AgentResponse:
        """Confirm and finalize booking"""
        pass
    
    @abstractmethod
    def validate_required_information(self, conversation: ConversationContext) -> bool:
        """Check if all required information has been collected"""
        pass
