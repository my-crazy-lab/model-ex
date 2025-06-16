"""
Cleaning Service Booking Agent

Main implementation of the conversational agent for cleaning service bookings.
Handles the complete workflow from information collection to booking confirmation.
"""

import re
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime, timedelta

from ..abstract.base_agent import (
    AbstractServiceBookingAgent, 
    AgentResponse, 
    ConversationContext,
    ConversationState
)
from ..abstract.state_manager import ConversationStateManager, StateTransition
from .states import CleaningServiceState, InfoCollectionState
from .actions import CleaningServiceActions
from .pricing import PricingCalculator
from ..nlp.intent_classifier import IntentClassifier
from ..nlp.entity_extractor import EntityExtractor
from ..nlp.response_generator import ResponseGenerator

logger = logging.getLogger(__name__)


class CleaningServiceAgent(AbstractServiceBookingAgent):
    """
    Intelligent conversational agent for cleaning service bookings
    
    Features:
    - Multi-step information collection
    - Smart service recommendations
    - Dynamic pricing calculation
    - Booking confirmation and scheduling
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Initialize state managers for each conversation
        self.conversation_state_managers: Dict[str, ConversationStateManager] = {}
        
        logger.info("CleaningServiceAgent initialized successfully")
    
    @classmethod
    def default_config(cls) -> Dict[str, Any]:
        """Default configuration for cleaning service agent"""
        base_config = super().default_config()
        base_config.update({
            # Agent-specific settings
            "service_type": "cleaning",
            "enable_additional_services": True,
            "enable_dynamic_pricing": True,
            "enable_scheduling": True,
            
            # Required information fields
            "required_info_fields": [
                "address",
                "area_m2", 
                "house_type",
                "preferred_time"
            ],
            
            # Service recommendations
            "max_additional_services": 4,
            "recommendation_threshold": 0.6,
            
            # Pricing settings
            "pricing_model": "standard",
            "enable_discounts": True,
            "minimum_service_fee": 100000,  # VND
            
            # Scheduling settings
            "booking_advance_hours": 24,
            "available_time_slots": [
                "08:00-12:00", "13:00-17:00", "18:00-22:00"
            ],
            
            # NLP settings
            "intent_confidence_threshold": 0.7,
            "entity_confidence_threshold": 0.6,
            "language": "vietnamese"
        })
        return base_config
    
    def _initialize_components(self):
        """Initialize agent components"""
        # NLP components
        self.intent_classifier = IntentClassifier(self.config)
        self.entity_extractor = EntityExtractor(self.config)
        self.response_generator = ResponseGenerator(self.config)
        
        # Business logic components
        self.pricing_calculator = PricingCalculator(self.config)
        self.actions = CleaningServiceActions(self.config)
        
        logger.debug("All components initialized")
    
    def _get_state_manager(self, session_id: str) -> ConversationStateManager:
        """Get or create state manager for conversation"""
        if session_id not in self.conversation_state_managers:
            state_manager = ConversationStateManager(
                CleaningServiceState, 
                CleaningServiceState.INITIAL
            )
            self._setup_state_transitions(state_manager)
            self.conversation_state_managers[session_id] = state_manager
        
        return self.conversation_state_managers[session_id]
    
    def _setup_state_transitions(self, state_manager: ConversationStateManager):
        """Setup state machine transitions for cleaning service workflow"""
        # Set required information fields
        state_manager.set_required_information(self.config["required_info_fields"])
        
        # Define state transitions
        transitions = [
            # Initial to information collection
            StateTransition(
                CleaningServiceState.INITIAL,
                CleaningServiceState.COLLECTING_INFO,
                condition=lambda ctx: True,
                description="Start information collection"
            ),
            
            # Information collection to complete
            StateTransition(
                CleaningServiceState.COLLECTING_INFO,
                CleaningServiceState.INFO_COMPLETE,
                condition=lambda ctx: state_manager.is_info_complete(),
                description="All required information collected"
            ),
            
            # Info complete to service suggestions
            StateTransition(
                CleaningServiceState.INFO_COMPLETE,
                CleaningServiceState.SUGGESTING_SERVICES,
                condition=lambda ctx: self.config["enable_additional_services"],
                description="Suggest additional services"
            ),
            
            # Service suggestions to pricing
            StateTransition(
                CleaningServiceState.SUGGESTING_SERVICES,
                CleaningServiceState.CALCULATING_PRICE,
                condition=lambda ctx: True,
                description="Calculate service pricing"
            ),
            
            # Direct from info complete to pricing (skip suggestions)
            StateTransition(
                CleaningServiceState.INFO_COMPLETE,
                CleaningServiceState.CALCULATING_PRICE,
                condition=lambda ctx: not self.config["enable_additional_services"],
                description="Skip to pricing calculation"
            ),
            
            # Pricing to booking confirmation
            StateTransition(
                CleaningServiceState.CALCULATING_PRICE,
                CleaningServiceState.CONFIRMING_BOOKING,
                condition=lambda ctx: True,
                description="Confirm booking details"
            ),
            
            # Booking confirmation to complete
            StateTransition(
                CleaningServiceState.CONFIRMING_BOOKING,
                CleaningServiceState.BOOKING_COMPLETE,
                condition=lambda ctx: True,
                description="Complete booking process"
            ),
            
            # Error recovery transitions
            StateTransition(
                CleaningServiceState.ERROR_RECOVERY,
                CleaningServiceState.COLLECTING_INFO,
                condition=lambda ctx: True,
                description="Restart information collection"
            )
        ]
        
        # Add all transitions to state manager
        for transition in transitions:
            state_manager.add_transition(transition)
    
    def _process_message_internal(
        self,
        message: str,
        conversation: ConversationContext,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """Internal message processing logic"""
        
        # Get state manager for this conversation
        state_manager = self._get_state_manager(conversation.session_id)
        current_state = state_manager.get_current_state()
        
        # Classify intent and extract entities
        intent = self.intent_classifier.classify(message, conversation)
        entities = self.entity_extractor.extract(message, conversation)
        
        # Store extracted entities in state manager
        for entity_type, entity_data in entities.items():
            if entity_data["confidence"] >= self.config["entity_confidence_threshold"]:
                state_manager.mark_info_collected(entity_type, entity_data["value"])
                conversation.add_entity(entity_type, entity_data["value"], entity_data["confidence"])
        
        # Route to appropriate handler based on current state
        try:
            if current_state == CleaningServiceState.INITIAL:
                response = self._handle_initial_state(message, conversation, state_manager, intent)
            
            elif current_state == CleaningServiceState.COLLECTING_INFO:
                response = self.collect_required_information(conversation, message)
            
            elif current_state == CleaningServiceState.INFO_COMPLETE:
                response = self._handle_info_complete(message, conversation, state_manager)
            
            elif current_state == CleaningServiceState.SUGGESTING_SERVICES:
                response = self.suggest_additional_services(conversation, message)
            
            elif current_state == CleaningServiceState.CALCULATING_PRICE:
                response = self.calculate_pricing(conversation, message)
            
            elif current_state == CleaningServiceState.CONFIRMING_BOOKING:
                response = self.confirm_booking(conversation, message)
            
            elif current_state == CleaningServiceState.BOOKING_COMPLETE:
                response = self._handle_booking_complete(message, conversation, state_manager)
            
            else:
                response = self._handle_unknown_state(message, conversation, state_manager)
            
            return response
            
        except Exception as e:
            logger.error(f"Error in message processing: {str(e)}")
            return self._create_error_response(str(e))
    
    def _handle_initial_state(
        self, 
        message: str, 
        conversation: ConversationContext,
        state_manager: ConversationStateManager,
        intent: Dict[str, Any]
    ) -> AgentResponse:
        """Handle initial conversation state"""
        
        # Check if user wants to book cleaning service
        if intent.get("intent") in ["book_service", "cleaning_request", "greeting"]:
            # Transition to information collection
            state_manager.transition_to(CleaningServiceState.COLLECTING_INFO)
            
            response_message = self.response_generator.generate_welcome_message()
            
            return AgentResponse(
                message=response_message,
                state=CleaningServiceState.COLLECTING_INFO,
                data={"next_action": "collect_address"},
                confidence=intent.get("confidence", 0.8)
            )
        else:
            # Handle other intents or provide help
            response_message = self.response_generator.generate_help_message()
            
            return AgentResponse(
                message=response_message,
                state=CleaningServiceState.INITIAL,
                data={"available_actions": ["book_service", "get_info", "contact_support"]},
                confidence=0.9
            )
    
    def _handle_info_complete(
        self,
        message: str,
        conversation: ConversationContext,
        state_manager: ConversationStateManager
    ) -> AgentResponse:
        """Handle transition from info complete state"""
        
        if self.config["enable_additional_services"]:
            # Transition to service suggestions
            state_manager.transition_to(CleaningServiceState.SUGGESTING_SERVICES)
            return self.suggest_additional_services(conversation, message)
        else:
            # Skip directly to pricing
            state_manager.transition_to(CleaningServiceState.CALCULATING_PRICE)
            return self.calculate_pricing(conversation, message)
    
    def _handle_booking_complete(
        self,
        message: str,
        conversation: ConversationContext,
        state_manager: ConversationStateManager
    ) -> AgentResponse:
        """Handle completed booking state"""
        
        booking_data = state_manager.get_booking_data("confirmation")
        
        response_message = self.response_generator.generate_completion_message(booking_data)
        
        return AgentResponse(
            message=response_message,
            state=CleaningServiceState.BOOKING_COMPLETE,
            data={
                "booking_confirmed": True,
                "booking_id": booking_data.get("booking_id"),
                "next_steps": ["wait_for_service", "contact_if_needed"]
            },
            confidence=1.0
        )
    
    def _handle_unknown_state(
        self,
        message: str,
        conversation: ConversationContext,
        state_manager: ConversationStateManager
    ) -> AgentResponse:
        """Handle unknown or error states"""
        
        logger.warning(f"Unknown state: {state_manager.get_current_state()}")
        
        # Transition to error recovery
        state_manager.transition_to(CleaningServiceState.ERROR_RECOVERY)
        
        response_message = "Xin lỗi, có vẻ như đã xảy ra lỗi. Hãy để tôi giúp bạn bắt đầu lại quá trình đặt dịch vụ."
        
        return AgentResponse(
            message=response_message,
            state=CleaningServiceState.ERROR_RECOVERY,
            data={"error": "unknown_state", "recovery_action": "restart"},
            confidence=0.5
        )
    
    def _create_error_response(self, error_message: str) -> AgentResponse:
        """Create standardized error response"""
        return AgentResponse(
            message="Xin lỗi, đã có lỗi xảy ra trong quá trình xử lý. Vui lòng thử lại.",
            state=CleaningServiceState.ERROR_RECOVERY,
            data={"error": error_message},
            confidence=0.0
        )
    
    def collect_required_information(
        self,
        conversation: ConversationContext,
        message: str
    ) -> AgentResponse:
        """Collect required information for service booking"""
        state_manager = self._get_state_manager(conversation.session_id)

        # Check what information is still needed
        missing_info = state_manager.get_missing_info()
        next_info = state_manager.get_next_required_info()

        if not missing_info:
            # All information collected, transition to next state
            state_manager.transition_to(CleaningServiceState.INFO_COMPLETE)
            return self._handle_info_complete(message, conversation, state_manager)

        # Generate response asking for next required information
        response_message = self.response_generator.generate_info_collection_prompt(
            next_info,
            state_manager.collected_info
        )

        return AgentResponse(
            message=response_message,
            state=CleaningServiceState.COLLECTING_INFO,
            data={
                "next_required": next_info,
                "missing_info": missing_info,
                "collected_info": state_manager.collected_info
            },
            confidence=0.9
        )

    def suggest_additional_services(
        self,
        conversation: ConversationContext,
        message: str
    ) -> AgentResponse:
        """Suggest additional services to customer"""
        state_manager = self._get_state_manager(conversation.session_id)

        # Get house characteristics for recommendations
        house_type = state_manager.collected_info.get("house_type", "")
        area_m2 = state_manager.collected_info.get("area_m2", 0)

        # Get service recommendations
        from .states import AdditionalServices
        recommendations = AdditionalServices.get_service_recommendations(
            house_type, area_m2, self.config["max_additional_services"]
        )

        # Store recommendations in state manager
        for service_id in recommendations:
            state_manager.add_suggested_service(service_id)

        # Generate response with service suggestions
        response_message = self.response_generator.generate_service_suggestions(
            recommendations, house_type, area_m2
        )

        # Check if user wants to proceed to pricing
        intent = self.intent_classifier.classify(message, conversation)
        if intent.get("intent") in ["skip_services", "proceed_pricing", "no_additional"]:
            state_manager.transition_to(CleaningServiceState.CALCULATING_PRICE)
            return self.calculate_pricing(conversation, message)

        return AgentResponse(
            message=response_message,
            state=CleaningServiceState.SUGGESTING_SERVICES,
            data={
                "suggested_services": recommendations,
                "available_actions": ["select_service", "skip_services", "get_details"]
            },
            confidence=0.8
        )

    def calculate_pricing(
        self,
        conversation: ConversationContext,
        message: str
    ) -> AgentResponse:
        """Calculate total service pricing"""
        state_manager = self._get_state_manager(conversation.session_id)

        # Get collected information
        area_m2 = state_manager.collected_info.get("area_m2", 0)
        house_type = state_manager.collected_info.get("house_type", "")
        selected_services = state_manager.selected_services

        # Calculate pricing
        pricing_result = self.pricing_calculator.calculate_total_price(
            area_m2=area_m2,
            house_type=house_type,
            additional_services=selected_services,
            customer_type="new"  # Could be determined from conversation context
        )

        # Store pricing in state manager
        state_manager.set_booking_data("pricing", pricing_result)

        # Generate pricing response
        response_message = self.response_generator.generate_pricing_summary(
            pricing_result, state_manager.collected_info
        )

        # Transition to booking confirmation
        state_manager.transition_to(CleaningServiceState.CONFIRMING_BOOKING)

        return AgentResponse(
            message=response_message,
            state=CleaningServiceState.CALCULATING_PRICE,
            data={
                "pricing": pricing_result,
                "next_action": "confirm_booking"
            },
            confidence=0.9
        )

    def confirm_booking(
        self,
        conversation: ConversationContext,
        message: str
    ) -> AgentResponse:
        """Confirm and finalize booking"""
        state_manager = self._get_state_manager(conversation.session_id)

        # Check user intent
        intent = self.intent_classifier.classify(message, conversation)

        if intent.get("intent") in ["confirm", "yes", "agree", "book_now"]:
            # User confirmed booking
            booking_result = self.actions.create_booking(
                customer_info=state_manager.collected_info,
                selected_services=state_manager.selected_services,
                pricing=state_manager.get_booking_data("pricing"),
                session_id=conversation.session_id
            )

            # Store booking confirmation
            state_manager.set_booking_data("confirmation", booking_result)
            state_manager.transition_to(CleaningServiceState.BOOKING_COMPLETE)

            response_message = self.response_generator.generate_booking_confirmation(
                booking_result
            )

            return AgentResponse(
                message=response_message,
                state=CleaningServiceState.BOOKING_COMPLETE,
                data={
                    "booking_confirmed": True,
                    "booking_id": booking_result.get("booking_id"),
                    "confirmation": booking_result
                },
                confidence=1.0
            )

        elif intent.get("intent") in ["cancel", "no", "decline"]:
            # User declined booking
            response_message = "Cảm ơn bạn đã quan tâm đến dịch vụ. Nếu có nhu cầu khác, vui lòng liên hệ lại với chúng tôi."

            state_manager.transition_to(CleaningServiceState.TERMINATED)

            return AgentResponse(
                message=response_message,
                state=CleaningServiceState.TERMINATED,
                data={"booking_cancelled": True},
                confidence=0.9
            )

        else:
            # Ask for clarification
            response_message = "Bạn có muốn xác nhận đặt dịch vụ với giá trên không? Vui lòng trả lời 'có' hoặc 'không'."

            return AgentResponse(
                message=response_message,
                state=CleaningServiceState.CONFIRMING_BOOKING,
                data={"awaiting_confirmation": True},
                confidence=0.7
            )

    def validate_required_information(self, conversation: ConversationContext) -> bool:
        """Check if all required information has been collected"""
        state_manager = self._get_state_manager(conversation.session_id)
        return state_manager.is_info_complete()
