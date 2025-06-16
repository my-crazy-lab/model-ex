"""
Intent Classification for Cleaning Service Agent

This module handles intent recognition from user messages to understand
what the user wants to do at each step of the conversation.
"""

import re
from typing import Dict, Any, List, Optional, Tuple
import logging
from ..abstract.base_agent import ConversationContext

logger = logging.getLogger(__name__)


class IntentClassifier:
    """
    Rule-based intent classifier for cleaning service conversations
    
    Recognizes intents such as:
    - Service booking requests
    - Information provision
    - Service selection/deselection
    - Confirmation/rejection
    - Questions and clarifications
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.confidence_threshold = config.get("intent_confidence_threshold", 0.7)
        self.language = config.get("language", "vietnamese")
        
        # Initialize intent patterns
        self._initialize_intent_patterns()
        
        logger.debug("IntentClassifier initialized")
    
    def _initialize_intent_patterns(self):
        """Initialize intent recognition patterns"""
        
        # Booking and service request intents
        self.booking_patterns = {
            "book_service": [
                r"đặt.*dịch vụ",
                r"book.*service",
                r"muốn.*dọn.*nhà",
                r"cần.*dọn dẹp",
                r"thuê.*người.*dọn",
                r"đặt lịch.*dọn",
                r"dọn dẹp.*nhà"
            ],
            "cleaning_request": [
                r"dọn dẹp",
                r"vệ sinh.*nhà",
                r"clean.*house",
                r"làm sạch",
                r"tổng vệ sinh"
            ]
        }
        
        # Information provision intents
        self.info_patterns = {
            "provide_address": [
                r"địa chỉ.*là",
                r"nhà.*ở",
                r"tôi ở",
                r"address.*is",
                r"\d+.*đường",
                r"\d+.*phố",
                r"quận.*\d+",
                r"huyện.*\w+",
                r"tp\.|thành phố"
            ],
            "provide_area": [
                r"\d+.*m2",
                r"\d+.*mét.*vuông",
                r"\d+.*square.*meter",
                r"diện tích.*\d+",
                r"khoảng.*\d+.*m",
                r"\d+.*m.*vuông"
            ],
            "provide_house_type": [
                r"chung cư",
                r"nhà riêng",
                r"villa",
                r"biệt thự",
                r"văn phòng",
                r"apartment",
                r"house",
                r"office"
            ],
            "provide_time": [
                r"sáng",
                r"chiều", 
                r"tối",
                r"morning",
                r"afternoon",
                r"evening",
                r"\d+.*giờ",
                r"\d+h",
                r"\d+-\d+h"
            ]
        }
        
        # Service selection intents
        self.service_patterns = {
            "select_service": [
                r"chọn.*nấu ăn",
                r"thêm.*ủi đồ",
                r"muốn.*thêm",
                r"có.*thêm",
                r"select.*cooking",
                r"add.*ironing",
                r"want.*additional"
            ],
            "skip_services": [
                r"không.*cần.*thêm",
                r"bỏ qua",
                r"skip",
                r"không.*muốn.*thêm",
                r"chỉ.*dọn dẹp.*thôi",
                r"không.*dịch vụ.*khác"
            ]
        }
        
        # Confirmation intents
        self.confirmation_patterns = {
            "confirm": [
                r"^có$",
                r"^yes$",
                r"^ok$",
                r"^được$",
                r"đồng ý",
                r"xác nhận",
                r"confirm",
                r"agree",
                r"chấp nhận"
            ],
            "reject": [
                r"^không$",
                r"^no$",
                r"từ chối",
                r"không đồng ý",
                r"cancel",
                r"hủy",
                r"reject"
            ]
        }
        
        # Question and clarification intents
        self.question_patterns = {
            "ask_price": [
                r"giá.*bao nhiêu",
                r"chi phí.*là",
                r"how much",
                r"price.*is",
                r"cost.*how much",
                r"tính.*tiền.*thế nào"
            ],
            "ask_time": [
                r"mất.*bao lâu",
                r"how long",
                r"thời gian.*bao nhiêu",
                r"duration",
                r"kéo dài.*bao lâu"
            ],
            "ask_details": [
                r"chi tiết",
                r"details",
                r"thông tin.*thêm",
                r"more.*info",
                r"giải thích.*thêm"
            ]
        }
        
        # Greeting and help intents
        self.general_patterns = {
            "greeting": [
                r"xin chào",
                r"hello",
                r"hi",
                r"chào",
                r"good.*morning",
                r"good.*afternoon"
            ],
            "help": [
                r"giúp.*đỡ",
                r"help",
                r"hướng dẫn",
                r"guide",
                r"không.*biết.*làm",
                r"how.*to"
            ],
            "goodbye": [
                r"tạm biệt",
                r"goodbye",
                r"bye",
                r"cảm ơn.*tạm biệt",
                r"kết thúc"
            ]
        }
    
    def classify(self, message: str, conversation: ConversationContext) -> Dict[str, Any]:
        """
        Classify user intent from message
        
        Args:
            message: User input message
            conversation: Current conversation context
            
        Returns:
            Dictionary with intent, confidence, and metadata
        """
        message_lower = message.lower().strip()
        
        # Get conversation state for context-aware classification
        current_state = getattr(conversation, 'current_state', None)
        
        # Try to match patterns in order of priority
        intent_results = []
        
        # Check all pattern categories
        pattern_categories = [
            ("booking", self.booking_patterns),
            ("info", self.info_patterns),
            ("service", self.service_patterns),
            ("confirmation", self.confirmation_patterns),
            ("question", self.question_patterns),
            ("general", self.general_patterns)
        ]
        
        for category, patterns in pattern_categories:
            for intent, pattern_list in patterns.items():
                confidence = self._match_patterns(message_lower, pattern_list)
                if confidence > 0:
                    intent_results.append({
                        "intent": intent,
                        "category": category,
                        "confidence": confidence,
                        "matched_patterns": self._get_matched_patterns(message_lower, pattern_list)
                    })
        
        # Sort by confidence and apply context-aware adjustments
        intent_results.sort(key=lambda x: x["confidence"], reverse=True)
        
        if intent_results:
            best_intent = intent_results[0]
            
            # Apply context-aware confidence adjustments
            adjusted_confidence = self._adjust_confidence_by_context(
                best_intent, current_state, conversation
            )
            
            return {
                "intent": best_intent["intent"],
                "category": best_intent["category"],
                "confidence": adjusted_confidence,
                "alternatives": intent_results[1:3],  # Top 2 alternatives
                "matched_patterns": best_intent["matched_patterns"]
            }
        
        # Default fallback intent
        return {
            "intent": "unknown",
            "category": "general",
            "confidence": 0.1,
            "alternatives": [],
            "matched_patterns": []
        }
    
    def _match_patterns(self, message: str, patterns: List[str]) -> float:
        """Match message against list of regex patterns"""
        matches = 0
        total_patterns = len(patterns)
        
        for pattern in patterns:
            if re.search(pattern, message, re.IGNORECASE):
                matches += 1
        
        # Calculate confidence based on match ratio
        if matches > 0:
            base_confidence = matches / total_patterns
            # Boost confidence for exact matches
            if matches == total_patterns:
                base_confidence = min(1.0, base_confidence * 1.2)
            return min(1.0, base_confidence * 0.8 + 0.2)  # Scale to 0.2-1.0 range
        
        return 0.0
    
    def _get_matched_patterns(self, message: str, patterns: List[str]) -> List[str]:
        """Get list of patterns that matched the message"""
        matched = []
        for pattern in patterns:
            if re.search(pattern, message, re.IGNORECASE):
                matched.append(pattern)
        return matched
    
    def _adjust_confidence_by_context(
        self,
        intent_result: Dict[str, Any],
        current_state: Any,
        conversation: ConversationContext
    ) -> float:
        """Adjust confidence based on conversation context"""
        base_confidence = intent_result["confidence"]
        intent = intent_result["intent"]
        
        # Context-aware adjustments
        if current_state:
            state_name = getattr(current_state, 'value', str(current_state))
            
            # Boost confidence for expected intents in specific states
            if state_name == "collecting_info":
                if intent in ["provide_address", "provide_area", "provide_house_type", "provide_time"]:
                    base_confidence = min(1.0, base_confidence * 1.3)
            
            elif state_name == "suggesting_services":
                if intent in ["select_service", "skip_services"]:
                    base_confidence = min(1.0, base_confidence * 1.2)
            
            elif state_name == "confirming_booking":
                if intent in ["confirm", "reject"]:
                    base_confidence = min(1.0, base_confidence * 1.4)
        
        # Boost confidence for consistent intent patterns
        recent_intents = self._get_recent_intents(conversation, limit=3)
        if intent in recent_intents:
            base_confidence = min(1.0, base_confidence * 1.1)
        
        return base_confidence
    
    def _get_recent_intents(self, conversation: ConversationContext, limit: int = 5) -> List[str]:
        """Get recent intents from conversation history"""
        recent_intents = []
        
        # Extract intents from recent messages (if available)
        for message in conversation.messages[-limit:]:
            if message.get("role") == "user":
                metadata = message.get("metadata", {})
                intent = metadata.get("intent")
                if intent:
                    recent_intents.append(intent)
        
        return recent_intents
    
    def get_intent_suggestions(self, current_state: Any) -> List[Dict[str, str]]:
        """Get suggested intents for current conversation state"""
        suggestions = []
        
        if current_state:
            state_name = getattr(current_state, 'value', str(current_state))
            
            if state_name == "initial":
                suggestions = [
                    {"intent": "book_service", "description": "Đặt dịch vụ dọn dẹp"},
                    {"intent": "ask_price", "description": "Hỏi về giá cả"},
                    {"intent": "ask_details", "description": "Hỏi chi tiết dịch vụ"}
                ]
            
            elif state_name == "collecting_info":
                suggestions = [
                    {"intent": "provide_address", "description": "Cung cấp địa chỉ"},
                    {"intent": "provide_area", "description": "Cung cấp diện tích"},
                    {"intent": "provide_house_type", "description": "Cho biết loại nhà"}
                ]
            
            elif state_name == "suggesting_services":
                suggestions = [
                    {"intent": "select_service", "description": "Chọn dịch vụ bổ sung"},
                    {"intent": "skip_services", "description": "Bỏ qua dịch vụ bổ sung"},
                    {"intent": "ask_details", "description": "Hỏi chi tiết dịch vụ"}
                ]
            
            elif state_name == "confirming_booking":
                suggestions = [
                    {"intent": "confirm", "description": "Xác nhận đặt lịch"},
                    {"intent": "reject", "description": "Hủy đặt lịch"},
                    {"intent": "ask_details", "description": "Hỏi thêm thông tin"}
                ]
        
        return suggestions
    
    def is_high_confidence(self, intent_result: Dict[str, Any]) -> bool:
        """Check if intent classification has high confidence"""
        return intent_result.get("confidence", 0) >= self.confidence_threshold
    
    def get_intent_explanation(self, intent: str) -> str:
        """Get human-readable explanation of intent"""
        explanations = {
            "book_service": "Khách hàng muốn đặt dịch vụ dọn dẹp",
            "provide_address": "Khách hàng cung cấp địa chỉ",
            "provide_area": "Khách hàng cung cấp diện tích nhà",
            "provide_house_type": "Khách hàng cho biết loại nhà",
            "provide_time": "Khách hàng cung cấp thời gian mong muốn",
            "select_service": "Khách hàng chọn dịch vụ bổ sung",
            "skip_services": "Khách hàng bỏ qua dịch vụ bổ sung",
            "confirm": "Khách hàng xác nhận",
            "reject": "Khách hàng từ chối",
            "ask_price": "Khách hàng hỏi về giá cả",
            "ask_time": "Khách hàng hỏi về thời gian",
            "ask_details": "Khách hàng hỏi chi tiết",
            "greeting": "Khách hàng chào hỏi",
            "help": "Khách hàng cần hỗ trợ",
            "unknown": "Không xác định được ý định"
        }
        
        return explanations.get(intent, f"Ý định: {intent}")
