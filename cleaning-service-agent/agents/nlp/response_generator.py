"""
Response Generator for Cleaning Service Agent

This module generates natural, contextual responses for different
conversation states and user interactions.
"""

from typing import Dict, Any, List, Optional
import random
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """
    Generates contextual responses for cleaning service conversations
    
    Features:
    - State-aware response generation
    - Personalized messaging
    - Template-based responses with variations
    - Dynamic content insertion
    - Tone and style consistency
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.language = config.get("language", "vietnamese")
        self.tone = config.get("tone", "friendly")
        
        # Initialize response templates
        self._initialize_response_templates()
        
        logger.debug("ResponseGenerator initialized")
    
    def _initialize_response_templates(self):
        """Initialize response templates for different scenarios"""
        
        # Welcome messages
        self.welcome_templates = [
            "Xin chào! Tôi là trợ lý đặt dịch vụ dọn dẹp nhà. Tôi sẽ giúp bạn đặt lịch dọn dẹp một cách nhanh chóng và tiện lợi.",
            "Chào bạn! Cảm ơn bạn đã quan tâm đến dịch vụ dọn dẹp nhà của chúng tôi. Hãy để tôi hỗ trợ bạn đặt lịch nhé!",
            "Xin chào! Tôi có thể giúp bạn đặt dịch vụ dọn dẹp nhà chuyên nghiệp. Chúng ta bắt đầu nhé!"
        ]
        
        # Information collection prompts
        self.info_collection_templates = {
            "address": [
                "Để bắt đầu, bạn có thể cho tôi biết địa chỉ nhà cần dọn dẹp không?",
                "Vui lòng cung cấp địa chỉ nhà bạn (bao gồm số nhà, đường, quận/huyện và thành phố).",
                "Địa chỉ nhà bạn ở đâu ạ? Tôi cần thông tin này để sắp xếp đội ngũ phù hợp."
            ],
            "area_m2": [
                "Nhà bạn có diện tích khoảng bao nhiêu mét vuông?",
                "Để tính giá chính xác, bạn có thể cho biết diện tích nhà không? (ví dụ: 80m², 120m²)",
                "Diện tích nhà bạn là bao nhiêu m² ạ?"
            ],
            "house_type": [
                "Nhà bạn thuộc loại nào? (chung cư, nhà riêng, villa, hay văn phòng)",
                "Bạn có thể cho biết loại nhà không? Điều này giúp chúng tôi chuẩn bị dụng cụ phù hợp.",
                "Nhà bạn là chung cư, nhà riêng, villa hay văn phòng ạ?"
            ],
            "preferred_time": [
                "Bạn muốn đặt lịch dọn dẹp vào thời gian nào? (sáng 8-12h, chiều 13-17h, hay tối 18-22h)",
                "Khung giờ nào thuận tiện cho bạn? Chúng tôi có 3 ca: sáng, chiều và tối.",
                "Bạn có thể cho biết thời gian mong muốn không? (sáng, chiều hoặc tối)"
            ]
        }
        
        # Service suggestion templates
        self.service_suggestion_templates = [
            "Dựa trên thông tin nhà bạn, tôi có một số dịch vụ bổ sung có thể hữu ích:",
            "Ngoài dọn dẹp cơ bản, chúng tôi còn có các dịch vụ sau phù hợp với nhà bạn:",
            "Để tiết kiệm thời gian, bạn có muốn thêm các dịch vụ này không:"
        ]
        
        # Pricing presentation templates
        self.pricing_templates = [
            "Dựa trên thông tin bạn cung cấp, đây là bảng giá chi tiết:",
            "Tôi đã tính toán chi phí cho dịch vụ của bạn:",
            "Sau đây là tổng chi phí dịch vụ:"
        ]
        
        # Confirmation templates
        self.confirmation_templates = [
            "Tuyệt vời! Tôi đã ghi nhận đặt lịch của bạn. Đây là thông tin xác nhận:",
            "Cảm ơn bạn! Đặt lịch đã được xác nhận thành công:",
            "Hoàn tất! Dịch vụ dọn dẹp đã được đặt lịch:"
        ]
        
        # Error and help templates
        self.error_templates = [
            "Xin lỗi, tôi chưa hiểu rõ. Bạn có thể nói rõ hơn không?",
            "Có vẻ như có chút nhầm lẫn. Bạn có thể thử lại không?",
            "Tôi cần thêm thông tin để hỗ trợ bạn tốt hơn."
        ]
        
        self.help_templates = [
            "Tôi có thể giúp bạn đặt dịch vụ dọn dẹp nhà. Hãy bắt đầu bằng cách nói 'tôi muốn đặt dịch vụ dọn dẹp'.",
            "Để đặt dịch vụ, tôi cần biết: địa chỉ, diện tích nhà, loại nhà và thời gian mong muốn.",
            "Bạn có thể nói 'đặt dịch vụ dọn dẹp' để bắt đầu, hoặc hỏi về giá cả, thời gian làm việc."
        ]
    
    def generate_welcome_message(self) -> str:
        """Generate welcome message for new conversations"""
        template = random.choice(self.welcome_templates)
        
        # Add current time context if appropriate
        current_hour = datetime.now().hour
        if 6 <= current_hour < 12:
            greeting = "Chào buổi sáng! "
        elif 12 <= current_hour < 18:
            greeting = "Chào buổi chiều! "
        else:
            greeting = "Chào buổi tối! "
        
        return greeting + template
    
    def generate_info_collection_prompt(
        self, 
        missing_info_type: str, 
        collected_info: Dict[str, Any]
    ) -> str:
        """Generate prompt for collecting specific information"""
        
        if missing_info_type not in self.info_collection_templates:
            return f"Vui lòng cung cấp thông tin về {missing_info_type}."
        
        templates = self.info_collection_templates[missing_info_type]
        base_prompt = random.choice(templates)
        
        # Add context based on already collected information
        context_additions = []
        
        if missing_info_type == "area_m2" and "address" in collected_info:
            context_additions.append("Tôi đã ghi nhận địa chỉ của bạn.")
        
        if missing_info_type == "house_type" and "area_m2" in collected_info:
            area = collected_info["area_m2"]
            context_additions.append(f"Với diện tích {area}m², ")
        
        if missing_info_type == "preferred_time" and len(collected_info) >= 2:
            context_additions.append("Chúng ta sắp hoàn tất thông tin rồi! ")
        
        # Combine context with prompt
        if context_additions:
            context = "".join(context_additions)
            return context + base_prompt
        
        return base_prompt
    
    def generate_service_suggestions(
        self, 
        recommended_services: List[str], 
        house_type: str, 
        area_m2: float
    ) -> str:
        """Generate service suggestion message"""
        
        intro = random.choice(self.service_suggestion_templates)
        
        # Import service information
        from ..cleaning_agent.states import AdditionalServices
        all_services = AdditionalServices.get_all_services()
        
        # Build service list
        service_lines = []
        for i, service_id in enumerate(recommended_services, 1):
            service = all_services.get(service_id, {})
            if service:
                name = service["name"]
                price = service["base_price"]
                description = service["description"]
                service_lines.append(
                    f"{i}. {name} - {price:,} VND\n   {description}"
                )
        
        services_text = "\n".join(service_lines)
        
        # Add contextual recommendation reason
        context_reason = self._generate_recommendation_reason(house_type, area_m2)
        
        footer = "\nBạn có muốn thêm dịch vụ nào không? Có thể trả lời 'có' hoặc 'bỏ qua' để tiếp tục."
        
        return f"{intro}\n\n{services_text}\n\n{context_reason}{footer}"
    
    def generate_pricing_summary(
        self, 
        pricing_result: Dict[str, Any], 
        customer_info: Dict[str, Any]
    ) -> str:
        """Generate pricing summary message"""
        
        intro = random.choice(self.pricing_templates)
        
        # Import pricing calculator for formatting
        from ..cleaning_agent.pricing import PricingCalculator
        pricing_calc = PricingCalculator(self.config)
        
        # Generate detailed breakdown
        breakdown = pricing_calc.get_pricing_breakdown_summary(pricing_result)
        
        # Add payment and booking information
        total_amount = pricing_result["total_amount"]
        
        footer_lines = [
            f"\n💰 Tổng chi phí: {total_amount:,} VND",
            "",
            "📋 Thông tin đặt lịch:",
            f"• Địa chỉ: {customer_info.get('address', 'N/A')}",
            f"• Diện tích: {customer_info.get('area_m2', 'N/A')}m²",
            f"• Loại nhà: {customer_info.get('house_type', 'N/A')}",
            f"• Thời gian: {customer_info.get('preferred_time', 'N/A')}",
            "",
            "Bạn có đồng ý với giá trên và muốn xác nhận đặt lịch không?"
        ]
        
        footer = "\n".join(footer_lines)
        
        return f"{intro}\n\n{breakdown}{footer}"
    
    def generate_booking_confirmation(self, booking_result: Dict[str, Any]) -> str:
        """Generate booking confirmation message"""
        
        intro = random.choice(self.confirmation_templates)
        
        # Extract booking details
        booking_id = booking_result.get("booking_id", "N/A")
        service_date = booking_result.get("service_date", "N/A")
        time_slot = booking_result.get("time_slot", "N/A")
        total_amount = booking_result.get("total_amount", 0)
        estimated_duration = booking_result.get("estimated_duration", "N/A")
        services_included = booking_result.get("services_included", [])
        
        # Build confirmation details
        details_lines = [
            f"🎫 Mã đặt lịch: {booking_id}",
            f"📅 Ngày làm việc: {service_date}",
            f"⏰ Khung giờ: {time_slot}",
            f"⏱️ Thời gian dự kiến: {estimated_duration}",
            f"💰 Tổng chi phí: {total_amount:,} VND",
            "",
            "🧹 Dịch vụ bao gồm:"
        ]
        
        # Add services list
        for service in services_included:
            details_lines.append(f"  • {service}")
        
        # Add important notes
        notes_lines = [
            "",
            "📞 Liên hệ: 1900-CLEAN (1900-25326)",
            "💳 Thanh toán: Tiền mặt hoặc chuyển khoản",
            "🔄 Hủy lịch: Miễn phí trước 12 giờ",
            "",
            "Cảm ơn bạn đã tin tưởng dịch vụ của chúng tôi! 🏠✨"
        ]
        
        details = "\n".join(details_lines + notes_lines)
        
        return f"{intro}\n\n{details}"
    
    def generate_completion_message(self, booking_data: Dict[str, Any]) -> str:
        """Generate conversation completion message"""
        
        booking_id = booking_data.get("booking_id", "N/A")
        
        messages = [
            f"Hoàn tất! Đặt lịch {booking_id} đã được xác nhận thành công.",
            "",
            "🎉 Chúng tôi sẽ liên hệ xác nhận lại trước ngày làm việc.",
            "📱 Bạn sẽ nhận được tin nhắn nhắc nhở trước 2 giờ.",
            "",
            "Nếu có thay đổi, vui lòng gọi 1900-CLEAN.",
            "Cảm ơn bạn và hẹn gặp lại! 😊"
        ]
        
        return "\n".join(messages)
    
    def generate_help_message(self) -> str:
        """Generate help message"""
        return random.choice(self.help_templates)
    
    def generate_error_message(self, error_context: str = "") -> str:
        """Generate error message with optional context"""
        base_message = random.choice(self.error_templates)
        
        if error_context:
            return f"{base_message} {error_context}"
        
        return base_message
    
    def generate_clarification_request(self, unclear_entity: str) -> str:
        """Generate clarification request for unclear information"""
        
        clarification_templates = {
            "address": "Địa chỉ bạn cung cấp chưa đầy đủ. Vui lòng bao gồm số nhà, tên đường, quận/huyện và thành phố.",
            "area_m2": "Diện tích bạn nhập chưa rõ. Vui lòng nhập số (ví dụ: 80, 120.5).",
            "house_type": "Loại nhà chưa rõ. Vui lòng chọn: chung cư, nhà riêng, villa, hoặc văn phòng.",
            "preferred_time": "Thời gian chưa rõ. Vui lòng chọn: sáng (8-12h), chiều (13-17h), hoặc tối (18-22h)."
        }
        
        return clarification_templates.get(
            unclear_entity, 
            f"Thông tin về {unclear_entity} chưa rõ. Bạn có thể cung cấp lại không?"
        )
    
    def _generate_recommendation_reason(self, house_type: str, area_m2: float) -> str:
        """Generate contextual reason for service recommendations"""
        
        reasons = []
        
        if house_type == "villa" and area_m2 > 150:
            reasons.append("Với villa rộng như nhà bạn, các dịch vụ này sẽ rất hữu ích.")
        elif house_type == "chung cư":
            reasons.append("Đối với chung cư, những dịch vụ này thường được ưa chuộng.")
        elif area_m2 > 100:
            reasons.append("Với diện tích lớn, bạn có thể tiết kiệm thời gian với các dịch vụ bổ sung.")
        
        if not reasons:
            reasons.append("Những dịch vụ này được khách hàng đánh giá cao.")
        
        return random.choice(reasons)
    
    def add_personality_touch(self, message: str) -> str:
        """Add personality elements to message"""
        
        # Add appropriate emojis based on content
        if "hoàn tất" in message.lower() or "thành công" in message.lower():
            if not any(emoji in message for emoji in ["🎉", "✅", "😊"]):
                message += " 🎉"
        
        elif "cảm ơn" in message.lower():
            if not any(emoji in message for emoji in ["😊", "🙏", "❤️"]):
                message += " 😊"
        
        elif "dọn dẹp" in message.lower() or "vệ sinh" in message.lower():
            if not any(emoji in message for emoji in ["🧹", "✨", "🏠"]):
                message = "🧹 " + message
        
        return message
    
    def format_currency(self, amount: float) -> str:
        """Format currency for display"""
        return f"{int(amount):,} VND".replace(",", ".")
    
    def format_list(self, items: List[str], conjunction: str = "và") -> str:
        """Format list of items with proper conjunction"""
        if not items:
            return ""
        elif len(items) == 1:
            return items[0]
        elif len(items) == 2:
            return f"{items[0]} {conjunction} {items[1]}"
        else:
            return f"{', '.join(items[:-1])} {conjunction} {items[-1]}"
