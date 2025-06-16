"""
State Definitions for Cleaning Service Agent

This module defines all conversation states and information collection states
used in the cleaning service booking workflow.
"""

from enum import Enum
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


class CleaningServiceState(Enum):
    """Main conversation states for cleaning service booking"""
    
    INITIAL = "initial"                        # Starting state
    COLLECTING_INFO = "collecting_info"        # Gathering required information
    INFO_COMPLETE = "info_complete"            # All required info collected
    SUGGESTING_SERVICES = "suggesting_services" # Recommending additional services
    CALCULATING_PRICE = "calculating_price"    # Computing total cost
    CONFIRMING_BOOKING = "confirming_booking"  # Final booking confirmation
    BOOKING_COMPLETE = "booking_complete"      # Successfully completed booking
    ERROR_RECOVERY = "error_recovery"          # Handling errors and recovery
    TERMINATED = "terminated"                  # Conversation ended


class InfoCollectionState(Enum):
    """Detailed states for information collection phase"""
    
    NEED_ADDRESS = "need_address"              # Collecting customer address
    VALIDATING_ADDRESS = "validating_address"  # Validating address format
    NEED_AREA = "need_area"                   # Collecting house area
    VALIDATING_AREA = "validating_area"       # Validating area input
    NEED_HOUSE_TYPE = "need_house_type"       # Collecting house type
    NEED_TIME_PREFERENCE = "need_time"        # Collecting preferred time
    VALIDATING_TIME = "validating_time"       # Validating time preference
    NEED_SPECIAL_REQUIREMENTS = "need_special" # Special cleaning requirements
    INFO_REVIEW = "info_review"               # Reviewing collected information


class ServiceSuggestionState(Enum):
    """States for additional service suggestion phase"""
    
    PRESENTING_OPTIONS = "presenting_options"  # Showing available services
    WAITING_SELECTION = "waiting_selection"    # Waiting for user selection
    CONFIRMING_SELECTION = "confirming_selection" # Confirming selected services
    CUSTOMIZING_SERVICE = "customizing_service"   # Customizing service details


class PricingState(Enum):
    """States for pricing calculation phase"""
    
    CALCULATING_BASE = "calculating_base"      # Computing base price
    ADDING_SERVICES = "adding_services"        # Adding additional service costs
    APPLYING_DISCOUNTS = "applying_discounts"  # Applying available discounts
    PRESENTING_TOTAL = "presenting_total"      # Showing final price
    WAITING_APPROVAL = "waiting_approval"      # Waiting for price approval


class BookingState(Enum):
    """States for booking confirmation phase"""
    
    SCHEDULING_TIME = "scheduling_time"        # Selecting appointment time
    CONFIRMING_DETAILS = "confirming_details"  # Confirming all booking details
    PROCESSING_PAYMENT = "processing_payment"  # Processing payment information
    GENERATING_CONFIRMATION = "generating_confirmation" # Creating booking confirmation
    SENDING_CONFIRMATION = "sending_confirmation"      # Sending confirmation to customer


@dataclass
class RequiredInformation:
    """Defines required information fields and their validation rules"""
    
    field_name: str
    display_name: str
    data_type: type
    required: bool = True
    validation_pattern: Optional[str] = None
    validation_function: Optional[callable] = None
    collection_prompt: str = ""
    validation_error_message: str = ""
    examples: List[str] = None
    
    def __post_init__(self):
        if self.examples is None:
            self.examples = []


class CleaningServiceInformation:
    """Defines all required information for cleaning service booking"""
    
    # Address information
    ADDRESS = RequiredInformation(
        field_name="address",
        display_name="Địa chỉ",
        data_type=str,
        required=True,
        validation_pattern=r".+",  # Basic non-empty validation
        collection_prompt="Vui lòng cho tôi biết địa chỉ nhà bạn cần dọn dẹp.",
        validation_error_message="Địa chỉ không hợp lệ. Vui lòng cung cấp địa chỉ đầy đủ.",
        examples=[
            "123 Nguyễn Văn A, Quận 1, TP.HCM",
            "Số 45 Trần Hưng Đạo, Hoàn Kiếm, Hà Nội",
            "78 Lê Lợi, Quận Hải Châu, Đà Nẵng"
        ]
    )
    
    # Area information
    AREA_M2 = RequiredInformation(
        field_name="area_m2",
        display_name="Diện tích",
        data_type=float,
        required=True,
        validation_pattern=r"^\d+(\.\d+)?$",
        collection_prompt="Diện tích nhà bạn là bao nhiêu mét vuông?",
        validation_error_message="Diện tích phải là số dương. Ví dụ: 80, 120.5",
        examples=["50", "80", "120", "150.5", "200"]
    )
    
    # House type information
    HOUSE_TYPE = RequiredInformation(
        field_name="house_type",
        display_name="Loại nhà",
        data_type=str,
        required=True,
        collection_prompt="Nhà bạn thuộc loại nào? (chung cư, nhà riêng, villa, văn phòng)",
        validation_error_message="Vui lòng chọn loại nhà: chung cư, nhà riêng, villa, hoặc văn phòng.",
        examples=["chung cư", "nhà riêng", "villa", "văn phòng"]
    )
    
    # Time preference information
    PREFERRED_TIME = RequiredInformation(
        field_name="preferred_time",
        display_name="Thời gian mong muốn",
        data_type=str,
        required=True,
        collection_prompt="Bạn muốn đặt lịch dọn dẹp vào thời gian nào? (sáng, chiều, tối)",
        validation_error_message="Vui lòng chọn khung giờ: sáng (8-12h), chiều (13-17h), hoặc tối (18-22h).",
        examples=["sáng", "chiều", "tối", "8-12h", "13-17h", "18-22h"]
    )
    
    # Special requirements (optional)
    SPECIAL_REQUIREMENTS = RequiredInformation(
        field_name="special_requirements",
        display_name="Yêu cầu đặc biệt",
        data_type=str,
        required=False,
        collection_prompt="Bạn có yêu cầu đặc biệt nào không? (có thể bỏ qua)",
        validation_error_message="",
        examples=[
            "Dọn dẹp kỹ phòng bếp",
            "Tránh sử dụng hóa chất mạnh",
            "Có thú cưng trong nhà",
            "Cần hoàn thành trước 15h"
        ]
    )
    
    @classmethod
    def get_all_required_fields(cls) -> List[RequiredInformation]:
        """Get all required information fields"""
        return [
            cls.ADDRESS,
            cls.AREA_M2,
            cls.HOUSE_TYPE,
            cls.PREFERRED_TIME
        ]
    
    @classmethod
    def get_optional_fields(cls) -> List[RequiredInformation]:
        """Get all optional information fields"""
        return [
            cls.SPECIAL_REQUIREMENTS
        ]
    
    @classmethod
    def get_field_by_name(cls, field_name: str) -> Optional[RequiredInformation]:
        """Get field definition by name"""
        all_fields = cls.get_all_required_fields() + cls.get_optional_fields()
        for field in all_fields:
            if field.field_name == field_name:
                return field
        return None


class AdditionalServices:
    """Defines available additional services"""
    
    COOKING = {
        "id": "cooking",
        "name": "Dịch vụ nấu ăn",
        "description": "Nấu 2 bữa ăn theo yêu cầu",
        "base_price": 200000,  # VND
        "duration_hours": 3,
        "requirements": ["kitchen_access", "ingredients_provided"],
        "popular": True
    }
    
    IRONING = {
        "id": "ironing",
        "name": "Dịch vụ ủi đồ",
        "description": "Ủi và gấp gọn quần áo (tối đa 10 bộ)",
        "base_price": 100000,  # VND
        "duration_hours": 1,
        "requirements": ["iron_available", "ironing_board"],
        "popular": True
    }
    
    PLANT_CARE = {
        "id": "plant_care",
        "name": "Chăm sóc cây cảnh",
        "description": "Tưới nước, cắt tỉa và chăm sóc cây cảnh",
        "base_price": 50000,   # VND
        "duration_hours": 0.5,
        "requirements": ["plants_available"],
        "popular": False
    }
    
    LAUNDRY = {
        "id": "laundry",
        "name": "Dịch vụ giặt giũ",
        "description": "Giặt và phơi quần áo",
        "base_price": 150000,  # VND
        "duration_hours": 2,
        "requirements": ["washing_machine", "detergent"],
        "popular": True
    }
    
    DEEP_CLEANING = {
        "id": "deep_cleaning",
        "name": "Dọn dẹp sâu",
        "description": "Vệ sinh chi tiết các góc khuất, tủ kệ",
        "base_price": 300000,  # VND
        "duration_hours": 2,
        "requirements": ["additional_time", "special_tools"],
        "popular": False
    }
    
    WINDOW_CLEANING = {
        "id": "window_cleaning",
        "name": "Lau cửa sổ",
        "description": "Lau sạch tất cả cửa sổ và cửa kính",
        "base_price": 80000,   # VND
        "duration_hours": 1,
        "requirements": ["window_access", "cleaning_tools"],
        "popular": True
    }
    
    @classmethod
    def get_all_services(cls) -> Dict[str, Dict[str, Any]]:
        """Get all available additional services"""
        return {
            "cooking": cls.COOKING,
            "ironing": cls.IRONING,
            "plant_care": cls.PLANT_CARE,
            "laundry": cls.LAUNDRY,
            "deep_cleaning": cls.DEEP_CLEANING,
            "window_cleaning": cls.WINDOW_CLEANING
        }
    
    @classmethod
    def get_popular_services(cls) -> Dict[str, Dict[str, Any]]:
        """Get popular additional services"""
        all_services = cls.get_all_services()
        return {k: v for k, v in all_services.items() if v.get("popular", False)}
    
    @classmethod
    def get_service_by_id(cls, service_id: str) -> Optional[Dict[str, Any]]:
        """Get service definition by ID"""
        return cls.get_all_services().get(service_id)
    
    @classmethod
    def calculate_total_additional_cost(cls, selected_service_ids: List[str]) -> int:
        """Calculate total cost of selected additional services"""
        total = 0
        all_services = cls.get_all_services()
        
        for service_id in selected_service_ids:
            service = all_services.get(service_id)
            if service:
                total += service["base_price"]
        
        return total
    
    @classmethod
    def get_service_recommendations(
        cls, 
        house_type: str, 
        area_m2: float,
        max_recommendations: int = 3
    ) -> List[str]:
        """Get service recommendations based on house characteristics"""
        recommendations = []
        
        # Base recommendations for all house types
        if house_type in ["chung cư", "nhà riêng", "villa"]:
            recommendations.extend(["ironing", "cooking"])
        
        # Area-based recommendations
        if area_m2 > 100:
            recommendations.append("deep_cleaning")
            recommendations.append("laundry")
        
        if area_m2 > 150:
            recommendations.append("window_cleaning")
        
        # House type specific recommendations
        if house_type == "villa":
            recommendations.extend(["plant_care", "window_cleaning"])
        elif house_type == "văn phòng":
            recommendations.extend(["window_cleaning", "deep_cleaning"])
        
        # Remove duplicates and limit to max recommendations
        unique_recommendations = list(dict.fromkeys(recommendations))
        return unique_recommendations[:max_recommendations]


class StateTransitionRules:
    """Defines rules for state transitions"""
    
    @staticmethod
    def can_proceed_to_suggestions(collected_info: Dict[str, Any]) -> bool:
        """Check if can proceed to service suggestions"""
        required_fields = ["address", "area_m2", "house_type", "preferred_time"]
        return all(field in collected_info for field in required_fields)
    
    @staticmethod
    def can_proceed_to_pricing(
        collected_info: Dict[str, Any], 
        service_selection_complete: bool = True
    ) -> bool:
        """Check if can proceed to pricing calculation"""
        return (
            StateTransitionRules.can_proceed_to_suggestions(collected_info) and
            service_selection_complete
        )
    
    @staticmethod
    def can_proceed_to_booking(
        collected_info: Dict[str, Any],
        pricing_approved: bool = False
    ) -> bool:
        """Check if can proceed to booking confirmation"""
        return (
            StateTransitionRules.can_proceed_to_pricing(collected_info) and
            pricing_approved
        )
    
    @staticmethod
    def is_booking_complete(
        collected_info: Dict[str, Any],
        booking_confirmed: bool = False
    ) -> bool:
        """Check if booking process is complete"""
        return (
            StateTransitionRules.can_proceed_to_booking(collected_info) and
            booking_confirmed
        )
