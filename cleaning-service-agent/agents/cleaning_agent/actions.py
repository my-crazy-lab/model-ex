"""
Business Actions for Cleaning Service Agent

This module implements all business logic actions for the cleaning service
booking process including validation, booking creation, and scheduling.
"""

import uuid
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when validation fails"""
    pass


class BookingError(Exception):
    """Raised when booking creation fails"""
    pass


class CleaningServiceActions:
    """
    Handles all business actions for cleaning service booking
    
    Provides methods for:
    - Information validation
    - Booking creation and management
    - Schedule management
    - Customer communication
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.bookings_db = {}  # In-memory storage (replace with real DB)
        self.schedule_db = {}  # Schedule management
        
        # Initialize available time slots
        self.available_time_slots = self.config.get("available_time_slots", [
            "08:00-12:00", "13:00-17:00", "18:00-22:00"
        ])
        
        logger.debug("CleaningServiceActions initialized")
    
    def validate_address(self, address: str) -> Tuple[bool, str]:
        """
        Validate customer address
        
        Args:
            address: Customer address string
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not address or len(address.strip()) < 10:
            return False, "Địa chỉ quá ngắn. Vui lòng cung cấp địa chỉ đầy đủ."
        
        # Basic address pattern validation (Vietnamese addresses)
        address_patterns = [
            r'\d+.*[,\s]+(quận|huyện|q\.|h\.)',  # Has district
            r'\d+.*[,\s]+(tp\.|thành phố|tphcm|hà nội|đà nẵng)',  # Has city
            r'\d+.*[,\s]+.*[,\s]+.*'  # At least 3 parts separated by commas/spaces
        ]
        
        address_lower = address.lower()
        if any(re.search(pattern, address_lower) for pattern in address_patterns):
            return True, ""
        
        return False, "Địa chỉ chưa đầy đủ. Vui lòng bao gồm số nhà, đường, quận/huyện và thành phố."
    
    def validate_area(self, area_input: str) -> Tuple[bool, float, str]:
        """
        Validate and parse area input
        
        Args:
            area_input: Area input string
            
        Returns:
            Tuple of (is_valid, parsed_area, error_message)
        """
        try:
            # Extract number from input
            area_match = re.search(r'(\d+(?:\.\d+)?)', area_input.replace(',', '.'))
            if not area_match:
                return False, 0.0, "Không tìm thấy số diện tích. Vui lòng nhập số (ví dụ: 80, 120.5)."
            
            area = float(area_match.group(1))
            
            # Validate reasonable range
            if area < 10:
                return False, 0.0, "Diện tích quá nhỏ (< 10m²). Vui lòng kiểm tra lại."
            elif area > 1000:
                return False, 0.0, "Diện tích quá lớn (> 1000m²). Vui lòng kiểm tra lại hoặc liên hệ trực tiếp."
            
            return True, area, ""
            
        except ValueError:
            return False, 0.0, "Định dạng diện tích không hợp lệ. Vui lòng nhập số (ví dụ: 80, 120.5)."
    
    def validate_house_type(self, house_type_input: str) -> Tuple[bool, str, str]:
        """
        Validate and normalize house type
        
        Args:
            house_type_input: House type input string
            
        Returns:
            Tuple of (is_valid, normalized_type, error_message)
        """
        house_type_lower = house_type_input.lower().strip()
        
        # Define valid house types and their variations
        house_type_mapping = {
            "chung cư": ["chung cu", "chung cư", "apartment", "căn hộ", "can ho"],
            "nhà riêng": ["nha rieng", "nhà riêng", "nha", "nhà", "house", "nhà ở"],
            "villa": ["villa", "biệt thự", "biet thu", "villa"],
            "văn phòng": ["van phong", "văn phòng", "office", "cơ quan", "co quan"]
        }
        
        for standard_type, variations in house_type_mapping.items():
            if any(variation in house_type_lower for variation in variations):
                return True, standard_type, ""
        
        return False, "", "Loại nhà không hợp lệ. Vui lòng chọn: chung cư, nhà riêng, villa, hoặc văn phòng."
    
    def validate_time_preference(self, time_input: str) -> Tuple[bool, str, str]:
        """
        Validate and normalize time preference
        
        Args:
            time_input: Time preference input string
            
        Returns:
            Tuple of (is_valid, normalized_time, error_message)
        """
        time_lower = time_input.lower().strip()
        
        # Define time mappings
        time_mapping = {
            "sáng": ["sang", "sáng", "morning", "8", "9", "10", "11", "8-12", "8h-12h"],
            "chiều": ["chieu", "chiều", "afternoon", "13", "14", "15", "16", "13-17", "13h-17h"],
            "tối": ["toi", "tối", "evening", "18", "19", "20", "21", "18-22", "18h-22h"]
        }
        
        for standard_time, variations in time_mapping.items():
            if any(variation in time_lower for variation in variations):
                return True, standard_time, ""
        
        return False, "", "Thời gian không hợp lệ. Vui lòng chọn: sáng (8-12h), chiều (13-17h), hoặc tối (18-22h)."
    
    def create_booking(
        self,
        customer_info: Dict[str, Any],
        selected_services: List[str],
        pricing: Dict[str, Any],
        session_id: str
    ) -> Dict[str, Any]:
        """
        Create a new booking
        
        Args:
            customer_info: Collected customer information
            selected_services: List of selected additional services
            pricing: Pricing calculation result
            session_id: Conversation session ID
            
        Returns:
            Booking confirmation data
        """
        try:
            # Generate booking ID
            booking_id = f"CLN{datetime.now().strftime('%Y%m%d')}{str(uuid.uuid4())[:8].upper()}"
            
            # Calculate service date (minimum advance booking)
            advance_hours = self.config.get("booking_advance_hours", 24)
            earliest_date = datetime.now() + timedelta(hours=advance_hours)
            
            # Map time preference to specific time slot
            time_preference = customer_info.get("preferred_time", "sáng")
            time_slot = self._map_time_preference_to_slot(time_preference)
            
            # Find available date
            service_date = self._find_available_date(earliest_date, time_slot)
            
            # Create booking record
            booking_data = {
                "booking_id": booking_id,
                "session_id": session_id,
                "customer_info": customer_info.copy(),
                "selected_services": selected_services.copy(),
                "pricing": pricing.copy(),
                "service_date": service_date.isoformat(),
                "time_slot": time_slot,
                "status": "confirmed",
                "created_at": datetime.now().isoformat(),
                "estimated_duration": self._calculate_service_duration(
                    customer_info.get("area_m2", 0),
                    selected_services
                ),
                "special_instructions": customer_info.get("special_requirements", ""),
                "contact_phone": self._extract_phone_from_info(customer_info),
                "payment_status": "pending"
            }
            
            # Store booking
            self.bookings_db[booking_id] = booking_data
            
            # Update schedule
            self._update_schedule(service_date, time_slot, booking_id)
            
            # Generate confirmation details
            confirmation = {
                "booking_id": booking_id,
                "service_date": service_date.strftime("%d/%m/%Y"),
                "time_slot": time_slot,
                "total_amount": pricing.get("total_amount", 0),
                "estimated_duration": booking_data["estimated_duration"],
                "services_included": self._format_services_list(selected_services),
                "customer_address": customer_info.get("address", ""),
                "contact_info": "Hotline: 1900-CLEAN (1900-25326)",
                "cancellation_policy": "Có thể hủy miễn phí trước 12 giờ",
                "payment_methods": ["Tiền mặt", "Chuyển khoản", "Ví điện tử"]
            }
            
            logger.info(f"Booking created successfully: {booking_id}")
            return confirmation
            
        except Exception as e:
            logger.error(f"Failed to create booking: {str(e)}")
            raise BookingError(f"Không thể tạo đặt lịch: {str(e)}")
    
    def _map_time_preference_to_slot(self, time_preference: str) -> str:
        """Map time preference to specific time slot"""
        mapping = {
            "sáng": "08:00-12:00",
            "chiều": "13:00-17:00", 
            "tối": "18:00-22:00"
        }
        return mapping.get(time_preference, "08:00-12:00")
    
    def _find_available_date(self, earliest_date: datetime, time_slot: str) -> datetime:
        """Find next available date for given time slot"""
        current_date = earliest_date.replace(hour=8, minute=0, second=0, microsecond=0)
        
        # Check next 30 days for availability
        for days_ahead in range(30):
            check_date = current_date + timedelta(days=days_ahead)
            date_key = check_date.strftime("%Y-%m-%d")
            
            # Check if slot is available
            if self._is_slot_available(date_key, time_slot):
                return check_date
        
        # If no availability found, return earliest date + 1 week
        return current_date + timedelta(days=7)
    
    def _is_slot_available(self, date_key: str, time_slot: str) -> bool:
        """Check if time slot is available on given date"""
        if date_key not in self.schedule_db:
            return True
        
        date_schedule = self.schedule_db[date_key]
        return time_slot not in date_schedule or len(date_schedule[time_slot]) < 3  # Max 3 bookings per slot
    
    def _update_schedule(self, service_date: datetime, time_slot: str, booking_id: str):
        """Update schedule with new booking"""
        date_key = service_date.strftime("%Y-%m-%d")
        
        if date_key not in self.schedule_db:
            self.schedule_db[date_key] = {}
        
        if time_slot not in self.schedule_db[date_key]:
            self.schedule_db[date_key][time_slot] = []
        
        self.schedule_db[date_key][time_slot].append(booking_id)
    
    def _calculate_service_duration(self, area_m2: float, selected_services: List[str]) -> str:
        """Calculate estimated service duration"""
        # Base cleaning time: 15 minutes per 10m²
        base_duration = max(1.0, area_m2 / 10 * 0.25)  # hours
        
        # Additional service time
        from .states import AdditionalServices
        additional_duration = 0
        all_services = AdditionalServices.get_all_services()
        
        for service_id in selected_services:
            service = all_services.get(service_id, {})
            additional_duration += service.get("duration_hours", 0)
        
        total_hours = base_duration + additional_duration
        
        if total_hours < 1:
            return "30-60 phút"
        elif total_hours < 2:
            return "1-2 giờ"
        elif total_hours < 4:
            return "2-4 giờ"
        else:
            return f"{int(total_hours)}-{int(total_hours)+1} giờ"
    
    def _extract_phone_from_info(self, customer_info: Dict[str, Any]) -> str:
        """Extract phone number from customer info (if available)"""
        # This would typically extract from address or other fields
        # For now, return placeholder
        return "Sẽ xác nhận qua tin nhắn"
    
    def _format_services_list(self, selected_services: List[str]) -> List[str]:
        """Format selected services for display"""
        from .states import AdditionalServices
        all_services = AdditionalServices.get_all_services()
        
        formatted = ["Dọn dẹp nhà cơ bản"]  # Base service
        
        for service_id in selected_services:
            service = all_services.get(service_id, {})
            if service:
                formatted.append(service["name"])
        
        return formatted
    
    def get_booking_details(self, booking_id: str) -> Optional[Dict[str, Any]]:
        """Get booking details by ID"""
        return self.bookings_db.get(booking_id)
    
    def cancel_booking(self, booking_id: str, reason: str = "") -> bool:
        """Cancel a booking"""
        if booking_id in self.bookings_db:
            booking = self.bookings_db[booking_id]
            booking["status"] = "cancelled"
            booking["cancellation_reason"] = reason
            booking["cancelled_at"] = datetime.now().isoformat()
            
            # Remove from schedule
            service_date = datetime.fromisoformat(booking["service_date"])
            date_key = service_date.strftime("%Y-%m-%d")
            time_slot = booking["time_slot"]
            
            if (date_key in self.schedule_db and 
                time_slot in self.schedule_db[date_key] and
                booking_id in self.schedule_db[date_key][time_slot]):
                self.schedule_db[date_key][time_slot].remove(booking_id)
            
            logger.info(f"Booking cancelled: {booking_id}")
            return True
        
        return False
    
    def get_available_slots(self, date: datetime) -> List[str]:
        """Get available time slots for a given date"""
        date_key = date.strftime("%Y-%m-%d")
        available_slots = []
        
        for slot in self.available_time_slots:
            if self._is_slot_available(date_key, slot):
                available_slots.append(slot)
        
        return available_slots
