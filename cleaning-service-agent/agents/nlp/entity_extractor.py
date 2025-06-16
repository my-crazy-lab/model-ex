"""
Entity Extraction for Cleaning Service Agent

This module extracts structured information from user messages including
addresses, areas, house types, time preferences, and service selections.
"""

import re
from typing import Dict, Any, List, Optional, Tuple
import logging
from ..abstract.base_agent import ConversationContext

logger = logging.getLogger(__name__)


class EntityExtractor:
    """
    Rule-based entity extractor for cleaning service information
    
    Extracts entities such as:
    - Address information
    - House area (m²)
    - House type
    - Time preferences
    - Service selections
    - Contact information
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.confidence_threshold = config.get("entity_confidence_threshold", 0.6)
        self.language = config.get("language", "vietnamese")
        
        # Initialize extraction patterns
        self._initialize_extraction_patterns()
        
        logger.debug("EntityExtractor initialized")
    
    def _initialize_extraction_patterns(self):
        """Initialize entity extraction patterns"""
        
        # Address extraction patterns
        self.address_patterns = [
            # Full address patterns
            r'(\d+[a-zA-Z]?\s+[^,]+,\s*[^,]+,\s*[^,]+)',  # 123 Street, District, City
            r'(số\s+\d+[^,]+,\s*[^,]+,\s*[^,]+)',         # Số 123 ..., ..., ...
            r'(\d+\s+[^,]+\s+(?:quận|huyện|q\.|h\.)[^,]+,\s*[^,]+)',  # With district
            
            # Partial address patterns
            r'((?:quận|huyện|q\.|h\.)\s*\d+[^,]*)',       # District only
            r'(tp\.|thành phố|tphcm|hà nội|đà nẵng)[^,]*', # City only
            r'(\d+\s+(?:đường|phố|street)[^,]+)',         # Street only
        ]
        
        # Area extraction patterns
        self.area_patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:m2|m²|mét vuông|square meter)',
            r'diện tích\s*(?:là|khoảng)?\s*(\d+(?:\.\d+)?)',
            r'khoảng\s*(\d+(?:\.\d+)?)\s*m',
            r'(\d+(?:\.\d+)?)\s*m\s*vuông',
            r'(\d+(?:\.\d+)?)\s*mét'
        ]
        
        # House type patterns
        self.house_type_patterns = {
            "chung cư": [
                r'chung\s*cư',
                r'căn\s*hộ',
                r'apartment',
                r'condo'
            ],
            "nhà riêng": [
                r'nhà\s*riêng',
                r'nhà\s*ở',
                r'house',
                r'nhà\s*phố'
            ],
            "villa": [
                r'villa',
                r'biệt\s*thự',
                r'mansion'
            ],
            "văn phòng": [
                r'văn\s*phòng',
                r'office',
                r'cơ\s*quan'
            ]
        }
        
        # Time preference patterns
        self.time_patterns = {
            "sáng": [
                r'sáng',
                r'morning',
                r'8\s*(?:-|đến|to)\s*12',
                r'8h\s*(?:-|đến|to)\s*12h',
                r'buổi\s*sáng'
            ],
            "chiều": [
                r'chiều',
                r'afternoon',
                r'13\s*(?:-|đến|to)\s*17',
                r'13h\s*(?:-|đến|to)\s*17h',
                r'buổi\s*chiều'
            ],
            "tối": [
                r'tối',
                r'evening',
                r'18\s*(?:-|đến|to)\s*22',
                r'18h\s*(?:-|đến|to)\s*22h',
                r'buổi\s*tối'
            ]
        }
        
        # Service selection patterns
        self.service_patterns = {
            "cooking": [
                r'nấu\s*ăn',
                r'cooking',
                r'làm\s*cơm',
                r'nấu\s*cơm'
            ],
            "ironing": [
                r'ủi\s*đồ',
                r'ironing',
                r'ủi\s*quần\s*áo',
                r'là\s*đồ'
            ],
            "plant_care": [
                r'chăm\s*sóc\s*cây',
                r'plant\s*care',
                r'tưới\s*cây',
                r'cây\s*cảnh'
            ],
            "laundry": [
                r'giặt\s*giũ',
                r'laundry',
                r'giặt\s*đồ',
                r'giặt\s*quần\s*áo'
            ],
            "deep_cleaning": [
                r'dọn\s*dẹp\s*sâu',
                r'deep\s*cleaning',
                r'vệ\s*sinh\s*kỹ',
                r'dọn\s*kỹ'
            ],
            "window_cleaning": [
                r'lau\s*cửa\s*sổ',
                r'window\s*cleaning',
                r'lau\s*kính',
                r'cửa\s*kính'
            ]
        }
        
        # Phone number patterns
        self.phone_patterns = [
            r'(\+84|84|0)(\d{9,10})',
            r'(\d{3,4}[-.\s]?\d{3,4}[-.\s]?\d{3,4})',
            r'(0\d{2,3}[-.\s]?\d{3,4}[-.\s]?\d{3,4})'
        ]
        
        # Email patterns
        self.email_patterns = [
            r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
        ]
    
    def extract(self, message: str, conversation: ConversationContext) -> Dict[str, Any]:
        """
        Extract entities from user message
        
        Args:
            message: User input message
            conversation: Current conversation context
            
        Returns:
            Dictionary of extracted entities with confidence scores
        """
        entities = {}
        
        # Extract different types of entities
        address_entity = self._extract_address(message)
        if address_entity:
            entities["address"] = address_entity
        
        area_entity = self._extract_area(message)
        if area_entity:
            entities["area_m2"] = area_entity
        
        house_type_entity = self._extract_house_type(message)
        if house_type_entity:
            entities["house_type"] = house_type_entity
        
        time_entity = self._extract_time_preference(message)
        if time_entity:
            entities["preferred_time"] = time_entity
        
        service_entities = self._extract_services(message)
        if service_entities:
            entities["selected_services"] = service_entities
        
        contact_entities = self._extract_contact_info(message)
        entities.update(contact_entities)
        
        # Extract special requirements
        special_req = self._extract_special_requirements(message)
        if special_req:
            entities["special_requirements"] = special_req
        
        logger.debug(f"Extracted entities: {list(entities.keys())}")
        return entities
    
    def _extract_address(self, message: str) -> Optional[Dict[str, Any]]:
        """Extract address information"""
        message_clean = message.strip()
        
        for pattern in self.address_patterns:
            match = re.search(pattern, message_clean, re.IGNORECASE)
            if match:
                address = match.group(1).strip()
                
                # Validate address quality
                confidence = self._calculate_address_confidence(address)
                
                if confidence >= 0.3:  # Lower threshold for addresses
                    return {
                        "value": address,
                        "confidence": confidence,
                        "raw_match": match.group(0)
                    }
        
        # Fallback: if message looks like an address
        if self._looks_like_address(message_clean):
            return {
                "value": message_clean,
                "confidence": 0.5,
                "raw_match": message_clean
            }
        
        return None
    
    def _extract_area(self, message: str) -> Optional[Dict[str, Any]]:
        """Extract area information"""
        for pattern in self.area_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                try:
                    area_value = float(match.group(1).replace(',', '.'))
                    
                    # Validate reasonable area range
                    if 10 <= area_value <= 1000:
                        confidence = 0.9 if 20 <= area_value <= 500 else 0.7
                        
                        return {
                            "value": area_value,
                            "confidence": confidence,
                            "raw_match": match.group(0),
                            "unit": "m²"
                        }
                except ValueError:
                    continue
        
        return None
    
    def _extract_house_type(self, message: str) -> Optional[Dict[str, Any]]:
        """Extract house type information"""
        message_lower = message.lower()
        
        for house_type, patterns in self.house_type_patterns.items():
            for pattern in patterns:
                if re.search(pattern, message_lower):
                    confidence = 0.9 if len(patterns) == 1 else 0.8
                    
                    return {
                        "value": house_type,
                        "confidence": confidence,
                        "raw_match": pattern
                    }
        
        return None
    
    def _extract_time_preference(self, message: str) -> Optional[Dict[str, Any]]:
        """Extract time preference information"""
        message_lower = message.lower()
        
        for time_period, patterns in self.time_patterns.items():
            for pattern in patterns:
                if re.search(pattern, message_lower):
                    confidence = 0.8
                    
                    return {
                        "value": time_period,
                        "confidence": confidence,
                        "raw_match": pattern
                    }
        
        return None
    
    def _extract_services(self, message: str) -> Optional[Dict[str, Any]]:
        """Extract selected services"""
        message_lower = message.lower()
        selected_services = []
        
        for service_id, patterns in self.service_patterns.items():
            for pattern in patterns:
                if re.search(pattern, message_lower):
                    selected_services.append({
                        "service_id": service_id,
                        "confidence": 0.8,
                        "raw_match": pattern
                    })
                    break  # Only match once per service
        
        if selected_services:
            return {
                "value": selected_services,
                "confidence": 0.8,
                "count": len(selected_services)
            }
        
        return None
    
    def _extract_contact_info(self, message: str) -> Dict[str, Any]:
        """Extract contact information (phone, email)"""
        entities = {}
        
        # Extract phone numbers
        for pattern in self.phone_patterns:
            match = re.search(pattern, message)
            if match:
                phone = match.group(0)
                entities["phone"] = {
                    "value": phone,
                    "confidence": 0.9,
                    "raw_match": phone
                }
                break
        
        # Extract email addresses
        for pattern in self.email_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                email = match.group(1)
                entities["email"] = {
                    "value": email,
                    "confidence": 0.9,
                    "raw_match": email
                }
                break
        
        return entities
    
    def _extract_special_requirements(self, message: str) -> Optional[Dict[str, Any]]:
        """Extract special requirements or notes"""
        # Keywords that indicate special requirements
        requirement_indicators = [
            r'yêu cầu.*đặc biệt',
            r'lưu ý',
            r'chú ý',
            r'cần.*tránh',
            r'không.*được',
            r'phải.*làm',
            r'đặc biệt.*cần'
        ]
        
        message_lower = message.lower()
        
        for indicator in requirement_indicators:
            if re.search(indicator, message_lower):
                # Extract the requirement text (simplified)
                return {
                    "value": message.strip(),
                    "confidence": 0.6,
                    "raw_match": message
                }
        
        # Check for specific requirement patterns
        specific_requirements = [
            r'có.*thú.*cưng',
            r'không.*hóa.*chất',
            r'trước.*\d+.*giờ',
            r'sau.*\d+.*giờ',
            r'phòng.*bếp.*kỹ',
            r'không.*làm.*ồn'
        ]
        
        for req_pattern in specific_requirements:
            if re.search(req_pattern, message_lower):
                return {
                    "value": message.strip(),
                    "confidence": 0.7,
                    "raw_match": req_pattern
                }
        
        return None
    
    def _calculate_address_confidence(self, address: str) -> float:
        """Calculate confidence score for address"""
        confidence = 0.0
        
        # Check for address components
        if re.search(r'\d+', address):  # Has number
            confidence += 0.3
        
        if re.search(r'(?:đường|phố|street)', address, re.IGNORECASE):  # Has street
            confidence += 0.2
        
        if re.search(r'(?:quận|huyện|q\.|h\.)', address, re.IGNORECASE):  # Has district
            confidence += 0.2
        
        if re.search(r'(?:tp\.|thành phố|tphcm|hà nội|đà nẵng)', address, re.IGNORECASE):  # Has city
            confidence += 0.3
        
        return min(1.0, confidence)
    
    def _looks_like_address(self, text: str) -> bool:
        """Check if text looks like an address"""
        # Simple heuristics
        has_number = bool(re.search(r'\d+', text))
        has_comma = ',' in text
        has_location_words = bool(re.search(
            r'(?:đường|phố|quận|huyện|tp|thành phố|street|district|city)',
            text, re.IGNORECASE
        ))
        
        return has_number and (has_comma or has_location_words)
    
    def validate_extracted_entity(self, entity_type: str, entity_data: Dict[str, Any]) -> bool:
        """Validate extracted entity"""
        if entity_type == "area_m2":
            area = entity_data.get("value", 0)
            return 10 <= area <= 1000
        
        elif entity_type == "address":
            address = entity_data.get("value", "")
            return len(address) >= 10
        
        elif entity_type == "house_type":
            house_type = entity_data.get("value", "")
            return house_type in ["chung cư", "nhà riêng", "villa", "văn phòng"]
        
        elif entity_type == "preferred_time":
            time_pref = entity_data.get("value", "")
            return time_pref in ["sáng", "chiều", "tối"]
        
        return True  # Default to valid
    
    def get_entity_suggestions(self, entity_type: str, partial_input: str = "") -> List[str]:
        """Get suggestions for entity completion"""
        suggestions = []
        
        if entity_type == "house_type":
            suggestions = ["chung cư", "nhà riêng", "villa", "văn phòng"]
        
        elif entity_type == "preferred_time":
            suggestions = ["sáng (8-12h)", "chiều (13-17h)", "tối (18-22h)"]
        
        elif entity_type == "area_m2":
            suggestions = ["50m²", "80m²", "100m²", "120m²", "150m²"]
        
        elif entity_type == "services":
            suggestions = [
                "nấu ăn", "ủi đồ", "chăm sóc cây cảnh", 
                "giặt giũ", "dọn dẹp sâu", "lau cửa sổ"
            ]
        
        # Filter suggestions based on partial input
        if partial_input:
            partial_lower = partial_input.lower()
            suggestions = [s for s in suggestions if partial_lower in s.lower()]
        
        return suggestions
