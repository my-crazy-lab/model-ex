"""
Pricing Calculator for Cleaning Service

This module handles all pricing calculations including base pricing,
additional services, discounts, and promotional offers.
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class PricingCalculator:
    """
    Handles dynamic pricing calculation for cleaning services
    
    Features:
    - Area-based pricing tiers
    - House type adjustments
    - Additional service pricing
    - Discount calculations
    - Promotional pricing
    - Time-based pricing adjustments
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pricing_model = config.get("pricing_model", "standard")
        self.enable_discounts = config.get("enable_discounts", True)
        self.minimum_service_fee = config.get("minimum_service_fee", 100000)
        
        # Initialize pricing tables
        self._initialize_pricing_tables()
        
        logger.debug(f"PricingCalculator initialized with model: {self.pricing_model}")
    
    def _initialize_pricing_tables(self):
        """Initialize pricing tables and rules"""
        
        # Base pricing per m² by area tiers (VND)
        self.area_pricing_tiers = {
            "small": {
                "range": (0, 50),
                "price_per_m2": 15000,
                "description": "Nhà nhỏ (≤50m²)"
            },
            "medium": {
                "range": (51, 100), 
                "price_per_m2": 12000,
                "description": "Nhà vừa (51-100m²)"
            },
            "large": {
                "range": (101, 200),
                "price_per_m2": 10000,
                "description": "Nhà lớn (101-200m²)"
            },
            "extra_large": {
                "range": (201, float('inf')),
                "price_per_m2": 8000,
                "description": "Nhà rất lớn (>200m²)"
            }
        }
        
        # House type multipliers
        self.house_type_multipliers = {
            "chung cư": 1.0,      # Base rate
            "nhà riêng": 1.1,     # 10% higher (more complex layout)
            "villa": 1.3,         # 30% higher (luxury service)
            "văn phòng": 1.2      # 20% higher (commercial rate)
        }
        
        # Additional services pricing (from states.py)
        self.additional_services_pricing = {
            "cooking": {"price": 200000, "name": "Dịch vụ nấu ăn"},
            "ironing": {"price": 100000, "name": "Dịch vụ ủi đồ"},
            "plant_care": {"price": 50000, "name": "Chăm sóc cây cảnh"},
            "laundry": {"price": 150000, "name": "Dịch vụ giặt giũ"},
            "deep_cleaning": {"price": 300000, "name": "Dọn dẹp sâu"},
            "window_cleaning": {"price": 80000, "name": "Lau cửa sổ"}
        }
        
        # Discount rules
        self.discount_rules = {
            "first_time_customer": {
                "percentage": 10,
                "description": "Giảm 10% cho khách hàng mới",
                "min_amount": 200000
            },
            "large_area": {
                "percentage": 5,
                "description": "Giảm 5% cho nhà >150m²",
                "min_area": 150
            },
            "multiple_services": {
                "percentage": 8,
                "description": "Giảm 8% khi đặt ≥3 dịch vụ bổ sung",
                "min_services": 3
            },
            "weekend_booking": {
                "percentage": -10,  # Negative = surcharge
                "description": "Phụ phí 10% cuối tuần",
                "applies_to": ["saturday", "sunday"]
            }
        }
        
        # Promotional campaigns
        self.active_promotions = {
            "new_year_2024": {
                "percentage": 15,
                "description": "Khuyến mãi Tết 2024 - Giảm 15%",
                "start_date": "2024-01-01",
                "end_date": "2024-02-15",
                "min_amount": 300000,
                "active": False  # Set to True to activate
            }
        }
    
    def calculate_total_price(
        self,
        area_m2: float,
        house_type: str,
        additional_services: List[str] = None,
        customer_type: str = "new",
        service_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Calculate total service price
        
        Args:
            area_m2: House area in square meters
            house_type: Type of house (chung cư, nhà riêng, villa, văn phòng)
            additional_services: List of additional service IDs
            customer_type: Customer type (new, returning, vip)
            service_date: Requested service date
            
        Returns:
            Detailed pricing breakdown
        """
        if additional_services is None:
            additional_services = []
        
        if service_date is None:
            service_date = datetime.now() + timedelta(days=1)
        
        # Calculate base cleaning price
        base_price_result = self._calculate_base_price(area_m2, house_type)
        
        # Calculate additional services price
        additional_services_result = self._calculate_additional_services_price(additional_services)
        
        # Calculate subtotal
        subtotal = base_price_result["amount"] + additional_services_result["total_amount"]
        
        # Apply discounts
        discounts_result = self._calculate_discounts(
            subtotal, area_m2, additional_services, customer_type, service_date
        )
        
        # Apply promotions
        promotions_result = self._calculate_promotions(subtotal, service_date)
        
        # Calculate final total
        total_discount = discounts_result["total_discount"] + promotions_result["total_discount"]
        final_total = max(subtotal - total_discount, self.minimum_service_fee)
        
        # Calculate savings
        total_savings = subtotal - final_total
        savings_percentage = (total_savings / subtotal * 100) if subtotal > 0 else 0
        
        # Build detailed result
        pricing_result = {
            "base_service": base_price_result,
            "additional_services": additional_services_result,
            "subtotal": subtotal,
            "discounts": discounts_result,
            "promotions": promotions_result,
            "total_amount": int(final_total),
            "total_savings": int(total_savings),
            "savings_percentage": round(savings_percentage, 1),
            "currency": "VND",
            "calculation_date": datetime.now().isoformat(),
            "service_date": service_date.isoformat(),
            "pricing_model": self.pricing_model
        }
        
        logger.info(f"Price calculated: {final_total:,.0f} VND for {area_m2}m² {house_type}")
        return pricing_result
    
    def _calculate_base_price(self, area_m2: float, house_type: str) -> Dict[str, Any]:
        """Calculate base cleaning service price"""
        
        # Determine pricing tier
        pricing_tier = self._get_pricing_tier(area_m2)
        tier_info = self.area_pricing_tiers[pricing_tier]
        
        # Calculate base amount
        base_amount = area_m2 * tier_info["price_per_m2"]
        
        # Apply house type multiplier
        house_multiplier = self.house_type_multipliers.get(house_type, 1.0)
        adjusted_amount = base_amount * house_multiplier
        
        return {
            "area_m2": area_m2,
            "house_type": house_type,
            "pricing_tier": pricing_tier,
            "price_per_m2": tier_info["price_per_m2"],
            "base_amount": int(base_amount),
            "house_type_multiplier": house_multiplier,
            "amount": int(adjusted_amount),
            "description": f"Dọn dẹp {area_m2}m² {house_type} ({tier_info['description']})"
        }
    
    def _calculate_additional_services_price(self, service_ids: List[str]) -> Dict[str, Any]:
        """Calculate additional services pricing"""
        
        services_breakdown = []
        total_amount = 0
        
        for service_id in service_ids:
            service_info = self.additional_services_pricing.get(service_id)
            if service_info:
                services_breakdown.append({
                    "service_id": service_id,
                    "name": service_info["name"],
                    "price": service_info["price"]
                })
                total_amount += service_info["price"]
        
        return {
            "services": services_breakdown,
            "total_amount": total_amount,
            "count": len(services_breakdown)
        }
    
    def _calculate_discounts(
        self,
        subtotal: float,
        area_m2: float,
        additional_services: List[str],
        customer_type: str,
        service_date: datetime
    ) -> Dict[str, Any]:
        """Calculate applicable discounts"""
        
        applicable_discounts = []
        total_discount = 0
        
        if not self.enable_discounts:
            return {
                "applicable_discounts": [],
                "total_discount": 0,
                "discount_percentage": 0
            }
        
        # First time customer discount
        if customer_type == "new":
            discount = self.discount_rules["first_time_customer"]
            if subtotal >= discount["min_amount"]:
                discount_amount = subtotal * discount["percentage"] / 100
                applicable_discounts.append({
                    "type": "first_time_customer",
                    "description": discount["description"],
                    "percentage": discount["percentage"],
                    "amount": int(discount_amount)
                })
                total_discount += discount_amount
        
        # Large area discount
        if area_m2 >= self.discount_rules["large_area"]["min_area"]:
            discount = self.discount_rules["large_area"]
            discount_amount = subtotal * discount["percentage"] / 100
            applicable_discounts.append({
                "type": "large_area",
                "description": discount["description"],
                "percentage": discount["percentage"],
                "amount": int(discount_amount)
            })
            total_discount += discount_amount
        
        # Multiple services discount
        if len(additional_services) >= self.discount_rules["multiple_services"]["min_services"]:
            discount = self.discount_rules["multiple_services"]
            discount_amount = subtotal * discount["percentage"] / 100
            applicable_discounts.append({
                "type": "multiple_services",
                "description": discount["description"],
                "percentage": discount["percentage"],
                "amount": int(discount_amount)
            })
            total_discount += discount_amount
        
        # Weekend surcharge
        if service_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
            surcharge = self.discount_rules["weekend_booking"]
            surcharge_amount = subtotal * abs(surcharge["percentage"]) / 100
            applicable_discounts.append({
                "type": "weekend_surcharge",
                "description": surcharge["description"],
                "percentage": surcharge["percentage"],
                "amount": int(surcharge_amount)
            })
            total_discount -= surcharge_amount  # Subtract because it's a surcharge
        
        discount_percentage = (total_discount / subtotal * 100) if subtotal > 0 else 0
        
        return {
            "applicable_discounts": applicable_discounts,
            "total_discount": int(total_discount),
            "discount_percentage": round(discount_percentage, 1)
        }
    
    def _calculate_promotions(self, subtotal: float, service_date: datetime) -> Dict[str, Any]:
        """Calculate promotional discounts"""
        
        applicable_promotions = []
        total_discount = 0
        
        for promo_id, promo in self.active_promotions.items():
            if not promo.get("active", False):
                continue
            
            # Check date range
            start_date = datetime.fromisoformat(promo["start_date"])
            end_date = datetime.fromisoformat(promo["end_date"])
            
            if start_date <= service_date <= end_date:
                # Check minimum amount
                if subtotal >= promo.get("min_amount", 0):
                    discount_amount = subtotal * promo["percentage"] / 100
                    applicable_promotions.append({
                        "promotion_id": promo_id,
                        "description": promo["description"],
                        "percentage": promo["percentage"],
                        "amount": int(discount_amount)
                    })
                    total_discount += discount_amount
        
        return {
            "applicable_promotions": applicable_promotions,
            "total_discount": int(total_discount)
        }
    
    def _get_pricing_tier(self, area_m2: float) -> str:
        """Determine pricing tier based on area"""
        for tier, info in self.area_pricing_tiers.items():
            min_area, max_area = info["range"]
            if min_area <= area_m2 <= max_area:
                return tier
        return "extra_large"  # Default for very large areas
    
    def get_price_estimate(self, area_m2: float, house_type: str) -> Dict[str, Any]:
        """Get quick price estimate without full calculation"""
        base_result = self._calculate_base_price(area_m2, house_type)
        
        # Add typical additional services estimate
        typical_additional = 150000  # Average additional services
        
        estimated_total = base_result["amount"] + typical_additional
        
        return {
            "base_price": base_result["amount"],
            "estimated_additional": typical_additional,
            "estimated_total": estimated_total,
            "price_range": {
                "min": base_result["amount"],
                "max": estimated_total + 200000  # Upper bound
            },
            "currency": "VND"
        }
    
    def format_price(self, amount: float) -> str:
        """Format price for display"""
        return f"{int(amount):,} VND".replace(",", ".")
    
    def get_pricing_breakdown_summary(self, pricing_result: Dict[str, Any]) -> str:
        """Generate human-readable pricing summary"""
        lines = []
        
        # Base service
        base = pricing_result["base_service"]
        lines.append(f"• {base['description']}: {self.format_price(base['amount'])}")
        
        # Additional services
        additional = pricing_result["additional_services"]
        if additional["services"]:
            lines.append("• Dịch vụ bổ sung:")
            for service in additional["services"]:
                lines.append(f"  - {service['name']}: {self.format_price(service['price'])}")
        
        # Discounts
        discounts = pricing_result["discounts"]
        if discounts["applicable_discounts"]:
            lines.append("• Giảm giá:")
            for discount in discounts["applicable_discounts"]:
                sign = "-" if discount["amount"] > 0 else "+"
                lines.append(f"  {sign} {discount['description']}: {self.format_price(abs(discount['amount']))}")
        
        # Total
        lines.append(f"• Tổng cộng: {self.format_price(pricing_result['total_amount'])}")
        
        if pricing_result["total_savings"] > 0:
            lines.append(f"• Tiết kiệm: {self.format_price(pricing_result['total_savings'])} ({pricing_result['savings_percentage']}%)")
        
        return "\n".join(lines)
