"""
Agent Configuration for Cleaning Service Agent

This module provides comprehensive configuration management for the
cleaning service booking agent including all customizable parameters.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import os


@dataclass
class AgentConfig:
    """
    Comprehensive configuration for Cleaning Service Agent
    
    Contains all configurable parameters for agent behavior,
    NLP processing, pricing, and business logic.
    """
    
    # === BASIC AGENT SETTINGS ===
    service_type: str = "cleaning"
    agent_name: str = "Cleaning Service Assistant"
    version: str = "1.0.0"
    language: str = "vietnamese"
    
    # === CONVERSATION MANAGEMENT ===
    max_conversation_turns: int = 20
    conversation_timeout_minutes: int = 30
    enable_conversation_logging: bool = True
    enable_analytics: bool = False
    
    # === FEATURE FLAGS ===
    enable_additional_services: bool = True
    enable_dynamic_pricing: bool = True
    enable_scheduling: bool = True
    enable_discounts: bool = True
    enable_promotions: bool = False
    enable_multi_language: bool = False
    
    # === REQUIRED INFORMATION ===
    required_info_fields: List[str] = field(default_factory=lambda: [
        "address",
        "area_m2", 
        "house_type",
        "preferred_time"
    ])
    
    optional_info_fields: List[str] = field(default_factory=lambda: [
        "special_requirements",
        "contact_phone",
        "contact_email"
    ])
    
    # === NLP SETTINGS ===
    intent_confidence_threshold: float = 0.7
    entity_confidence_threshold: float = 0.6
    nlp_model: str = "rule_based"  # Options: rule_based, ml_based, hybrid
    enable_spell_correction: bool = False
    enable_context_awareness: bool = True
    
    # === SERVICE RECOMMENDATIONS ===
    max_additional_services: int = 4
    recommendation_threshold: float = 0.6
    enable_smart_recommendations: bool = True
    recommendation_strategy: str = "popularity_based"  # Options: popularity_based, profile_based, hybrid
    
    # === PRICING SETTINGS ===
    pricing_model: str = "standard"  # Options: standard, premium, economy
    minimum_service_fee: int = 100000  # VND
    currency: str = "VND"
    enable_real_time_pricing: bool = False
    pricing_precision: int = 1000  # Round to nearest 1000 VND
    
    # === SCHEDULING SETTINGS ===
    booking_advance_hours: int = 24
    max_booking_days_ahead: int = 30
    available_time_slots: List[str] = field(default_factory=lambda: [
        "08:00-12:00",
        "13:00-17:00", 
        "18:00-22:00"
    ])
    enable_weekend_booking: bool = True
    weekend_surcharge_percentage: float = 10.0
    
    # === BUSINESS RULES ===
    max_service_area: float = 1000.0  # m²
    min_service_area: float = 10.0    # m²
    service_coverage_areas: List[str] = field(default_factory=lambda: [
        "TP.HCM",
        "Hà Nội", 
        "Đà Nẵng"
    ])
    
    # === RESPONSE GENERATION ===
    response_tone: str = "friendly"  # Options: friendly, professional, casual
    enable_emojis: bool = True
    enable_personalization: bool = True
    response_length: str = "medium"  # Options: short, medium, long
    
    # === ERROR HANDLING ===
    max_retry_attempts: int = 3
    enable_graceful_degradation: bool = True
    fallback_to_human: bool = False
    error_escalation_threshold: int = 3
    
    # === PERFORMANCE SETTINGS ===
    response_timeout_seconds: float = 5.0
    max_concurrent_conversations: int = 100
    enable_caching: bool = True
    cache_ttl_minutes: int = 60
    
    # === INTEGRATION SETTINGS ===
    enable_external_apis: bool = False
    payment_gateway: str = "none"  # Options: none, stripe, paypal, vnpay
    calendar_integration: str = "none"  # Options: none, google, outlook
    crm_integration: str = "none"  # Options: none, salesforce, hubspot
    
    # === SECURITY SETTINGS ===
    enable_data_encryption: bool = True
    session_security_level: str = "standard"  # Options: basic, standard, high
    enable_audit_logging: bool = True
    data_retention_days: int = 90
    
    # === MONITORING AND ANALYTICS ===
    enable_performance_monitoring: bool = True
    enable_conversation_analytics: bool = False
    enable_business_metrics: bool = True
    metrics_collection_interval: int = 300  # seconds
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters"""
        
        # Validate thresholds
        if not 0.0 <= self.intent_confidence_threshold <= 1.0:
            raise ValueError("intent_confidence_threshold must be between 0.0 and 1.0")
        
        if not 0.0 <= self.entity_confidence_threshold <= 1.0:
            raise ValueError("entity_confidence_threshold must be between 0.0 and 1.0")
        
        # Validate time settings
        if self.conversation_timeout_minutes <= 0:
            raise ValueError("conversation_timeout_minutes must be positive")
        
        if self.booking_advance_hours < 0:
            raise ValueError("booking_advance_hours cannot be negative")
        
        # Validate area limits
        if self.min_service_area >= self.max_service_area:
            raise ValueError("min_service_area must be less than max_service_area")
        
        # Validate required fields
        if not self.required_info_fields:
            raise ValueError("required_info_fields cannot be empty")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AgentConfig':
        """Create configuration from dictionary"""
        return cls(**config_dict)
    
    @classmethod
    def from_env(cls) -> 'AgentConfig':
        """Create configuration from environment variables"""
        config = cls()
        
        # Override with environment variables if present
        env_mappings = {
            'CLEANING_AGENT_LANGUAGE': 'language',
            'CLEANING_AGENT_TIMEOUT': 'conversation_timeout_minutes',
            'CLEANING_AGENT_PRICING_MODEL': 'pricing_model',
            'CLEANING_AGENT_MIN_FEE': 'minimum_service_fee',
            'CLEANING_AGENT_ADVANCE_HOURS': 'booking_advance_hours',
            'CLEANING_AGENT_ENABLE_DISCOUNTS': 'enable_discounts',
            'CLEANING_AGENT_ENABLE_ANALYTICS': 'enable_analytics'
        }
        
        for env_var, config_attr in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Convert to appropriate type
                current_value = getattr(config, config_attr)
                if isinstance(current_value, bool):
                    setattr(config, config_attr, env_value.lower() in ['true', '1', 'yes'])
                elif isinstance(current_value, int):
                    setattr(config, config_attr, int(env_value))
                elif isinstance(current_value, float):
                    setattr(config, config_attr, float(env_value))
                else:
                    setattr(config, config_attr, env_value)
        
        return config
    
    def get_nlp_config(self) -> Dict[str, Any]:
        """Get NLP-specific configuration"""
        return {
            'language': self.language,
            'intent_confidence_threshold': self.intent_confidence_threshold,
            'entity_confidence_threshold': self.entity_confidence_threshold,
            'nlp_model': self.nlp_model,
            'enable_spell_correction': self.enable_spell_correction,
            'enable_context_awareness': self.enable_context_awareness
        }
    
    def get_pricing_config(self) -> Dict[str, Any]:
        """Get pricing-specific configuration"""
        return {
            'pricing_model': self.pricing_model,
            'minimum_service_fee': self.minimum_service_fee,
            'currency': self.currency,
            'enable_discounts': self.enable_discounts,
            'enable_promotions': self.enable_promotions,
            'enable_real_time_pricing': self.enable_real_time_pricing,
            'pricing_precision': self.pricing_precision,
            'weekend_surcharge_percentage': self.weekend_surcharge_percentage
        }
    
    def get_scheduling_config(self) -> Dict[str, Any]:
        """Get scheduling-specific configuration"""
        return {
            'booking_advance_hours': self.booking_advance_hours,
            'max_booking_days_ahead': self.max_booking_days_ahead,
            'available_time_slots': self.available_time_slots,
            'enable_weekend_booking': self.enable_weekend_booking,
            'weekend_surcharge_percentage': self.weekend_surcharge_percentage
        }
    
    def get_business_rules_config(self) -> Dict[str, Any]:
        """Get business rules configuration"""
        return {
            'max_service_area': self.max_service_area,
            'min_service_area': self.min_service_area,
            'service_coverage_areas': self.service_coverage_areas,
            'required_info_fields': self.required_info_fields,
            'optional_info_fields': self.optional_info_fields
        }
    
    def is_feature_enabled(self, feature_name: str) -> bool:
        """Check if a feature is enabled"""
        feature_flags = {
            'additional_services': self.enable_additional_services,
            'dynamic_pricing': self.enable_dynamic_pricing,
            'scheduling': self.enable_scheduling,
            'discounts': self.enable_discounts,
            'promotions': self.enable_promotions,
            'multi_language': self.enable_multi_language,
            'analytics': self.enable_analytics,
            'external_apis': self.enable_external_apis,
            'caching': self.enable_caching
        }
        
        return feature_flags.get(feature_name, False)
    
    def update_feature_flag(self, feature_name: str, enabled: bool):
        """Update a feature flag"""
        feature_mappings = {
            'additional_services': 'enable_additional_services',
            'dynamic_pricing': 'enable_dynamic_pricing',
            'scheduling': 'enable_scheduling',
            'discounts': 'enable_discounts',
            'promotions': 'enable_promotions',
            'multi_language': 'enable_multi_language',
            'analytics': 'enable_analytics',
            'external_apis': 'enable_external_apis',
            'caching': 'enable_caching'
        }
        
        attr_name = feature_mappings.get(feature_name)
        if attr_name:
            setattr(self, attr_name, enabled)
        else:
            raise ValueError(f"Unknown feature: {feature_name}")


# Predefined configuration profiles
class ConfigProfiles:
    """Predefined configuration profiles for different use cases"""
    
    @staticmethod
    def development() -> AgentConfig:
        """Development configuration with debugging enabled"""
        config = AgentConfig()
        config.enable_conversation_logging = True
        config.enable_analytics = True
        config.conversation_timeout_minutes = 60  # Longer timeout for testing
        config.enable_graceful_degradation = True
        config.response_timeout_seconds = 10.0  # Longer timeout for debugging
        return config
    
    @staticmethod
    def production() -> AgentConfig:
        """Production configuration optimized for performance"""
        config = AgentConfig()
        config.enable_conversation_logging = False
        config.enable_analytics = True
        config.conversation_timeout_minutes = 30
        config.enable_caching = True
        config.enable_performance_monitoring = True
        config.response_timeout_seconds = 3.0  # Faster response for production
        return config
    
    @staticmethod
    def minimal() -> AgentConfig:
        """Minimal configuration with basic features only"""
        config = AgentConfig()
        config.enable_additional_services = False
        config.enable_dynamic_pricing = False
        config.enable_discounts = False
        config.enable_promotions = False
        config.enable_analytics = False
        config.enable_emojis = False
        config.response_length = "short"
        return config
    
    @staticmethod
    def premium() -> AgentConfig:
        """Premium configuration with all features enabled"""
        config = AgentConfig()
        config.enable_additional_services = True
        config.enable_dynamic_pricing = True
        config.enable_discounts = True
        config.enable_promotions = True
        config.enable_analytics = True
        config.enable_smart_recommendations = True
        config.enable_personalization = True
        config.pricing_model = "premium"
        config.max_additional_services = 6
        return config


# Default configuration instance
DEFAULT_CONFIG = AgentConfig()

# Environment-based configuration
ENV_CONFIG = AgentConfig.from_env()
