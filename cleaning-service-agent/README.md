# 🏠 Cleaning Service Booking Agent

## 📚 Tổng Quan

**Cleaning Service Booking Agent** là một intelligent conversational agent được thiết kế để giúp khách hàng đặt dịch vụ dọn dẹp nhà một cách tự động và thông minh. Agent sử dụng **multi-step state management** và **natural language processing** để thu thập thông tin, đề xuất dịch vụ, và hoàn tất việc đặt lịch.

---

## 🎯 Chức Năng Chính

### **4 Bước Quy Trình Đặt Dịch vụ:**

1. **📋 Thu Thập Thông Tin**
   - Địa chỉ nhà khách hàng
   - Diện tích nhà (m²)
   - Loại nhà (chung cư, nhà riêng, villa)
   - Thời gian mong muốn

2. **💡 Đề Xuất Dịch Vụ Bổ Sung**
   - Dịch vụ nấu ăn
   - Dịch vụ ủi đồ
   - Dịch vụ chăm sóc cây cảnh
   - Dịch vụ giặt giũ

3. **💰 Tính Toán Giá Dịch Vụ**
   - Giá cơ bản theo diện tích
   - Phụ phí dịch vụ bổ sung
   - Giảm giá (nếu có)
   - Tổng chi phí cuối cùng

4. **📅 Đặt Lịch Hẹn**
   - Xác nhận thông tin
   - Chọn thời gian phù hợp
   - Tạo booking confirmation
   - Gửi thông báo xác nhận

---

## 🏗️ Kiến Trúc Hệ Thống

### **Core Components:**

```
cleaning-service-agent/
├── README.md                           # Documentation
├── requirements.txt                    # Dependencies
├── setup.py                           # Package setup
├── agents/                            # Core agent implementations
│   ├── __init__.py
│   ├── abstract/                      # Abstract base classes
│   │   ├── __init__.py
│   │   ├── base_agent.py             # Base conversational agent
│   │   └── state_manager.py          # State management system
│   ├── cleaning_agent/               # Main cleaning service agent
│   │   ├── __init__.py
│   │   ├── agent.py                  # Main agent implementation
│   │   ├── states.py                 # State definitions
│   │   ├── actions.py                # Action handlers
│   │   └── pricing.py                # Pricing logic
│   ├── nlp/                          # Natural language processing
│   │   ├── __init__.py
│   │   ├── intent_classifier.py      # Intent recognition
│   │   ├── entity_extractor.py       # Entity extraction
│   │   └── response_generator.py     # Response generation
│   └── utils/                        # Utility functions
│       ├── __init__.py
│       ├── validators.py             # Input validation
│       ├── formatters.py             # Output formatting
│       └── database.py               # Data persistence
├── config/                           # Configuration files
│   ├── __init__.py
│   ├── agent_config.py              # Agent configuration
│   ├── pricing_config.py            # Pricing rules
│   └── service_config.py            # Service definitions
├── examples/                         # Usage examples
│   ├── __init__.py
│   ├── basic_conversation.py        # Basic usage example
│   ├── advanced_booking.py          # Advanced features
│   └── integration_demo.py          # Integration examples
├── tests/                           # Test suite
│   ├── __init__.py
│   ├── test_agent.py               # Agent tests
│   ├── test_states.py              # State management tests
│   └── test_pricing.py             # Pricing logic tests
└── data/                           # Data files
    ├── services.json               # Service definitions
    ├── pricing_rules.json          # Pricing configurations
    └── sample_conversations.json   # Training data
```

---

## 🚀 Quick Start

### **Installation:**

```bash
# Clone repository
git clone <repository-url>
cd cleaning-service-agent

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

### **Basic Usage:**

```python
from agents.cleaning_agent import CleaningServiceAgent

# Initialize agent
agent = CleaningServiceAgent()

# Start conversation
response = agent.process_message("Tôi muốn đặt dịch vụ dọn dẹp nhà")
print(response)

# Continue conversation
response = agent.process_message("Nhà tôi ở 123 Nguyễn Văn A, Q1, TPHCM")
print(response)

response = agent.process_message("Diện tích khoảng 80m2")
print(response)
```

### **Advanced Usage:**

```python
from agents.cleaning_agent import CleaningServiceAgent
from config.agent_config import AgentConfig

# Custom configuration
config = AgentConfig(
    enable_additional_services=True,
    pricing_model="premium",
    language="vietnamese"
)

# Initialize with custom config
agent = CleaningServiceAgent(config=config)

# Process with context
context = {
    "customer_id": "CUST001",
    "session_id": "SESSION123",
    "preferred_language": "vi"
}

response = agent.process_message(
    message="Tôi cần dọn dẹp nhà cấp tốc",
    context=context
)
```

---

## 🎯 Features

### **✅ Intelligent Conversation Flow:**
- **State-based dialogue management**
- **Context-aware responses**
- **Multi-turn conversation handling**
- **Error recovery mechanisms**

### **✅ Smart Information Extraction:**
- **Address parsing và validation**
- **Area calculation từ description**
- **Time preference detection**
- **Service requirement analysis**

### **✅ Dynamic Pricing System:**
- **Area-based pricing tiers**
- **Service combination discounts**
- **Time-based pricing adjustments**
- **Promotional pricing rules**

### **✅ Flexible Service Management:**
- **Modular service definitions**
- **Customizable service packages**
- **Add-on service recommendations**
- **Service availability checking**

### **✅ Robust Booking System:**
- **Calendar integration**
- **Conflict detection**
- **Automatic confirmation**
- **Booking modification support**

---

## 📊 State Management

### **Conversation States:**

```python
class ConversationState(Enum):
    INITIAL = "initial"                    # Starting state
    COLLECTING_INFO = "collecting_info"    # Gathering basic info
    INFO_COMPLETE = "info_complete"        # All info collected
    SUGGESTING_SERVICES = "suggesting"     # Recommending add-ons
    CALCULATING_PRICE = "calculating"      # Computing total cost
    CONFIRMING_BOOKING = "confirming"      # Final confirmation
    BOOKING_COMPLETE = "complete"          # Successfully booked
    ERROR_RECOVERY = "error_recovery"      # Handling errors
```

### **Information Collection States:**

```python
class InfoCollectionState(Enum):
    NEED_ADDRESS = "need_address"          # Collecting address
    NEED_AREA = "need_area"               # Collecting area info
    NEED_HOUSE_TYPE = "need_house_type"   # House type classification
    NEED_TIME_PREFERENCE = "need_time"    # Time preferences
    INFO_VALIDATION = "validating"        # Validating collected info
```

---

## 💰 Pricing Model

### **Base Pricing Structure:**

```python
# Area-based pricing (VND per m²)
AREA_PRICING = {
    "small": {"range": (0, 50), "price_per_m2": 15000},      # ≤50m²
    "medium": {"range": (51, 100), "price_per_m2": 12000},   # 51-100m²
    "large": {"range": (101, 200), "price_per_m2": 10000},   # 101-200m²
    "extra_large": {"range": (201, float('inf')), "price_per_m2": 8000}  # >200m²
}

# Additional services pricing
ADDITIONAL_SERVICES = {
    "cooking": {"price": 200000, "description": "Nấu ăn (2 bữa)"},
    "ironing": {"price": 100000, "description": "Ủi đồ (10 bộ)"},
    "plant_care": {"price": 50000, "description": "Chăm sóc cây cảnh"},
    "laundry": {"price": 150000, "description": "Giặt giũ"}
}
```

---

## 🔧 Configuration

### **Agent Configuration:**

```python
# config/agent_config.py
class AgentConfig:
    def __init__(self):
        # Conversation settings
        self.max_turns = 20
        self.timeout_minutes = 30
        self.language = "vietnamese"
        
        # Feature flags
        self.enable_additional_services = True
        self.enable_dynamic_pricing = True
        self.enable_booking_confirmation = True
        
        # NLP settings
        self.intent_confidence_threshold = 0.7
        self.entity_extraction_model = "vi_core_news_sm"
        
        # Pricing settings
        self.pricing_model = "standard"
        self.discount_enabled = True
        self.promotional_pricing = False
```

---

## 🧪 Testing

### **Run Tests:**

```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/test_agent.py -v
python -m pytest tests/test_pricing.py -v

# Run with coverage
python -m pytest tests/ --cov=agents --cov-report=html
```

### **Test Coverage:**

- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end conversation flows
- **Performance Tests**: Response time và memory usage
- **Edge Case Tests**: Error handling và recovery

---

## 📈 Performance Metrics

### **Response Time Targets:**
- **Intent Classification**: < 100ms
- **Entity Extraction**: < 200ms
- **Price Calculation**: < 50ms
- **Total Response Time**: < 500ms

### **Accuracy Targets:**
- **Intent Recognition**: > 95%
- **Entity Extraction**: > 90%
- **Address Validation**: > 98%
- **Booking Success Rate**: > 99%

---

## 🔮 Future Enhancements

### **Planned Features:**
- **Multi-language support** (English, Chinese)
- **Voice interface integration**
- **AI-powered service recommendations**
- **Real-time staff availability**
- **Customer feedback integration**
- **Mobile app integration**

### **Technical Improvements:**
- **Advanced NLP models** (BERT, GPT integration)
- **Reinforcement learning** cho conversation optimization
- **Microservices architecture**
- **Real-time analytics dashboard**

---

## 📞 Support

### **Documentation:**
- **API Reference**: `/docs/api/`
- **Configuration Guide**: `/docs/configuration/`
- **Deployment Guide**: `/docs/deployment/`

### **Contact:**
- **Technical Support**: tech-support@cleaningservice.com
- **Business Inquiries**: business@cleaningservice.com
- **Bug Reports**: GitHub Issues

---

**Version**: 1.0.0  
**Last Updated**: 2024-01-15  
**License**: MIT
