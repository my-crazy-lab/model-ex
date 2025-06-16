"""
Basic Conversation Example for Cleaning Service Agent

This example demonstrates a complete conversation flow from initial
greeting to booking confirmation.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from agents.cleaning_agent.agent import CleaningServiceAgent
import uuid
import json


def print_separator():
    """Print conversation separator"""
    print("\n" + "="*60 + "\n")


def print_user_message(message):
    """Print user message with formatting"""
    print(f"👤 Khách hàng: {message}")


def print_agent_response(response):
    """Print agent response with formatting"""
    print(f"🤖 Trợ lý: {response.message}")
    
    # Print additional data if available
    if response.data:
        print(f"📊 Dữ liệu: {json.dumps(response.data, ensure_ascii=False, indent=2)}")
    
    print(f"🎯 Trạng thái: {response.state.value}")
    print(f"📈 Độ tin cậy: {response.confidence:.2f}")


def simulate_conversation():
    """Simulate a complete cleaning service booking conversation"""
    
    print("🏠 DEMO: Đặt Dịch Vụ Dọn Dẹp Nhà")
    print("Mô phỏng cuộc trò chuyện hoàn chỉnh từ chào hỏi đến xác nhận đặt lịch")
    print_separator()
    
    # Initialize agent
    agent = CleaningServiceAgent()
    session_id = str(uuid.uuid4())
    customer_id = "DEMO_CUSTOMER_001"
    
    # Conversation flow
    conversation_steps = [
        # Step 1: Initial greeting and service request
        {
            "user_message": "Xin chào, tôi muốn đặt dịch vụ dọn dẹp nhà",
            "description": "Khách hàng chào hỏi và yêu cầu đặt dịch vụ"
        },
        
        # Step 2: Provide address
        {
            "user_message": "Nhà tôi ở 123 Nguyễn Văn A, Quận 1, TP.HCM",
            "description": "Cung cấp địa chỉ nhà"
        },
        
        # Step 3: Provide area
        {
            "user_message": "Diện tích nhà khoảng 80m2",
            "description": "Cung cấp diện tích nhà"
        },
        
        # Step 4: Provide house type
        {
            "user_message": "Nhà tôi là chung cư",
            "description": "Cho biết loại nhà"
        },
        
        # Step 5: Provide time preference
        {
            "user_message": "Tôi muốn đặt lịch vào buổi sáng",
            "description": "Cung cấp thời gian mong muốn"
        },
        
        # Step 6: Select additional services
        {
            "user_message": "Tôi muốn thêm dịch vụ nấu ăn và ủi đồ",
            "description": "Chọn dịch vụ bổ sung"
        },
        
        # Step 7: Proceed to pricing
        {
            "user_message": "Tính giá cho tôi",
            "description": "Yêu cầu tính giá"
        },
        
        # Step 8: Confirm booking
        {
            "user_message": "Tôi đồng ý với giá này, xác nhận đặt lịch",
            "description": "Xác nhận đặt lịch"
        },
        
        # Step 9: Final acknowledgment
        {
            "user_message": "Cảm ơn bạn",
            "description": "Cảm ơn và kết thúc"
        }
    ]
    
    # Execute conversation
    for i, step in enumerate(conversation_steps, 1):
        print(f"📍 Bước {i}: {step['description']}")
        print_user_message(step["user_message"])
        
        # Process message through agent
        response = agent.process_message(
            message=step["user_message"],
            session_id=session_id,
            customer_id=customer_id
        )
        
        print_agent_response(response)
        print_separator()
        
        # Add small delay for readability
        import time
        time.sleep(1)
    
    # Show final conversation summary
    print("📋 TÓM TẮT CUỘC TRÒ CHUYỆN")
    conversation_history = agent.get_conversation_history(session_id)
    
    if conversation_history:
        print(f"Tổng số tin nhắn: {len(conversation_history)}")
        print(f"Thời gian bắt đầu: {conversation_history[0]['timestamp']}")
        print(f"Thời gian kết thúc: {conversation_history[-1]['timestamp']}")
        
        # Extract key information
        user_messages = [msg for msg in conversation_history if msg['role'] == 'user']
        agent_messages = [msg for msg in conversation_history if msg['role'] == 'agent']
        
        print(f"Tin nhắn từ khách hàng: {len(user_messages)}")
        print(f"Phản hồi từ trợ lý: {len(agent_messages)}")
    
    print_separator()


def interactive_demo():
    """Interactive demo where user can type messages"""
    
    print("🏠 DEMO TƯƠNG TÁC: Đặt Dịch Vụ Dọn Dẹp Nhà")
    print("Bạn có thể trò chuyện trực tiếp với trợ lý AI")
    print("Gõ 'quit' để thoát")
    print_separator()
    
    # Initialize agent
    agent = CleaningServiceAgent()
    session_id = str(uuid.uuid4())
    customer_id = "INTERACTIVE_USER"
    
    print("🤖 Trợ lý: Xin chào! Tôi có thể giúp bạn đặt dịch vụ dọn dẹp nhà. Hãy bắt đầu nhé!")
    
    while True:
        try:
            # Get user input
            user_input = input("\n👤 Bạn: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'thoát', 'bye']:
                print("🤖 Trợ lý: Cảm ơn bạn! Hẹn gặp lại! 👋")
                break
            
            if not user_input:
                continue
            
            # Process message
            response = agent.process_message(
                message=user_input,
                session_id=session_id,
                customer_id=customer_id
            )
            
            # Display response
            print(f"🤖 Trợ lý: {response.message}")
            
            # Show state information (optional)
            if response.data:
                print(f"   📊 [{response.state.value}] - Độ tin cậy: {response.confidence:.2f}")
            
        except KeyboardInterrupt:
            print("\n\n🤖 Trợ lý: Cảm ơn bạn! Hẹn gặp lại!")
            break
        except Exception as e:
            print(f"❌ Lỗi: {str(e)}")


def test_error_scenarios():
    """Test error handling and edge cases"""
    
    print("🧪 TEST: Xử Lý Lỗi và Trường Hợp Đặc Biệt")
    print_separator()
    
    agent = CleaningServiceAgent()
    session_id = str(uuid.uuid4())
    
    error_test_cases = [
        {
            "message": "asdfghjkl",
            "description": "Tin nhắn không có nghĩa"
        },
        {
            "message": "Địa chỉ: abc",
            "description": "Địa chỉ không hợp lệ"
        },
        {
            "message": "Diện tích: -50m2",
            "description": "Diện tích âm"
        },
        {
            "message": "Nhà tôi là lâu đài",
            "description": "Loại nhà không hỗ trợ"
        },
        {
            "message": "Tôi muốn đặt lịch vào năm 2025",
            "description": "Thời gian không hợp lệ"
        }
    ]
    
    for i, test_case in enumerate(error_test_cases, 1):
        print(f"🧪 Test {i}: {test_case['description']}")
        print_user_message(test_case["message"])
        
        response = agent.process_message(
            message=test_case["message"],
            session_id=session_id
        )
        
        print_agent_response(response)
        print_separator()


def performance_test():
    """Test agent performance with multiple concurrent conversations"""
    
    print("⚡ TEST: Hiệu Suất và Đa Phiên")
    print_separator()
    
    agent = CleaningServiceAgent()
    
    # Simulate multiple conversations
    sessions = []
    for i in range(5):
        session_id = f"SESSION_{i+1}"
        sessions.append(session_id)
        
        # Start each conversation
        response = agent.process_message(
            message="Tôi muốn đặt dịch vụ dọn dẹp",
            session_id=session_id,
            customer_id=f"CUSTOMER_{i+1}"
        )
        
        print(f"📱 Phiên {i+1}: {response.message[:50]}...")
    
    print(f"\n📊 Tổng số phiên hoạt động: {agent.get_active_conversations_count()}")
    
    # Test conversation isolation
    print("\n🔒 Test phân tách phiên:")
    for session_id in sessions[:2]:
        response = agent.process_message(
            message="Nhà tôi ở Hà Nội",
            session_id=session_id
        )
        print(f"   {session_id}: {response.state.value}")
    
    print_separator()


def main():
    """Main demo function"""
    
    print("🚀 CLEANING SERVICE AGENT DEMO")
    print("Chọn chế độ demo:")
    print("1. Mô phỏng cuộc trò chuyện hoàn chỉnh")
    print("2. Demo tương tác")
    print("3. Test xử lý lỗi")
    print("4. Test hiệu suất")
    print("5. Chạy tất cả")
    
    try:
        choice = input("\nNhập lựa chọn (1-5): ").strip()
        
        if choice == "1":
            simulate_conversation()
        elif choice == "2":
            interactive_demo()
        elif choice == "3":
            test_error_scenarios()
        elif choice == "4":
            performance_test()
        elif choice == "5":
            print("🔄 Chạy tất cả demo...")
            simulate_conversation()
            test_error_scenarios()
            performance_test()
            print("\n🎉 Hoàn thành tất cả demo!")
        else:
            print("❌ Lựa chọn không hợp lệ")
            
    except KeyboardInterrupt:
        print("\n\n👋 Tạm biệt!")
    except Exception as e:
        print(f"❌ Lỗi: {str(e)}")


if __name__ == "__main__":
    main()
