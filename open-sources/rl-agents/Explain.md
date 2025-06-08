# Giải thích chi tiết về Codebase RL-Agents

## Tổng quan
RL-Agents là một bộ sưu tập các thuật toán Học Tăng cường (Reinforcement Learning) được triển khai bằng Python. Dự án cung cấp các triển khai của nhiều agent RL khác nhau có thể tương tác với các môi trường tuân theo giao diện OpenAI Gym.

## Cấu trúc dự án

### 1. Các thành phần cốt lõi

#### 1.1 Agents
Đây là phần trung tâm của dự án, triển khai các thuật toán RL khác nhau:

##### 1.1.1 Planning-based Agents
- **MCTS (Monte-Carlo Tree Search)**: 
  - Được triển khai trong `rl_agents/agents/tree_search/mcts.py`
  - Hoạt động bằng cách xây dựng cây tìm kiếm, mỗi nút đại diện cho một trạng thái
  - Sử dụng phương pháp UCT (Upper Confidence Tree) để cân bằng giữa khai thác và khám phá
  - Hàm `run()` thực hiện một lần lặp MCTS, bắt đầu từ trạng thái ban đầu
  - Hàm `plan()` chạy nhiều lần lặp để xây dựng cây và trả về kế hoạch tối ưu

- **OLOP (Open Loop Optimistic Planning)**:
  - Được triển khai trong `rl_agents/agents/tree_search/olop.py`
  - Khác với MCTS, OLOP không lưu trữ trạng thái tại mỗi nút, chỉ lưu trữ chuỗi hành động
  - Sử dụng giới hạn trên Hoeffding để ước tính giá trị của các nút

##### 1.1.2 Value-based Agents
- **DQN (Deep Q-Network)**:
  - Được triển khai trong `rl_agents/agents/deep_q_network/abstract.py`
  - Sử dụng mạng neural để ước tính hàm Q (state-action value function)
  - Phương thức `act()` chọn hành động dựa trên giá trị Q và chính sách khám phá
  - Sử dụng bộ nhớ replay để lưu trữ và học từ các trải nghiệm trước đó

- **Fitted-Q**:
  - Phiên bản batch của Q-learning, học từ một tập dữ liệu cố định

##### 1.1.3 Safe RL Agents
- **BFTQ (Budgeted Fitted-Q)**:
  - Phiên bản của Fitted-Q trong môi trường có ràng buộc ngân sách
  - Tối đa hóa phần thưởng kỳ vọng trong khi giữ chi phí kỳ vọng dưới ngân sách

#### 1.2 Các tiện ích chung

##### 1.2.1 Exploration Policies
- **Epsilon-Greedy**:
  - Được triển khai trong `rl_agents/agents/common/exploration/epsilon_greedy.py`
  - Chọn hành động tốt nhất với xác suất 1-ε và hành động ngẫu nhiên với xác suất ε
  - Tham số ε giảm dần theo thời gian (exploration schedule)

- **Boltzmann**:
  - Chọn hành động dựa trên phân phối xác suất tỷ lệ với giá trị Q
  - Tham số nhiệt độ kiểm soát mức độ khám phá

##### 1.2.2 Configuration System
- **Configurable**:
  - Được triển khai trong `rl_agents/configuration.py`
  - Cung cấp cơ chế để quản lý cấu hình với các giá trị mặc định
  - Phương thức `default_config()` cung cấp cấu hình mặc định
  - Phương thức `rec_update()` cập nhật đệ quy cấu hình

### 2. Cấu trúc lớp trừu tượng

#### 2.1 AbstractAgent
- Giao diện cơ bản cho tất cả các agent
- Các phương thức chính:
  - `act(state)`: Chọn hành động dựa trên trạng thái hiện tại
  - `record(state, action, reward, next_state, done, info)`: Ghi lại trải nghiệm
  - `reset()`: Đặt lại trạng thái của agent

#### 2.2 AbstractTreeSearchAgent
- Lớp cơ sở cho các agent dựa trên tìm kiếm cây
- Quản lý cây tìm kiếm và chiến lược lập kế hoạch
- Phương thức `plan(observation)` lập kế hoạch chuỗi hành động tối ưu

#### 2.3 Node
- Đại diện cho một nút trong cây tìm kiếm
- Lưu trữ thông tin về trạng thái, giá trị và số lần thăm
- Các phương thức chính:
  - `expand()`: Mở rộng nút bằng cách tạo các nút con
  - `selection_rule()`: Quy tắc chọn hành động tốt nhất
  - `get_value()`: Trả về ước tính giá trị của nút

### 3. Cách triển khai chi tiết

#### 3.1 MCTS (Monte-Carlo Tree Search)
```python
def run(self, state, observation):
    """
        Chạy một lần lặp MCTS, bắt đầu từ trạng thái đã cho
    """
    node = self.root  # Bắt đầu từ nút gốc
    total_reward = 0
    depth = 0
    terminal = False
    
    # Giai đoạn Selection và Expansion
    while depth < self.config['horizon'] and node.children and not terminal:
        action = node.sampling_rule(temperature=self.config['temperature'])  # Chọn hành động theo UCT
        observation, reward, terminal, truncated, _ = self.step(state, action)  # Thực hiện hành động
        total_reward += self.config["gamma"] ** depth * reward  # Tích lũy phần thưởng
        node = node.get_child(action, observation)  # Di chuyển đến nút con
        depth += 1
    
    # Giai đoạn Simulation và Backpropagation được thực hiện trong các phương thức khác
```

#### 3.2 DQN (Deep Q-Network)
```python
def act(self, state, step_exploration_time=True):
    """
        Hành động dựa trên mô hình giá trị state-action và chính sách khám phá
    """
    self.previous_state = state
    if step_exploration_time:
        self.exploration_policy.step_time()  # Cập nhật lịch trình khám phá
    
    # Xử lý quan sát đa agent
    if isinstance(state, tuple):
        return tuple(self.act(agent_state, step_exploration_time=False) for agent_state in state)

    # Cài đặt single-agent
    values = self.get_state_action_values(state)  # Lấy giá trị Q từ mạng neural
    self.exploration_policy.update(values)  # Cập nhật chính sách khám phá
    return self.exploration_policy.sample()  # Chọn hành động theo chính sách khám phá
```

### 4. Cách sử dụng

#### 4.1 Cấu hình Agent
Các agent được cấu hình thông qua tệp JSON:
```json
{
  "__class__": "<class 'rl_agents.agents.deep_q_network.pytorch.DQNAgent'>",
  "model": {
    "type": "MultiLayerPerceptron",
    "layers": [512, 512]
  },
  "exploration": {
    "method": "EpsilonGreedy",
    "temperature": 1.0,
    "final_temperature": 0.1
  }
}
```

#### 4.2 Chạy thử nghiệm
```bash
# Huấn luyện agent DQN trên môi trường CartPole
python experiments.py evaluate configs/CartPoleEnv/env.json configs/CartPoleEnv/DQNAgent.json --train --episodes=200

# Chạy benchmark nhiều agent
python experiments.py benchmark cartpole_benchmark.json --test --processes=4
```

### 5. Quy trình hoạt động

1. **Khởi tạo**:
   - Tạo môi trường từ cấu hình
   - Tạo agent từ cấu hình
   - Thiết lập các công cụ giám sát

2. **Vòng lặp huấn luyện**:
   - Agent quan sát trạng thái hiện tại
   - Agent chọn hành động dựa trên chính sách
   - Môi trường thực hiện hành động và trả về trạng thái mới, phần thưởng
   - Agent ghi lại trải nghiệm và cập nhật mô hình

3. **Đánh giá**:
   - Chạy agent đã huấn luyện trong môi trường
   - Ghi lại hiệu suất (phần thưởng tích lũy, độ dài tập)
   - Hiển thị kết quả và lưu mô hình
```
