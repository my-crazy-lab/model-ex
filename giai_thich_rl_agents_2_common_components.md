# 🔧 Giải Thích Chi Tiết: Common Components - Building Blocks của RL Agents

## 📚 Tổng Quan Common Components

**Common Components** là các building blocks được share across tất cả RL algorithms trong framework này. Chúng cung cấp **standardized interfaces** và **reusable implementations** cho các functionality cơ bản.

---

## 🏗️ Kiến Trúc Common Components

### Core Structure:
```
common/
├── abstract.py          # Base agent interfaces
├── exploration/         # Exploration strategies
├── memory.py           # Experience replay buffers
├── models.py           # Neural network architectures
├── optimizers.py       # Optimization algorithms
├── factory.py          # Component factory pattern
├── graphics.py         # Visualization utilities
├── seeding.py          # Random seed management
└── utils.py            # Utility functions
```

---

## 📖 Phân Tích Chi Tiết: Abstract Agent

### Dòng 6-12: Base Agent Class
```python
class AbstractAgent(Configurable, ABC):
    def __init__(self, config=None):
        super(AbstractAgent, self).__init__(config)
        self.writer = None      # Tensorboard writer for logging
        self.directory = None   # Run directory for saving models
```

**Giải thích:**
- **Configurable**: Inherit configuration management system
- **ABC**: Abstract Base Class - enforces interface implementation
- **writer**: TensorBoard integration cho training visualization
- **directory**: File system management cho model persistence

### Dòng 16-27: Core Interface - Record
```python
@abstractmethod
def record(self, state, action, reward, next_state, done, info):
    """
    Record a transition of the environment to update the agent
    :param state: s, the current state of the agent
    :param action: a, the action performed
    :param reward: r(s, a), the reward collected
    :param next_state: s', the new state of the agent after the action was performed
    :param done: whether the next state is terminal
    :return:
    """
    raise NotImplementedError()
```

**Giải thích:**
- **Experience Recording**: Standard interface cho storing transitions
- **SARS' Tuple**: (State, Action, Reward, State', Done) - fundamental RL data
- **info**: Additional environment information (costs, metadata, etc.)
- **Abstract Method**: Mỗi agent phải implement cách riêng

### Dòng 29-37: Core Interface - Act
```python
@abstractmethod
def act(self, state):
    """
    Pick an action
    :param state: s, the current state of the agent
    :return: a, the action to perform
    """
    raise NotImplementedError()
```

**Giải thích:**
- **Action Selection**: Core decision-making interface
- **State Input**: Current observation từ environment
- **Action Output**: Decision để execute trong environment
- **Policy Implementation**: Mỗi algorithm có policy khác nhau

### Dòng 39-46: Planning Interface
```python
def plan(self, state):
    """
    Plan an optimal trajectory from an initial state.
    :param state: s, the initial state of the agent
    :return: [a0, a1, a2...], a sequence of actions to perform
    """
    return [self.act(state)]
```

**Giải thích:**
- **Trajectory Planning**: Multi-step lookahead capability
- **Default Implementation**: Single-step planning (just act)
- **Model-Based Extension**: Advanced agents có thể override cho multi-step planning
- **Return Format**: Sequence of actions

### Dòng 48-53: Reset Interface
```python
@abstractmethod
def reset(self):
    """
    Reset the agent to its initial internal state
    """
    raise NotImplementedError()
```

**Giải thích:**
- **State Reset**: Clear internal state cho new episode
- **Memory Management**: Reset experience buffers, networks, etc.
- **Reproducibility**: Ensure consistent starting conditions
- **Episode Boundary**: Called at start of each episode

### Dòng 55-62: Seeding Interface
```python
@abstractmethod
def seed(self, seed=None):
    """
    Seed the agent's random number generator
    :param seed: the seed to be used to generate random numbers
    :return: the used seed
    """
    raise NotImplementedError()
```

**Giải thích:**
- **Reproducibility**: Control randomness cho consistent results
- **Random Number Generation**: Seed all RNG sources
- **Return Seed**: Return actual seed used (useful for logging)
- **Debugging**: Essential cho debugging và comparison

### Dòng 64-78: Model Persistence
```python
@abstractmethod
def save(self, filename):
    """
    Save the model parameters to a file
    :param str filename: the path of the file to save the model parameters in
    """
    raise NotImplementedError()

@abstractmethod
def load(self, filename):
    """
    Load the model parameters from a file
    :param str filename: the path of the file to load the model parameters from
    """
    raise NotImplementedError()
```

**Giải thích:**
- **Model Persistence**: Save/load trained models
- **Checkpoint System**: Enable training resumption
- **Model Deployment**: Load trained models cho production
- **Experiment Management**: Save best models during training

### Dòng 80-98: Optional Interfaces
```python
def eval(self):
    """
    Set to testing mode. Disable any unnecessary exploration.
    """
    pass

def set_writer(self, writer):
    """
    Set a tensorboard writer to log the agent internal variables.
    :param SummaryWriter writer: a summary writer
    """
    self.writer = writer

def set_time(self, time):
    """ Set a local time, to control the agent internal schedules (e.g. exploration) """
    pass
```

**Giải thích:**
- **eval()**: Switch to evaluation mode (disable exploration)
- **set_writer()**: TensorBoard integration cho monitoring
- **set_time()**: Time-based scheduling (exploration decay, learning rate scheduling)

### Dòng 101-112: Stochastic Agent Extension
```python
class AbstractStochasticAgent(AbstractAgent):
    """
    Agents that implement a stochastic policy
    """
    def action_distribution(self, state):
        """
        Compute the distribution of actions for a given state
        :param state: the current state
        :return: a dictionary {action:probability}
        """
        raise NotImplementedError()
```

**Giải thích:**
- **Stochastic Policies**: Agents với probabilistic action selection
- **Action Distribution**: Return probability distribution over actions
- **Policy Analysis**: Useful cho understanding agent behavior
- **Entropy Calculation**: Enable entropy-based exploration

---

## 🔍 Exploration Strategies Deep Dive

### Epsilon-Greedy Exploration
```python
# File: exploration/epsilon_greedy.py
class EpsilonGreedy:
    def __init__(self, epsilon=0.1, final_epsilon=0.01, decay_steps=10000):
        self.epsilon = epsilon
        self.final_epsilon = final_epsilon
        self.decay_steps = decay_steps
        self.step = 0
    
    def select_action(self, q_values):
        # Decay epsilon over time
        current_epsilon = max(
            self.final_epsilon,
            self.epsilon * (1 - self.step / self.decay_steps)
        )
        
        if np.random.random() < current_epsilon:
            return np.random.randint(len(q_values))  # Random action
        else:
            return np.argmax(q_values)  # Greedy action
```

### Boltzmann Exploration
```python
# File: exploration/boltzmann.py
class BoltzmannExploration:
    def __init__(self, temperature=1.0, final_temperature=0.1, decay_steps=10000):
        self.temperature = temperature
        self.final_temperature = final_temperature
        self.decay_steps = decay_steps
    
    def select_action(self, q_values):
        # Temperature decay
        current_temp = max(
            self.final_temperature,
            self.temperature * (1 - self.step / self.decay_steps)
        )
        
        # Softmax with temperature
        exp_values = np.exp(q_values / current_temp)
        probabilities = exp_values / np.sum(exp_values)
        
        return np.random.choice(len(q_values), p=probabilities)
```

---

## 💾 Memory Systems

### Experience Replay Buffer
```python
# File: memory.py
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        """Store a transition"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """Sample a batch of transitions"""
        batch = random.sample(self.buffer, batch_size)
        return zip(*batch)  # Unpack into separate arrays
    
    def __len__(self):
        return len(self.buffer)
```

### Prioritized Experience Replay
```python
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha  # Prioritization strength
        self.beta = beta    # Importance sampling correction
        self.buffer = []
        self.priorities = np.zeros(capacity)
        self.position = 0
    
    def push(self, state, action, reward, next_state, done, priority=None):
        """Store transition with priority"""
        max_priority = np.max(self.priorities) if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.priorities[self.position] = max_priority if priority is None else priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """Sample based on priorities"""
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]
        
        # Convert priorities to probabilities
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        
        # Importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        
        batch = [self.buffer[idx] for idx in indices]
        return zip(*batch), indices, weights
```

---

## 🧠 Neural Network Models

### Standard MLP
```python
# File: models.py
class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[64, 64], activation='relu'):
        super(MLP, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(self._get_activation(activation))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def _get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x):
        return self.network(x)
```

### Dueling Network Architecture
```python
class DuelingMLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[64, 64]):
        super(DuelingMLP, self).__init__()
        
        # Shared feature layers
        self.feature_layers = self._build_layers(input_size, hidden_sizes)
        
        # Value stream
        self.value_stream = nn.Linear(hidden_sizes[-1], 1)
        
        # Advantage stream
        self.advantage_stream = nn.Linear(hidden_sizes[-1], output_size)
    
    def forward(self, x):
        features = self.feature_layers(x)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine value and advantage
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values
```

---

## ⚙️ Optimizers và Training Utilities

### Adaptive Learning Rate
```python
# File: optimizers.py
class AdaptiveOptimizer:
    def __init__(self, parameters, lr=1e-3, lr_decay=0.99, min_lr=1e-6):
        self.optimizer = torch.optim.Adam(parameters, lr=lr)
        self.lr_decay = lr_decay
        self.min_lr = min_lr
        self.initial_lr = lr
    
    def step(self, loss):
        """Step with adaptive learning rate"""
        self.optimizer.step()
        
        # Decay learning rate based on loss
        current_lr = self.optimizer.param_groups[0]['lr']
        new_lr = max(self.min_lr, current_lr * self.lr_decay)
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
    
    def zero_grad(self):
        self.optimizer.zero_grad()
```

---

## 🎯 Factory Pattern cho Component Creation

### Agent Factory
```python
# File: factory.py
class AgentFactory:
    @staticmethod
    def create_agent(agent_type, env, config):
        """Factory method to create agents"""
        if agent_type == "DQN":
            from rl_agents.agents.dqn import DQNAgent
            return DQNAgent(env, config)
        elif agent_type == "BFTQ":
            from rl_agents.agents.budgeted_ftq import BFTQAgent
            return BFTQAgent(env, config)
        # ... other agent types
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

# Usage:
agent = AgentFactory.create_agent("DQN", env, config)
```

---

## 🎯 Key Benefits của Common Components

### ✅ **Advantages:**
- **Code Reusability**: Shared components across algorithms
- **Standardized Interface**: Consistent API across agents
- **Modularity**: Easy to swap components
- **Extensibility**: Easy to add new exploration strategies, memory types
- **Testing**: Standardized testing interface
- **Debugging**: Common debugging utilities

### 🔧 **Design Patterns:**
- **Abstract Factory**: Component creation
- **Strategy Pattern**: Exploration strategies
- **Template Method**: Agent lifecycle
- **Observer Pattern**: TensorBoard logging

---

## 💡 Best Practices

### **Configuration Management:**
```python
# Hierarchical configuration
config = {
    "agent": {
        "type": "DQN",
        "exploration": {
            "type": "epsilon_greedy",
            "epsilon": 0.1,
            "decay_steps": 10000
        },
        "memory": {
            "type": "replay_buffer",
            "capacity": 10000
        },
        "network": {
            "hidden_sizes": [64, 64],
            "activation": "relu"
        }
    }
}
```

### **Logging và Monitoring:**
```python
# Standardized logging
if self.writer:
    self.writer.add_scalar('Loss/Q_Loss', loss, self.step)
    self.writer.add_scalar('Exploration/Epsilon', epsilon, self.step)
    self.writer.add_histogram('Network/Weights', weights, self.step)
```

**Next Algorithm**: Chúng ta sẽ tiếp tục với **Deep Q-Network (DQN)** - một trong những thuật toán deep RL quan trọng nhất!
