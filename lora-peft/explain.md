# 📖 LoRA/PEFT Codebase Reading Guide

This guide provides a structured workflow to understand the codebase architecture, design decisions, and implementation logic.

## 🎯 Reading Strategy

### Phase 1: Foundation Understanding (30 minutes)
### Phase 2: Core Components Deep Dive (60 minutes)  
### Phase 3: Integration & Workflows (45 minutes)
### Phase 4: Advanced Features & Examples (45 minutes)

---

## 📋 Phase 1: Foundation Understanding

### 1.1 Start Here: Project Overview
**File**: `README.md`
- **Purpose**: Understand project scope, features, and capabilities
- **Key Concepts**: PEFT methods, supported models, task types
- **Reading Time**: 10 minutes

### 1.2 Dependencies & Setup
**File**: `requirements.txt` + `setup.py`
- **Purpose**: Understand external dependencies and package structure
- **Key Libraries**: transformers, peft, datasets, accelerate, bitsandbytes
- **Reading Time**: 5 minutes

### 1.3 Configuration Architecture
**Files**: `config/__init__.py` → `config/model_config.py` → `config/training_config.py`

**Reading Order & Focus**:
```python
# 1. config/__init__.py - See what's exported
# 2. config/model_config.py - Understand model configuration design
#    Focus on: ModelConfig class, PEFTConfig class, predefined configs
# 3. config/training_config.py - Understand training parameters
#    Focus on: TrainingConfig class, to_training_arguments() method
```

**Key Design Decisions**:
- **Dataclasses**: Clean, type-safe configuration
- **Predefined Configs**: Common use cases pre-configured
- **Validation**: `__post_init__` methods for config validation
- **Flexibility**: Easy to extend and customize

**Reading Time**: 15 minutes

---

## 🔧 Phase 2: Core Components Deep Dive

### 2.1 Data Processing Pipeline
**Files**: `data/__init__.py` → `data/data_loader.py` → `data/preprocessing.py`

**Reading Order**:
```python
# 1. data/__init__.py - Understand exports and main classes
# 2. data/data_loader.py - Data loading strategies
#    Focus on: DataLoader class, load_dataset_from_hub(), COMMON_DATASETS
# 3. data/preprocessing.py - Task-specific preprocessing
#    Focus on: DataPreprocessor (abstract), TextClassificationPreprocessor
```

**Key Design Patterns**:
- **Abstract Base Class**: `DataPreprocessor` defines interface
- **Strategy Pattern**: Different preprocessors for different tasks
- **Factory Pattern**: `get_preprocessor()` function
- **Flexibility**: Support for HF Hub + custom datasets

**Critical Understanding**:
- How tokenization works for different tasks
- Batch processing and padding strategies
- Label handling for different task types

**Reading Time**: 20 minutes

### 2.2 Model Architecture
**Files**: `models/__init__.py` → `models/base_model.py` → `models/peft_model.py`

**Reading Order**:
```python
# 1. models/__init__.py - See main model classes
# 2. models/base_model.py - Base model wrapper
#    Focus on: BaseModelWrapper class, quantization logic, model loading
# 3. models/peft_model.py - PEFT implementation
#    Focus on: PEFTModelWrapper class, _create_peft_config methods
```

**Key Architecture Decisions**:
- **Wrapper Pattern**: Encapsulate HF models with additional functionality
- **Composition**: PEFTModelWrapper contains BaseModelWrapper
- **Quantization Support**: Integrated 4-bit/8-bit quantization
- **Task-Specific Loading**: Different model types for different tasks

**Critical Understanding**:
- How PEFT configs are created for different methods (LoRA, Prefix, etc.)
- Target module selection logic for different model architectures
- Model state management (training vs inference)

**Reading Time**: 25 minutes

### 2.3 Training System
**Files**: `training/__init__.py` → `training/utils.py` → `training/trainer.py`

**Reading Order**:
```python
# 1. training/__init__.py - Main training exports
# 2. training/utils.py - Training utilities and callbacks
#    Focus on: compute_metrics functions, EarlyStoppingCallback, WandBCallback
# 3. training/trainer.py - Main training orchestration
#    Focus on: PEFTTrainer class, train() method, setup methods
```

**Key Design Patterns**:
- **Template Method**: PEFTTrainer defines training workflow
- **Observer Pattern**: Callbacks for training events
- **Builder Pattern**: Progressive setup of trainer components
- **Facade Pattern**: Simple interface hiding complex HF Trainer setup

**Critical Understanding**:
- How HF Trainer is configured and customized
- Callback system for monitoring and control
- Configuration management during training

**Reading Time**: 15 minutes

---

## 🔍 Phase 3: Integration & Workflows

### 3.1 Evaluation System
**Files**: `evaluation/__init__.py` → `evaluation/metrics.py` → `evaluation/evaluator.py`

**Reading Order**:
```python
# 1. evaluation/__init__.py - Evaluation exports
# 2. evaluation/metrics.py - Metrics calculation
#    Focus on: MetricsCalculator hierarchy, task-specific metrics
# 3. evaluation/evaluator.py - Model evaluation orchestration
#    Focus on: ModelEvaluator class, evaluate_dataset() method
```

**Key Design Patterns**:
- **Strategy Pattern**: Different metrics for different tasks
- **Template Method**: Common evaluation workflow
- **Factory Pattern**: `get_metrics_calculator()`

**Critical Understanding**:
- How predictions are extracted from models
- Batch processing for evaluation
- Metrics calculation for different task types

**Reading Time**: 15 minutes

### 3.2 Inference Pipeline
**Files**: `inference/__init__.py` → `inference/pipeline.py`

**Reading Order**:
```python
# 1. inference/__init__.py - Inference exports
# 2. inference/pipeline.py - Production inference
#    Focus on: InferencePipeline class, task-specific methods
```

**Key Design Decisions**:
- **Facade Pattern**: Simple interface for complex inference
- **Method Overloading**: Different methods for different tasks
- **Batch Processing**: Efficient handling of multiple inputs
- **Pipeline Integration**: Uses HF pipelines under the hood

**Critical Understanding**:
- How models are loaded for inference
- Batch processing strategies
- Task-specific inference methods

**Reading Time**: 15 minutes

### 3.3 End-to-End Workflows
**File**: `cli.py`

**Reading Focus**:
```python
# Focus on: train_command(), evaluate_command(), infer_command()
# Understand: How all components work together
```

**Key Integration Points**:
- Configuration → Model → Data → Training → Evaluation → Inference
- Error handling and logging throughout the pipeline
- CLI argument parsing and validation

**Reading Time**: 15 minutes

---

## 🚀 Phase 4: Advanced Features & Examples

### 4.1 Practical Examples
**Files**: `examples/text_classification.py` → `examples/text_generation.py` → `examples/question_answering.py`

**Reading Strategy**:
```python
# 1. Start with text_classification.py - simplest example
#    Focus on: Complete workflow from config to inference
# 2. Move to text_generation.py - understand generation specifics
#    Focus on: Generation parameters, different decoding strategies
# 3. Finish with question_answering.py - most complex example
#    Focus on: Context handling, answer extraction
```

**Key Learning Points**:
- How to configure different task types
- Dataset preparation for each task
- Model evaluation strategies
- Inference parameter tuning

**Reading Time**: 30 minutes

### 4.2 Interactive Learning
**File**: `notebooks/01_setup_environment.ipynb`

**Reading Focus**:
- Environment verification strategies
- Step-by-step setup process
- Testing and validation approaches

**Reading Time**: 15 minutes

---

## 🧠 Design Philosophy & Mindset

### Core Principles

#### 1. **Modularity & Separation of Concerns**
```
Config → Data → Models → Training → Evaluation → Inference
```
Each module has a single responsibility and clear interfaces.

#### 2. **Extensibility**
- Abstract base classes for easy extension
- Factory patterns for adding new components
- Configuration-driven behavior

#### 3. **Production Readiness**
- Comprehensive error handling
- Logging throughout the pipeline
- Memory optimization features
- Batch processing support

#### 4. **Developer Experience**
- Type hints throughout
- Clear documentation
- Sensible defaults
- Progressive complexity

### Key Abstractions

#### 1. **Configuration as Code**
```python
# Everything is configurable through dataclasses
model_config = ModelConfig(...)
peft_config = PEFTConfig(...)
training_config = TrainingConfig(...)
```

#### 2. **Wrapper Pattern for Models**
```python
# Encapsulate complexity, add functionality
base_wrapper = BaseModelWrapper(config)
peft_wrapper = PEFTModelWrapper(model_config, peft_config)
```

#### 3. **Pipeline Abstraction**
```python
# Hide complexity behind simple interfaces
trainer = PEFTTrainer(...)
trainer.train(dataset, eval_dataset, preprocessor)
```

### Memory Management Strategy

#### 1. **Quantization Support**
- 4-bit and 8-bit quantization
- QLoRA integration
- Memory-efficient loading

#### 2. **Gradient Checkpointing**
- Trade compute for memory
- Configurable through training config

#### 3. **Batch Processing**
- Configurable batch sizes
- Gradient accumulation
- Memory monitoring

---

## 🎯 Reading Checkpoints

After each phase, you should understand:

### ✅ Phase 1 Checkpoint
- [ ] Project scope and capabilities
- [ ] Configuration architecture
- [ ] Main dependencies and their roles

### ✅ Phase 2 Checkpoint  
- [ ] Data loading and preprocessing strategies
- [ ] Model wrapper architecture
- [ ] Training pipeline design
- [ ] How PEFT methods are implemented

### ✅ Phase 3 Checkpoint
- [ ] Evaluation system design
- [ ] Inference pipeline architecture
- [ ] End-to-end workflow integration

### ✅ Phase 4 Checkpoint
- [ ] Practical usage patterns
- [ ] Task-specific implementations
- [ ] Production deployment considerations

---

## 🔄 Next Steps After Reading

1. **Experiment**: Run the examples with different configurations
2. **Extend**: Add a new PEFT method or task type
3. **Optimize**: Implement custom metrics or callbacks
4. **Deploy**: Use the inference pipeline in a production setting

---

## 💡 Pro Tips for Code Reading

1. **Follow the Data Flow**: Trace how data moves through the system
2. **Understand the Abstractions**: Focus on interfaces before implementations
3. **Check the Tests**: Look at how components are tested (if available)
4. **Read the Configs**: Understanding configuration reveals design intent
5. **Start Simple**: Begin with basic examples before complex features

This reading guide should give you a comprehensive understanding of the codebase architecture, design decisions, and implementation strategies used in this LoRA/PEFT framework.
