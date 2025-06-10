# 🔗 Cross-Modal Training - Multi-Modal AI System

This project implements comprehensive cross-modal training techniques for multi-modal AI models, enabling training and inference across different modalities (text, image, audio, video).

## 📋 What is Cross-Modal Training?

Cross-Modal Training is the process of training AI models to understand and process multiple modalities simultaneously:
- **Text-Image**: Understanding relationships between text descriptions and images
- **Text-Audio**: Connecting spoken language with textual content
- **Image-Audio**: Linking visual content with audio descriptions
- **Multi-Modal Fusion**: Combining multiple modalities for enhanced understanding

## 🎯 Why Cross-Modal Training Matters

### Traditional Single-Modal Limitations
```
Single-Modal Approach:
Text Model: "A cat sitting on a chair" → Text understanding only
Image Model: [Cat image] → Visual understanding only
Audio Model: [Cat sound] → Audio understanding only

Problems:
→ No cross-modal understanding
→ Cannot relate text descriptions to images
→ Missing rich multi-modal context
→ Limited real-world applicability
```

### Cross-Modal Solution
```
Cross-Modal Approach:
Input: Text + Image + Audio
Model: Multi-Modal Transformer
Output: Rich understanding across all modalities

Benefits:
→ Unified representation across modalities
→ Better context understanding
→ Enhanced performance on complex tasks
→ Real-world applicability (VQA, captioning, retrieval)
→ Zero-shot cross-modal transfer
```

## 📁 Project Structure

```
cross-modal-training/
├── README.md                    # This file
├── requirements.txt             # Dependencies
├── setup.py                     # Package setup
├── src/                         # Source code
│   ├── models/                  # Multi-modal models
│   │   ├── clip_model.py        # CLIP implementation
│   │   ├── blip_model.py        # BLIP implementation
│   │   ├── flamingo_model.py    # Flamingo implementation
│   │   └── custom_fusion.py     # Custom fusion architectures
│   ├── data/                    # Data processing
│   │   ├── multimodal_dataset.py # Multi-modal datasets
│   │   ├── text_processor.py    # Text preprocessing
│   │   ├── image_processor.py   # Image preprocessing
│   │   └── audio_processor.py   # Audio preprocessing
│   ├── training/                # Training systems
│   │   ├── cross_modal_trainer.py # Cross-modal training
│   │   ├── contrastive_learning.py # Contrastive learning
│   │   └── fusion_trainer.py    # Fusion training
│   ├── inference/               # Inference pipelines
│   │   ├── multimodal_pipeline.py # Multi-modal inference
│   │   ├── retrieval_system.py  # Cross-modal retrieval
│   │   └── generation_pipeline.py # Multi-modal generation
│   ├── evaluation/              # Evaluation metrics
│   │   ├── cross_modal_metrics.py # Cross-modal evaluation
│   │   ├── retrieval_metrics.py # Retrieval evaluation
│   │   └── generation_metrics.py # Generation evaluation
│   └── utils/                   # Utility functions
│       ├── visualization.py     # Multi-modal visualization
│       ├── data_utils.py        # Data utilities
│       └── model_utils.py       # Model utilities
├── examples/                    # Complete examples
│   ├── image_captioning.py      # Image captioning
│   ├── visual_question_answering.py # VQA
│   ├── text_to_image_retrieval.py # Cross-modal retrieval
│   └── audio_visual_learning.py # Audio-visual learning
├── experiments/                 # Experiment scripts
│   ├── clip_training.py         # CLIP training experiment
│   ├── blip_finetuning.py       # BLIP fine-tuning
│   └── custom_fusion_training.py # Custom fusion training
├── notebooks/                   # Jupyter notebooks
│   ├── cross_modal_tutorial.ipynb # Tutorial notebook
│   ├── model_comparison.ipynb   # Model comparison
│   └── visualization_demo.ipynb # Visualization demo
├── tests/                       # Test files
└── docs/                        # Documentation
    ├── model_architectures.md
    ├── training_strategies.md
    └── evaluation_protocols.md
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd cross-modal-training

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### 2. Basic Usage

```python
from src.models import CLIPModel
from src.data import MultiModalDataset
from src.training import CrossModalTrainer

# Load pre-trained CLIP model
model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')

# Prepare multi-modal dataset
dataset = MultiModalDataset(
    data_path='path/to/multimodal/data',
    modalities=['text', 'image']
)

# Initialize trainer
trainer = CrossModalTrainer(
    model=model,
    dataset=dataset,
    learning_rate=1e-4,
    batch_size=32
)

# Train the model
trainer.train(num_epochs=10)
```

### 3. Run Examples

```bash
# Image captioning example
python examples/image_captioning.py

# Visual question answering
python examples/visual_question_answering.py

# Cross-modal retrieval
python examples/text_to_image_retrieval.py

# Audio-visual learning
python examples/audio_visual_learning.py
```

## 🔧 Key Features

### ✅ Multiple Model Architectures
- **CLIP (Contrastive Language-Image Pre-training)**
- **BLIP (Bootstrapped Language-Image Pre-training)**
- **Flamingo (Few-shot learning with multi-modal models)**
- **Custom Fusion Architectures**

### ✅ Cross-Modal Training Strategies
- **Contrastive Learning** for alignment
- **Masked Language/Image Modeling**
- **Cross-Modal Attention** mechanisms
- **Multi-Task Learning** across modalities

### ✅ Multi-Modal Data Support
- **Text-Image** pairs (COCO, Flickr30k)
- **Text-Audio** pairs (AudioCaps, Clotho)
- **Video-Text** pairs (MSR-VTT, ActivityNet)
- **Custom multi-modal datasets**

### ✅ Advanced Training Techniques
- **Parameter-Efficient Fine-tuning** (LoRA, AdaLoRA)
- **Gradient Accumulation** for large batches
- **Mixed Precision Training** for efficiency
- **Distributed Training** support

## 📊 Supported Tasks

### 1. Image Captioning
```python
from src.inference import ImageCaptioningPipeline

# Initialize pipeline
captioner = ImageCaptioningPipeline.from_pretrained('blip-image-captioning-base')

# Generate caption
image_path = 'path/to/image.jpg'
caption = captioner(image_path)
print(f"Caption: {caption}")
```

### 2. Visual Question Answering
```python
from src.inference import VQAPipeline

# Initialize VQA pipeline
vqa = VQAPipeline.from_pretrained('blip-vqa-base')

# Answer question about image
image_path = 'path/to/image.jpg'
question = "What color is the cat?"
answer = vqa(image_path, question)
print(f"Answer: {answer}")
```

### 3. Cross-Modal Retrieval
```python
from src.inference import CrossModalRetrieval

# Initialize retrieval system
retriever = CrossModalRetrieval.from_pretrained('clip-vit-base-patch32')

# Text-to-image retrieval
query = "A dog playing in the park"
similar_images = retriever.text_to_image(query, top_k=5)

# Image-to-text retrieval
image_path = 'path/to/image.jpg'
similar_texts = retriever.image_to_text(image_path, top_k=5)
```

### 4. Multi-Modal Generation
```python
from src.inference import MultiModalGenerator

# Initialize generator
generator = MultiModalGenerator.from_pretrained('flamingo-3b')

# Generate text from image and context
image_path = 'path/to/image.jpg'
context = "This image shows"
generated_text = generator.generate(
    image=image_path,
    text_context=context,
    max_length=100
)
```

## 🎯 Model Architectures

### CLIP (Contrastive Language-Image Pre-training)
```python
class CLIPModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.text_encoder = TextEncoder(config.text_config)
        self.vision_encoder = VisionEncoder(config.vision_config)
        self.text_projection = nn.Linear(config.text_embed_dim, config.projection_dim)
        self.visual_projection = nn.Linear(config.vision_embed_dim, config.projection_dim)
        
    def forward(self, input_ids, pixel_values, attention_mask=None):
        # Encode text and images
        text_features = self.text_encoder(input_ids, attention_mask)
        image_features = self.vision_encoder(pixel_values)
        
        # Project to shared space
        text_embeds = self.text_projection(text_features)
        image_embeds = self.visual_projection(image_features)
        
        # Normalize embeddings
        text_embeds = F.normalize(text_embeds, dim=-1)
        image_embeds = F.normalize(image_embeds, dim=-1)
        
        return {
            'text_embeds': text_embeds,
            'image_embeds': image_embeds,
            'logits_per_text': text_embeds @ image_embeds.T,
            'logits_per_image': image_embeds @ text_embeds.T
        }
```

### BLIP (Bootstrapped Language-Image Pre-training)
```python
class BLIPModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vision_encoder = VisionTransformer(config.vision_config)
        self.text_encoder = BertModel(config.text_config)
        self.text_decoder = BertLMHeadModel(config.text_config)
        self.cross_attention = CrossAttentionLayer(config)
        
    def forward(self, pixel_values, input_ids, decoder_input_ids=None):
        # Encode image
        image_features = self.vision_encoder(pixel_values)
        
        # Encode text (for understanding tasks)
        text_features = self.text_encoder(input_ids)
        
        # Cross-modal fusion
        fused_features = self.cross_attention(text_features, image_features)
        
        # Generate text (for generation tasks)
        if decoder_input_ids is not None:
            generation_output = self.text_decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=image_features
            )
            return generation_output
        
        return fused_features
```

## 📈 Performance Metrics

### Cross-Modal Retrieval Metrics
```python
# Text-to-Image Retrieval
text_to_image_r1 = 0.58  # Recall@1
text_to_image_r5 = 0.81  # Recall@5
text_to_image_r10 = 0.88 # Recall@10

# Image-to-Text Retrieval
image_to_text_r1 = 0.43  # Recall@1
image_to_text_r5 = 0.73  # Recall@5
image_to_text_r10 = 0.82 # Recall@10
```

### Image Captioning Metrics
```python
# COCO Captions Evaluation
bleu_4 = 0.32      # BLEU-4 score
meteor = 0.27      # METEOR score
rouge_l = 0.56     # ROUGE-L score
cider = 1.15       # CIDEr score
spice = 0.21       # SPICE score
```

### Visual Question Answering Metrics
```python
# VQA v2.0 Evaluation
overall_accuracy = 0.72    # Overall accuracy
yes_no_accuracy = 0.87     # Yes/No questions
number_accuracy = 0.45     # Number questions
other_accuracy = 0.65      # Other questions
```

## 🔬 Advanced Features

### 1. Parameter-Efficient Fine-tuning
```python
from src.training import LoRATrainer

# Fine-tune with LoRA
trainer = LoRATrainer(
    model=model,
    lora_config={
        'r': 16,
        'lora_alpha': 32,
        'target_modules': ['q_proj', 'v_proj'],
        'lora_dropout': 0.1
    }
)

trainer.train(dataset, num_epochs=5)
```

### 2. Contrastive Learning
```python
from src.training import ContrastiveLearning

# Contrastive learning setup
contrastive_trainer = ContrastiveLearning(
    model=model,
    temperature=0.07,
    negative_sampling='hard',
    batch_size=256
)

contrastive_trainer.train(multimodal_dataset)
```

### 3. Multi-Task Learning
```python
from src.training import MultiTaskTrainer

# Multi-task training
tasks = {
    'captioning': image_captioning_dataset,
    'vqa': vqa_dataset,
    'retrieval': retrieval_dataset
}

multitask_trainer = MultiTaskTrainer(
    model=model,
    tasks=tasks,
    task_weights={'captioning': 1.0, 'vqa': 0.5, 'retrieval': 0.3}
)

multitask_trainer.train(num_epochs=10)
```

### 4. Zero-Shot Transfer
```python
from src.inference import ZeroShotClassifier

# Zero-shot image classification
classifier = ZeroShotClassifier.from_pretrained('clip-vit-large-patch14')

# Classify image with text labels
image_path = 'path/to/image.jpg'
candidate_labels = ['cat', 'dog', 'bird', 'car']
predictions = classifier(image_path, candidate_labels)

print(f"Predictions: {predictions}")
```

## 🧪 Evaluation Protocols

### Cross-Modal Retrieval Evaluation
```python
from src.evaluation import RetrievalEvaluator

evaluator = RetrievalEvaluator()

# Evaluate text-to-image retrieval
text_queries = ["A cat sitting on a chair", "A dog running in the park"]
image_database = load_image_database()

metrics = evaluator.evaluate_text_to_image(
    model=model,
    text_queries=text_queries,
    image_database=image_database,
    k_values=[1, 5, 10]
)

print(f"Text-to-Image R@1: {metrics['recall_at_1']}")
print(f"Text-to-Image R@5: {metrics['recall_at_5']}")
print(f"Text-to-Image R@10: {metrics['recall_at_10']}")
```

### Image Captioning Evaluation
```python
from src.evaluation import CaptioningEvaluator

evaluator = CaptioningEvaluator()

# Evaluate image captioning
test_images = load_test_images()
reference_captions = load_reference_captions()

generated_captions = []
for image in test_images:
    caption = model.generate_caption(image)
    generated_captions.append(caption)

metrics = evaluator.evaluate(
    generated_captions=generated_captions,
    reference_captions=reference_captions
)

print(f"BLEU-4: {metrics['bleu_4']}")
print(f"METEOR: {metrics['meteor']}")
print(f"CIDEr: {metrics['cider']}")
```

## 🎨 Visualization Tools

### Multi-Modal Attention Visualization
```python
from src.utils import AttentionVisualizer

visualizer = AttentionVisualizer(model)

# Visualize cross-modal attention
image_path = 'path/to/image.jpg'
text = "A beautiful sunset over the ocean"

attention_map = visualizer.visualize_cross_attention(
    image=image_path,
    text=text,
    layer=6,
    head=8
)

visualizer.plot_attention_map(attention_map)
```

### Embedding Space Visualization
```python
from src.utils import EmbeddingVisualizer

visualizer = EmbeddingVisualizer()

# Visualize text and image embeddings in shared space
text_embeddings = model.encode_text(text_samples)
image_embeddings = model.encode_image(image_samples)

visualizer.plot_embedding_space(
    text_embeddings=text_embeddings,
    image_embeddings=image_embeddings,
    labels=labels,
    method='tsne'
)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests and documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- [OpenAI CLIP](https://github.com/openai/CLIP) - Contrastive Language-Image Pre-training
- [Salesforce BLIP](https://github.com/salesforce/BLIP) - Bootstrapped Language-Image Pre-training
- [DeepMind Flamingo](https://www.deepmind.com/blog/tackling-multiple-tasks-with-a-single-visual-language-model) - Few-shot learning
- [Hugging Face Transformers](https://github.com/huggingface/transformers) - Multi-modal model implementations
