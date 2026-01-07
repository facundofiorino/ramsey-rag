# Model Training Procedure

## 1. Overview

### 1.1 Purpose
- Train custom LLM using Ollama framework with domain-specific data
- Establish repeatable and configurable training workflow
- Optimize model performance for target use cases
- Enable efficient fine-tuning and adaptation

### 1.2 Training Objectives
- Achieve target performance metrics on validation set
- Maintain training efficiency and reproducibility
- Support multiple Ollama model architectures
- Enable iterative improvement through hyperparameter tuning

## 2. Model Selection

### 2.1 Ollama Model Options

#### 2.1.1 Available Base Models
- **Llama 3.2** (1B, 3B, 8B, 70B, 405B parameters)
- **Mistral** (7B parameters)
- **Mixtral** (8x7B MoE)
- **Gemma 2** (2B, 9B, 27B parameters)
- **Qwen 2.5** (0.5B to 72B parameters)
- **Phi-3** (3.8B parameters)
- **Neural Chat** (7B parameters)

#### 2.1.2 Model Selection Criteria
- Task requirements and complexity
- Available computational resources
- Inference latency constraints
- Memory footprint limitations
- License compatibility
- Community support and updates

#### 2.1.3 Recommended Starting Points
- **Small-scale experiments:** Llama 3.2 3B or Phi-3
- **Balanced performance:** Llama 3.2 8B or Mistral 7B
- **High performance:** Llama 3.2 70B or Mixtral 8x7B
- **Resource-constrained:** Gemma 2 2B or Qwen 2.5 0.5B

### 2.2 Model Configuration System

#### 2.2.1 Configuration File Structure
- **Location:** configs/model_config.yaml
```yaml
model:
  base_model: "llama3.2:8b"  # Configurable Ollama model
  model_family: "llama"
  context_window: 8192
  quantization: "q4_K_M"  # Optional quantization

training:
  method: "fine-tuning"  # or "continued-pretraining"
  epochs: 3
  batch_size: 4
  gradient_accumulation_steps: 8
  learning_rate: 2e-5
  warmup_steps: 100

advanced:
  lora_enabled: true  # LoRA for efficient fine-tuning
  lora_rank: 8
  lora_alpha: 16
  lora_dropout: 0.05
```

#### 2.2.2 Model Registry
- **Location:** configs/model_registry.yaml
- Catalog of available Ollama models
- Model specifications and requirements
- Performance benchmarks
- Resource requirements

### 2.3 Fine-Tuning vs. Continued Pre-Training

#### 2.3.1 Fine-Tuning Approach
- Start from instruction-tuned model
- Adapt to domain-specific tasks
- Faster convergence
- Lower computational cost
- **Use when:** Task-specific adaptation needed

#### 2.3.2 Continued Pre-Training
- Start from base model
- Learn domain knowledge from scratch
- Requires more data and compute
- Better for specialized domains
- **Use when:** Domain is significantly different from base training

## 3. Training Environment Setup

### 3.1 Hardware Requirements

#### 3.1.1 Minimum Requirements
- **CPU:** 8+ cores
- **RAM:** 32GB for 7B models, 64GB+ for larger models
- **GPU:** Optional but highly recommended
  - NVIDIA GPU with 16GB+ VRAM for 7B-8B models
  - 24GB+ VRAM for 13B-30B models
  - 48GB+ VRAM for 70B+ models
- **Storage:** 100GB+ SSD for model and data

#### 3.1.2 Recommended Setup
- **GPU:** NVIDIA A100 (40GB/80GB) or H100
- **RAM:** 128GB+ for large model training
- **Storage:** NVMe SSD for fast I/O
- Multi-GPU setup for distributed training

### 3.2 Software Environment

#### 3.2.1 Core Dependencies
- **Python:** 3.10+
- **Ollama:** Latest version installed
- **PyTorch:** 2.5.1 with CUDA support
- **Transformers:** 4.48.1
- **Accelerate:** 1.3.0 for distributed training
- **bitsandbytes:** For quantization (optional)

#### 3.2.2 Environment Setup
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull base model
ollama pull llama3.2:8b

# Activate virtual environment
source venv/bin/activate

# Verify GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

### 3.3 Ollama Integration

#### 3.3.1 Ollama API Setup
- **Location:** src/models/ollama_client.py
- Connect to Ollama server
- Model loading and management
- Generation configuration
- Error handling and retries

#### 3.3.2 Model Pulling and Management
```python
# Example: Programmatic model management
import ollama

# Pull model if not available
ollama.pull('llama3.2:8b')

# List available models
models = ollama.list()

# Get model info
info = ollama.show('llama3.2:8b')
```

## 4. Training Data Preparation

### 4.1 Data Loading
- **Location:** src/models/data_loader.py
- Load processed data from data/processed/
- Support for multiple data formats (JSONL, JSON, Parquet)
- Memory-efficient loading for large datasets
- Data streaming for datasets larger than RAM

### 4.2 Training Data Format

#### 4.2.1 Instruction Format (for fine-tuning)
```json
{
  "instruction": "Task description or question",
  "context": "Relevant context from ramsey_data",
  "response": "Expected model output"
}
```

#### 4.2.2 Completion Format (for continued pre-training)
```json
{
  "text": "Full text from preprocessed chunks"
}
```

### 4.3 Data Preprocessing for Training
- **Location:** src/models/training_preprocessor.py
- Convert preprocessed data to training format
- Tokenization with Ollama-compatible tokenizer
- Padding and truncation strategies
- Attention mask creation

### 4.4 DataLoader Configuration
- Batch size optimization
- Shuffling strategy
- Num workers for parallel loading
- Pin memory for GPU training
- Prefetching for efficiency

## 5. Hyperparameter Tuning

### 5.1 Key Hyperparameters

#### 5.1.1 Learning Rate
- **Range:** 1e-6 to 5e-5
- **Starting point:** 2e-5 for fine-tuning
- Learning rate scheduling (linear, cosine, polynomial)
- Warmup steps (typically 5-10% of total steps)

#### 5.1.2 Batch Size and Gradient Accumulation
- **Effective batch size:** 32-128
- Physical batch size limited by GPU memory
- Gradient accumulation to achieve effective batch size
- Trade-off between speed and memory

#### 5.1.3 Training Duration
- **Epochs:** 1-5 for fine-tuning, 10+ for pre-training
- Early stopping based on validation performance
- Save checkpoints at regular intervals
- Monitor for overfitting

#### 5.1.4 LoRA Parameters (if enabled)
- **Rank (r):** 4-32 (typical: 8-16)
- **Alpha:** Typically 2*rank or equal to rank
- **Dropout:** 0.0-0.1
- **Target modules:** Query, key, value projections

### 5.2 Hyperparameter Search Strategy

#### 5.2.1 Search Methods
- **Grid Search:** Exhaustive but expensive
- **Random Search:** More efficient exploration
- **Bayesian Optimization:** Intelligent search (Ray Tune)
- **Manual Tuning:** Based on validation metrics

#### 5.2.2 Search Space Definition
- **Location:** configs/hyperparameter_search.yaml
```yaml
search_space:
  learning_rate: [1e-5, 2e-5, 5e-5]
  batch_size: [4, 8, 16]
  lora_rank: [8, 16, 32]
  epochs: [2, 3, 4]
```

#### 5.2.3 Search Execution
- **Location:** src/models/hyperparameter_search.py
- Parallel trial execution
- Early stopping for poor performers
- Result tracking and visualization

### 5.3 Optimization Techniques

#### 5.3.1 Optimizer Selection
- **AdamW:** Default choice, good general performance
- **Adam:** Alternative without weight decay
- **SGD with momentum:** For specific cases
- Optimizer parameters (beta1, beta2, epsilon, weight_decay)

#### 5.3.2 Learning Rate Scheduling
- **Cosine Annealing:** Smooth decay
- **Linear Warmup + Decay:** Common choice
- **Constant with Warmup:** For short training
- **Polynomial Decay:** Alternative to linear

#### 5.3.3 Regularization
- Weight decay: 0.01-0.1
- Dropout in LoRA layers
- Gradient clipping: max_norm=1.0
- Early stopping patience

## 6. Training Workflow

### 6.1 Training Pipeline Architecture
```
Load Configuration
    ↓
Initialize Ollama Model
    ↓
Load Training Data
    ↓
Setup Training Components
    ↓
Training Loop
    ├─ Forward Pass
    ├─ Loss Calculation
    ├─ Backward Pass
    ├─ Gradient Accumulation
    ├─ Optimizer Step
    └─ Validation Check
    ↓
Save Final Model
    ↓
Evaluation
```

### 6.2 Training Script
- **Location:** src/models/train.py
- Main training orchestration
- Distributed training support
- Checkpoint management
- Logging and monitoring

### 6.3 Training Loop Implementation

#### 6.3.1 Epoch Structure
```python
for epoch in range(num_epochs):
    # Training phase
    model.train()
    for batch in train_dataloader:
        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss

        # Backward pass
        loss.backward()

        # Gradient accumulation
        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    # Validation phase
    model.eval()
    val_loss = validate(model, val_dataloader)

    # Checkpoint saving
    if val_loss < best_val_loss:
        save_checkpoint(model, optimizer, epoch)
```

#### 6.3.2 Gradient Accumulation
- Simulate large batch sizes on limited memory
- Accumulate gradients over multiple mini-batches
- Update weights after accumulation
- Scale loss by accumulation steps

#### 6.3.3 Mixed Precision Training
- **Tool:** PyTorch AMP (Automatic Mixed Precision)
- FP16/BF16 training for speed and memory efficiency
- Gradient scaling for numerical stability
- Faster training on modern GPUs

### 6.4 Distributed Training (Optional)

#### 6.4.1 Multi-GPU Training
- **Tool:** PyTorch DistributedDataParallel (DDP)
- Data parallelism across GPUs
- Gradient synchronization
- Efficient GPU utilization

#### 6.4.2 DeepSpeed Integration (Optional)
- ZeRO optimization stages
- Memory-efficient training
- Support for larger models
- Pipeline parallelism

## 7. Checkpointing and Model Saving

### 7.1 Checkpoint Strategy
- **Location:** checkpoints/
- Save at end of each epoch
- Keep best N checkpoints based on validation loss
- Include optimizer state for resumable training
- Metadata (config, metrics, timestamp)

### 7.2 Checkpoint Content
```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'train_loss': train_loss,
    'val_loss': val_loss,
    'config': config,
    'timestamp': timestamp
}
```

### 7.3 Model Export

#### 7.3.1 Ollama Model Export
- Export trained model in Ollama format
- Create Modelfile for Ollama
- Push to local Ollama registry
- Test model availability

#### 7.3.2 Modelfile Creation
```dockerfile
FROM llama3.2:8b

# Set parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40

# Set custom system message
SYSTEM """You are a helpful assistant trained on Ramsey data."""

# Add adapter weights (if using LoRA)
ADAPTER ./adapter_model
```

### 7.4 Resume Training
- Load checkpoint and resume
- Verify epoch and step count
- Resume from exact state
- Handle interrupted training

## 8. Monitoring and Logging

### 8.1 Training Metrics

#### 8.1.1 Loss Metrics
- Training loss (per batch, per epoch)
- Validation loss
- Loss curves and trends
- Gradient norms

#### 8.1.2 Performance Metrics
- Perplexity
- Learning rate trajectory
- Training throughput (samples/sec, tokens/sec)
- GPU utilization and memory usage

### 8.2 Logging Infrastructure

#### 8.2.1 Console Logging
- **Tool:** tqdm for progress bars
- Real-time training progress
- Current metrics display
- ETA estimation

#### 8.2.2 File Logging
- **Location:** logs/training/
- Detailed training logs
- Timestamped entries
- Error and warning tracking

#### 8.2.3 Experiment Tracking (Optional)
- **Tools:** MLflow, Weights & Biases, TensorBoard
- Hyperparameter logging
- Metric visualization
- Model comparison
- Experiment reproducibility

### 8.3 Visualization

#### 8.3.1 Training Curves
- Loss curves (train vs. validation)
- Learning rate schedule
- Gradient norms
- Generate plots automatically

#### 8.3.2 Model Comparison
- Compare different model configurations
- Hyperparameter impact analysis
- Training efficiency metrics

## 9. Validation During Training

### 9.1 Validation Strategy
- Validate at end of each epoch
- Track validation metrics
- Early stopping on validation plateau
- Save best model based on validation

### 9.2 Validation Metrics
- Validation loss (primary metric)
- Perplexity
- Task-specific metrics (if applicable)
- Sample generation quality (qualitative)

### 9.3 Overfitting Detection
- Monitor train vs. validation gap
- Detect performance degradation
- Implement early stopping
- Adjust regularization if needed

## 10. Execution Commands

### 10.1 Basic Training
```bash
python src/models/train.py \
  --config configs/model_config.yaml \
  --data data/processed/ \
  --output models/trained/
```

### 10.2 Training with Custom Parameters
```bash
python src/models/train.py \
  --config configs/model_config.yaml \
  --model llama3.2:8b \
  --epochs 3 \
  --batch-size 4 \
  --learning-rate 2e-5 \
  --output models/trained/
```

### 10.3 Resume from Checkpoint
```bash
python src/models/train.py \
  --config configs/model_config.yaml \
  --resume checkpoints/epoch_2.pt \
  --output models/trained/
```

### 10.4 Hyperparameter Search
```bash
python src/models/hyperparameter_search.py \
  --config configs/hyperparameter_search.yaml \
  --data data/processed/ \
  --output experiments/
```

### 10.5 Multi-GPU Training
```bash
torchrun --nproc_per_node=4 src/models/train.py \
  --config configs/model_config.yaml \
  --distributed \
  --output models/trained/
```

## 11. Testing and Validation

### 11.1 Unit Tests
- **Location:** tests/models/test_training.py
- Test data loading
- Test model initialization
- Test training step
- Mock training runs

### 11.2 Integration Tests
- End-to-end training on small dataset
- Validate checkpoint saving/loading
- Test model export
- Verify Ollama integration

### 11.3 Smoke Tests
- Quick training run (1-2 batches)
- Verify no crashes
- Check GPU memory usage
- Validate output formats

## 12. Troubleshooting

### 12.1 Common Issues

#### 12.1.1 Out of Memory (OOM)
- **Solutions:**
  - Reduce batch size
  - Increase gradient accumulation steps
  - Enable gradient checkpointing
  - Use smaller model variant
  - Reduce context length

#### 12.1.2 Slow Training
- **Solutions:**
  - Increase batch size if memory allows
  - Enable mixed precision training
  - Optimize data loading (num_workers, pin_memory)
  - Use faster storage (SSD)
  - Profile and identify bottlenecks

#### 12.1.3 Poor Convergence
- **Solutions:**
  - Adjust learning rate
  - Check data quality
  - Verify loss calculation
  - Increase warmup steps
  - Try different optimizer

#### 12.1.4 Ollama Integration Issues
- **Solutions:**
  - Verify Ollama server is running
  - Check model availability
  - Validate API endpoints
  - Review Ollama logs

### 12.2 Debug Mode
```bash
python src/models/train.py \
  --config configs/model_config.yaml \
  --debug \
  --max-steps 10
```

### 12.3 Profiling
- PyTorch profiler for bottleneck identification
- Memory profiling (torch.cuda.memory_summary)
- CPU/GPU utilization monitoring
- I/O performance analysis

## 13. Best Practices

### 13.1 Training Guidelines
- Start with small model for rapid prototyping
- Establish baseline before optimization
- Use validation set for all decisions
- Keep detailed training logs
- Version control all configurations

### 13.2 Reproducibility
- Set random seeds (Python, NumPy, PyTorch)
- Document exact package versions
- Save all hyperparameters
- Use deterministic algorithms when possible
- Track git commit hash

### 13.3 Resource Management
- Monitor GPU memory usage
- Clean up unused tensors
- Use gradient checkpointing for large models
- Implement proper error handling
- Graceful shutdown on interruption

## 14. Post-Training Steps

### 14.1 Model Export to Ollama
```bash
# Create Modelfile
python src/models/create_modelfile.py \
  --checkpoint models/trained/best_model.pt \
  --output models/ollama/

# Create Ollama model
ollama create ramsey-model -f models/ollama/Modelfile

# Test model
ollama run ramsey-model "Test prompt"
```

### 14.2 Model Validation
- Test on held-out test set
- Generate sample outputs
- Compare with baseline
- Document performance

### 14.3 Model Versioning
- Tag model with version number
- Document changes from previous version
- Store in model registry
- Create release notes

## 15. Future Enhancements

### 15.1 Planned Improvements
- Distributed training across multiple nodes
- Advanced pruning and quantization
- Continuous learning pipeline
- Automated hyperparameter optimization

### 15.2 Advanced Techniques
- Curriculum learning
- Multi-task learning
- Knowledge distillation
- Retrieval-augmented generation (RAG) integration
