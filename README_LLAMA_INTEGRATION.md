# Llama 3.1-8B-Instruct Integration for Squash Coaching

This document describes the integration of the Llama 3.1-8B-Instruct model into the squash coaching system for enhanced AI-powered analysis and insights.

## 🚀 Overview

The Llama integration adds advanced AI capabilities to the squash coaching system, providing:

- **AI-powered shot pattern analysis**
- **Player movement optimization insights**
- **Comprehensive match reports**
- **Personalized coaching plans**
- **Tactical recommendations**
- **Performance improvement suggestions**

## 📋 Requirements

### System Requirements
- Python 3.8+
- Virtual environment (recommended)
- Sufficient RAM (8GB+ recommended for model loading)
- GPU with CUDA support (optional, for faster inference)

### Dependencies
The following packages are required and included in `requirements.txt`:
```
transformers==4.46.3
torch>=2.0.1
numpy>=1.24.3
```

## 🔧 Installation

1. **Activate your virtual environment:**
   ```bash
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation:**
   ```bash
   python test_llama_integration.py
   ```

## 🎯 Features

### 1. Shot Pattern Analysis
Analyzes shot distribution and provides AI-powered insights:
```python
from llama_coaching_enhancement import get_llama_enhancer

enhancer = get_llama_enhancer()
analysis = enhancer.analyze_shot_patterns(shot_data)
```

**Output:**
- Shot distribution statistics
- AI-generated pattern analysis
- Specific coaching recommendations
- Tactical suggestions

### 2. Player Movement Analysis
Analyzes player movement patterns and provides optimization insights:
```python
movement_analysis = enhancer.analyze_player_movement(player_positions, court_dimensions)
```

**Output:**
- Movement efficiency analysis
- Court coverage assessment
- Positioning recommendations
- Fitness and conditioning insights

### 3. Match Report Generation
Generates comprehensive match reports with AI insights:
```python
match_report = enhancer.generate_match_report(match_data)
```

**Output:**
- Executive summary
- Key performance indicators
- Tactical analysis
- Technical assessment
- Training recommendations

### 4. Personalized Coaching Plans
Creates tailored coaching plans based on player data:
```python
coaching_plan = enhancer.generate_personalized_coaching_plan(player_profile)
```

**Output:**
- Technical development priorities
- Tactical training focus areas
- Physical conditioning recommendations
- Mental game strategies
- Specific drills and exercises

## 📁 Output Files

The Llama integration generates the following output files:

### AI Analysis (JSON)
- `output/ai_analysis/shot_pattern_analysis.json`
- `output/ai_analysis/movement_analysis.json`
- `output/ai_analysis/match_report.json`
- `output/ai_analysis/personalized_coaching_plan.json`

### Human-Readable Reports (TXT)
- `output/reports/ai_shot_analysis.txt`
- `output/reports/ai_match_report.txt`
- `output/reports/ai_coaching_plan.txt`

## 🧪 Testing

### Run Integration Tests
```bash
python test_llama_integration.py
```

This will test:
- Transformers library import
- Model loading
- Insight generation
- Shot analysis
- System integration

### Run Demo
```bash
python llama_demo.py
```

This demonstrates:
- Basic Llama model usage
- Coaching enhancement features
- Integration capabilities

## 🚀 Usage

### 1. Full System Analysis
```bash
# Run complete analysis with Llama enhancement
python run_analysis.py

# Or with custom video
python run_with_venv.py --video your_video.mp4
```

### 2. Standalone Llama Usage
```python
from llama_coaching_enhancement import LlamaCoachingEnhancer

# Initialize enhancer
enhancer = LlamaCoachingEnhancer()
enhancer.initialize_model()

# Generate coaching insight
insight = enhancer.generate_coaching_insight(
    "As a squash coach, provide tips for improving serve accuracy."
)
print(insight)
```

### 3. Basic Llama Model Usage
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# Generate response
messages = [{"role": "user", "content": "Who are you?"}]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=40)
response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])
print(response)
```

## 🔍 Configuration

### Model Configuration
The Llama enhancer can be configured with different parameters:

```python
enhancer = LlamaCoachingEnhancer(
    model_name="meta-llama/Llama-3.1-8B-Instruct",  # Model to use
    device="cuda"  # Device (auto-detected if None)
)
```

### Generation Parameters
Customize text generation parameters:

```python
insight = enhancer.generate_coaching_insight(
    prompt="Your coaching prompt",
    max_new_tokens=200,  # Maximum tokens to generate
    temperature=0.7,     # Creativity level (0.0-1.0)
    do_sample=True       # Enable sampling
)
```

## 🛠️ Troubleshooting

### Common Issues

1. **Model Loading Fails**
   - Ensure sufficient RAM (8GB+ recommended)
   - Check internet connection for model download
   - Try using CPU if GPU memory is insufficient

2. **CUDA Out of Memory**
   - Reduce batch size
   - Use CPU instead of GPU
   - Close other applications to free memory

3. **Import Errors**
   - Ensure virtual environment is activated
   - Install requirements: `pip install -r requirements.txt`
   - Check Python version (3.8+ required)

### Performance Optimization

1. **GPU Usage**
   - Use CUDA-enabled GPU for faster inference
   - Ensure sufficient GPU memory (8GB+ recommended)

2. **Memory Management**
   - Use `torch.float16` for reduced memory usage
   - Enable `low_cpu_mem_usage=True` for large models

3. **Batch Processing**
   - Process multiple analyses in batches
   - Use async processing for better performance

## 📊 Performance Metrics

### Model Performance
- **Model Size**: ~8B parameters
- **Memory Usage**: ~16GB (GPU) / ~32GB (CPU)
- **Inference Speed**: ~10-50 tokens/second (GPU) / ~2-10 tokens/second (CPU)
- **Response Quality**: High-quality, contextually relevant coaching insights

### System Integration
- **Seamless Integration**: Works with existing squash analysis pipeline
- **Fallback Support**: Graceful degradation when model is unavailable
- **Error Handling**: Comprehensive error handling and logging

## 🔮 Future Enhancements

### Planned Features
- **Multi-modal Analysis**: Combine video and text analysis
- **Real-time Coaching**: Live coaching suggestions during play
- **Player Profiling**: Advanced player behavior analysis
- **Match Prediction**: Predict match outcomes based on patterns

### Model Improvements
- **Fine-tuning**: Custom training on squash-specific data
- **Model Compression**: Reduced model size for faster inference
- **Ensemble Methods**: Combine multiple models for better accuracy

## 📚 API Reference

### LlamaCoachingEnhancer Class

#### Methods

##### `__init__(model_name, device)`
Initialize the coaching enhancer.

**Parameters:**
- `model_name` (str): HuggingFace model name
- `device` (str): Device to run model on

##### `initialize_model()`
Initialize the Llama model and tokenizer.

##### `generate_coaching_insight(prompt, max_new_tokens)`
Generate coaching insight from prompt.

**Parameters:**
- `prompt` (str): Coaching prompt
- `max_new_tokens` (int): Maximum tokens to generate

**Returns:**
- `str`: Generated coaching insight

##### `analyze_shot_patterns(shot_data)`
Analyze shot patterns and provide insights.

**Parameters:**
- `shot_data` (List[Dict]): List of shot dictionaries

**Returns:**
- `Dict`: Analysis results with insights

##### `analyze_player_movement(player_positions, court_dimensions)`
Analyze player movement patterns.

**Parameters:**
- `player_positions` (Dict): Player position data
- `court_dimensions` (tuple): Court dimensions

**Returns:**
- `Dict`: Movement analysis results

##### `generate_match_report(match_data)`
Generate comprehensive match report.

**Parameters:**
- `match_data` (Dict): Match statistics and data

**Returns:**
- `Dict`: Match report with AI insights

##### `generate_personalized_coaching_plan(player_profile)`
Generate personalized coaching plan.

**Parameters:**
- `player_profile` (Dict): Player data and preferences

**Returns:**
- `Dict`: Personalized coaching plan

## 🤝 Contributing

To contribute to the Llama integration:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This integration is part of the squash coaching system and follows the same license terms.

## 🆘 Support

For support with the Llama integration:

1. Check the troubleshooting section
2. Review the test scripts
3. Check the demo examples
4. Open an issue with detailed error information

---

**Note**: The Llama 3.1-8B-Instruct model requires significant computational resources. Ensure your system meets the requirements before running the integration.
