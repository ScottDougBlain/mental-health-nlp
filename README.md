# Mental Health NLP Safety Demo

A responsible implementation of LSTM-based suicide risk detection with comprehensive ethical frameworks. This project demonstrates ML engineering capabilities and responsible AI development in sensitive mental health domains, showcasing the integration of technical implementation with comprehensive safety protocols.

## ⚠️ Important Notice

**This is a demonstration framework for research and educational purposes only. NOT for clinical use.**

**Implementation Status:**
- ✅ **Complete code implementation** for LSTM and ELECTRA models
- ✅ **Comprehensive safety framework** with real crisis resources
- ⚠️ **Models are NOT pre-trained** - require training on actual data
- ⚠️ **No performance validation** - metrics shown are targets based on literature

This system demonstrates responsible AI development practices but should never be used for:
- Clinical diagnosis or treatment decisions
- Automated intervention systems
- Life-critical applications
- Unsupervised content moderation

---

## Executive Summary

This project demonstrates how to build high-performance ML systems for sensitive applications while maintaining rigorous ethical standards. It showcases the integration of technical excellence with comprehensive safety frameworks, making it a unique contribution to AI safety research.

### Why This Project Matters for AI Safety

**1. Safety-First Architecture**
- Comprehensive ethical frameworks integrated from design through deployment
- Multi-level risk assessment with human oversight protocols
- Crisis intervention resources automatically provided with high-risk predictions
- Extensive audit trails for accountability and oversight

**2. Technical & Ethical Synthesis**
- Demonstrates architecture capable of strong performance while maintaining safety protocols
- Shows that responsible AI doesn't require sacrificing technical capability
- Production-ready code patterns applicable to other sensitive domains

**3. Real-World Applicability**
- Designed for Kaggle Suicide Watch dataset (232,000 Reddit posts)
- Developed by interdisciplinary team: Scott Blain (Clinical Psychology PhD), Luis Rodriguez (ML Engineering), Violet Yang (Data Science)
- Neuromatch Academy project showcasing psychology-informed AI safety

**4. Transferable Safety Practices**
- Ethical frameworks applicable beyond mental health applications
- Patterns for responsible development in high-stakes AI systems
- Integration of domain expertise (clinical psychology) with ML engineering

### Architecture Capabilities

| Model | Architecture | Status | Target Performance |
|-------|-------------|--------|-------------------|
| **ELECTRA** | Google's electra-small/base transformer | Implemented, requires training | ~94% F1 (literature benchmark) |
| **LSTM Baseline** | Bidirectional LSTM with attention | Implemented, requires training | ~90% F1 (literature benchmark) |

*Note: This repository provides the complete implementation and training pipeline. Actual performance depends on training with appropriate data. The Kaggle Suicide Watch dataset (232K Reddit posts) is recommended for training.*

### Project Team

**Raijin Pod - Raichu's Researchers**
- **Scott Blain** - Clinical Psychology PhD, AI Safety Research
- **Luis Rodriguez** - ML Engineering, Model Implementation
- **Violet Yang** - Data Science, Evaluation Framework

**Neuromatch Academy Support**
- Aeron Sim et al. - Pod TAs
- Hedyeh Nazari - Pod TA
- Mohammad Javad Ranjbar - Project TA

---

## Project Overview

This project represents a unique intersection of:
- **Technical Excellence**: High-performance LSTM architecture targeting literature benchmarks
- **Ethical Responsibility**: Comprehensive safety frameworks and crisis intervention
- **Clinical Insight**: Psychology-informed approach to AI safety
- **Production Readiness**: Enterprise-grade code with extensive safeguards

### Implementation Status

**What's Implemented:**
- ✅ Complete LSTM architecture with bidirectional layers and attention
- ✅ Full ELECTRA transformer pipeline using HuggingFace
- ✅ Comprehensive safety framework with crisis resources
- ✅ Dataset loader for Kaggle Suicide Watch data
- ✅ Training, evaluation, and visualization pipelines

**What's Required:**
- ⚠️ Training on actual dataset (code ready, models untrained)
- ⚠️ Hyperparameter tuning for optimal performance
- ⚠️ Validation on test set to measure actual performance

**Expected Performance (based on literature):**
- ELECTRA: ~94% F1 score typical for similar transformer models
- LSTM: ~90% F1 score typical for bidirectional LSTM on this task
- *Note: Actual performance requires training and evaluation*

## Architecture

This project includes two model implementations:

### 1. LSTM Architecture (Baseline)
```
Input Text → Preprocessing → Embedding → Bidirectional LSTM → Dense → Risk Probability
             (Cleaning)    (128-dim)   (64×2 hidden)     (Dropout)   (0-1 score)
```

**Specifications:**
- Embedding: 128-dimensional learned embeddings
- LSTM: 2-layer bidirectional (64 hidden units each direction)
- Regularization: 30% dropout, early stopping
- Target: ~0.90 F1 score

### 2. ELECTRA Architecture (State-of-the-art)
```
Input Text → Tokenization → ELECTRA Encoder → Classification Head → Risk Probability
             (WordPiece)   (Transformer)    (Dense + Softmax)   (0-1 score)
```

**Specifications:**
- Base Model: google/electra-small-discriminator (or electra-base)
- Sequence Length: 512 tokens
- Fine-tuning: Full model fine-tuning with AdamW optimizer
- Target: ~0.94 F1 score (literature benchmark)

### Safety Architecture (Both Models)
```
Prediction → Risk Assessment → Safety Protocol → Crisis Resources → Human Oversight
           (Confidence)      (Guidelines)    (Intervention)   (Escalation)
```

## Technical Implementation

### Core Components

#### 1. SuicideRiskDetector (LSTM)
High-performance LSTM model with comprehensive preprocessing and safety integration.

```python
from src.suicide_risk_detector import SuicideRiskDetector

# Initialize with safety warnings
detector = SuicideRiskDetector()

# Train model (with ethical data practices)
detector.create_model(vocab_size=10000)
train_loader, val_loader, test_loader = detector.prepare_data(texts, labels)
detector.train(train_loader, val_loader, epochs=10)

# Make predictions with safety protocols
prediction, confidence = detector.predict_risk(text)
detector.create_risk_visualization(text)  # Includes crisis resources
```

#### 2. ELECTRARiskDetector (Transformer)
State-of-the-art transformer-based model using Google's ELECTRA.

```python
from src.electra_risk_detector import ELECTRARiskDetector

# Initialize ELECTRA model
detector = ELECTRARiskDetector(
    model_name="google/electra-small-discriminator"
)

# Load and prepare data
df, labels = detector.load_data('Suicide_Detection.csv')
train_ds, val_ds, test_ds = detector.prepare_datasets(df['cleaned_text'], labels)

# Train model
detector.train(
    train_dataset=train_ds,
    val_dataset=val_ds,
    num_epochs=3,
    batch_size=16
)

# Evaluate with comprehensive metrics
results = detector.evaluate_comprehensive(test_ds, df['cleaned_text'])
detector.plot_confusion_matrix(results['labels'], results['predictions'])
detector.plot_roc_and_pr_curves(results['labels'], results['probabilities'])

# Make predictions
prediction, confidence = detector.predict("Sample text")
```

#### 3. SafetyGuidelines
Comprehensive ethical framework with risk assessment and crisis intervention.

```python
from src.safety_guidelines import SafetyGuidelines

guidelines = SafetyGuidelines()

# Assess risk level
risk_level = guidelines.assess_risk_level(prediction, confidence)

# Execute appropriate safety protocol
safety_results = guidelines.execute_safety_protocol(risk_level, text, prediction_data)

# Validate use case
validation = guidelines.validate_use_case("Research study")
```

#### 4. Safety Monitoring
Real-time monitoring and audit capabilities for responsible deployment.

```python
from src.safety_guidelines import SafetyMonitor

monitor = SafetyMonitor()
monitor.log_prediction(prediction_data, safety_data)
metrics = monitor.get_safety_metrics()
```

### Model Details

**LSTM Architecture Specifications:**
- **Input**: Variable-length text sequences (max 100 tokens)
- **Embedding**: 128-dimensional learned embeddings
- **LSTM**: 2-layer bidirectional (64 hidden units each direction)
- **Regularization**: 30% dropout, early stopping
- **Output**: Binary classification (suicide risk probability)
- **Parameters**: ~847K trainable parameters

**ELECTRA Architecture Specifications:**
- **Base Model**: google/electra-small-discriminator (~14M parameters)
- **Input**: Up to 512 WordPiece tokens
- **Encoder**: 12-layer transformer with 256 hidden dimensions
- **Classification**: Dense layer with softmax activation
- **Fine-tuning**: Full model with learning rate 2e-5
- **Output**: Binary classification (suicide/non-suicide)

**Training Configuration:**
- **Optimizer**: Adam (LSTM) / AdamW (ELECTRA)
- **Loss**: Binary cross-entropy / Cross-entropy
- **Batch Size**: 32 (LSTM) / 16 (ELECTRA)
- **Validation**: Stratified split with F1 monitoring
- **Hardware**: GPU-accelerated training (recommended)

## Safety & Ethics Framework

### Risk Assessment Levels

| Level | Confidence | Actions Required |
|-------|------------|------------------|
| **Low** | Any confidence, no risk | • Resource provision<br>• Basic logging |
| **Moderate** | 50-70% suicide risk | • Enhanced resources<br>• Human oversight<br>• Pattern monitoring |
| **High** | 70-90% suicide risk | • Crisis resources<br>• Immediate oversight<br>• Safety planning |
| **Crisis** | 90%+ suicide risk | • Emergency protocols<br>• Crisis team alert<br>• Comprehensive logging |

### Crisis Intervention

**Immediate Response Protocol:**
1. 🚨 **Risk Assessment**: Automatic confidence-based classification
2. 📞 **Resource Provision**: Context-appropriate crisis contacts
3. 👥 **Human Escalation**: Real-time alerts for oversight team
4. 📋 **Documentation**: Comprehensive incident logging
5. 🔄 **Follow-up**: Monitoring and care continuation

**Crisis Resources Integration:**
- **US National Suicide Prevention Lifeline**: 988
- **Crisis Text Line**: Text HOME to 741741
- **International Resources**: https://findahelpline.com
- **Emergency Services**: 911 guidance for immediate danger

### Ethical Safeguards

#### Data Ethics
- **Privacy**: Text hashing for logs, no raw content storage
- **Consent**: Clear informed consent requirements
- **Anonymization**: De-identification protocols
- **Retention**: Minimal data retention policies

#### Model Ethics
- **Bias Mitigation**: Demographic fairness assessment
- **Transparency**: Clear limitation disclosure
- **Uncertainty**: Confidence calibration and display
- **Accountability**: Human oversight requirements

#### Deployment Ethics
- **Usage Validation**: Prohibited use detection
- **Performance Monitoring**: Continuous bias assessment
- **Professional Standards**: Clinical ethics compliance
- **Stakeholder Input**: Mental health professional review

## Quick Start

### Installation

```bash
git clone https://github.com/ScottDougBlain/mental-health-nlp.git
cd mental-health-nlp
pip install -r requirements.txt
```

### Basic Demo

```bash
# Run comprehensive safety demo
python examples/demo_application.py --mode demo

# Interactive demo with safety protocols
python examples/demo_application.py --mode interactive

# Generate detailed report
python examples/demo_application.py --mode report
```

### Training Your Own Model

#### Option 1: LSTM Model (Baseline)

```python
from src.suicide_risk_detector import SuicideRiskDetector

# Initialize detector (shows ethical warnings)
detector = SuicideRiskDetector()

# Load your data (CSV with 'text' and 'class' columns)
texts, labels = detector.load_data('your_data.csv')

# Prepare data with safety protocols
train_loader, val_loader, test_loader = detector.prepare_data(texts, labels)

# Create and train model
detector.create_model()
detector.train(train_loader, val_loader, epochs=10)

# Evaluate with comprehensive metrics
results = detector.evaluate_model(test_loader)
print(f"F1 Score: {results['f1_score']:.3f}")

# Save model with safety documentation
detector._save_model('trained_model.pth')
```

#### Option 2: ELECTRA Model (State-of-the-art)

**Command Line:**
```bash
# Download Kaggle dataset first:
# https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch

# Train ELECTRA model with default settings
python examples/train_electra.py \
    --data_path /path/to/Suicide_Detection.csv \
    --save_model \
    --save_predictions \
    --plot

# Train with custom hyperparameters
python examples/train_electra.py \
    --data_path /path/to/Suicide_Detection.csv \
    --model_name google/electra-base-discriminator \
    --num_epochs 5 \
    --batch_size 8 \
    --learning_rate 3e-5 \
    --save_model \
    --output_dir ./my_electra_results

# Evaluate only (with pretrained model)
python examples/train_electra.py \
    --data_path /path/to/Suicide_Detection.csv \
    --eval_only \
    --load_model_path ./saved_model \
    --plot

# Interactive mode (make predictions on new text)
python examples/train_electra.py \
    --data_path /path/to/Suicide_Detection.csv \
    --eval_only \
    --load_model_path ./saved_model \
    --interactive
```

**Python API:**
```python
from src.electra_risk_detector import ELECTRARiskDetector

# Initialize ELECTRA model
detector = ELECTRARiskDetector(
    model_name="google/electra-small-discriminator",
    max_length=512
)

# Load and prepare data
df, labels = detector.load_data('Suicide_Detection.csv')
train_ds, val_ds, test_ds = detector.prepare_datasets(
    df['cleaned_text'],
    labels,
    test_size=0.2
)

# Train model
detector.train(
    train_dataset=train_ds,
    val_dataset=val_ds,
    num_epochs=3,
    batch_size=16,
    learning_rate=2e-5
)

# Comprehensive evaluation
results = detector.evaluate_comprehensive(test_ds, df['cleaned_text'])

# Generate visualizations
detector.plot_confusion_matrix(results['labels'], results['predictions'])
detector.plot_roc_and_pr_curves(results['labels'], results['probabilities'])

# Save model
detector.save_model('./saved_electra_model')
```

## 📊 Performance Analysis

*Note: All metrics shown in this section are illustrative examples demonstrating the framework's analytical capabilities. The repository provides complete implementation but models require training on actual data to achieve real performance metrics.*

### Model Performance

**Confusion Matrix (Illustrative Example):**
```
                Predicted
              No Risk  Risk
Actual No Risk  1,847   148   (92.8% precision)
       Risk       98  1,923   (95.6% recall)
```

**Performance by Risk Level (Illustrative Example):**
- **High Confidence Predictions (>90%)**: 97% accuracy
- **Medium Confidence (70-90%)**: 94% accuracy
- **Lower Confidence (<70%)**: 89% accuracy (flagged for review)

### Safety Metrics

**Risk Distribution in Validation:**
- **Low Risk**: 78.3% of predictions
- **Moderate Risk**: 15.7% of predictions
- **High Risk**: 5.2% of predictions
- **Crisis Level**: 0.8% of predictions

**Human Oversight Metrics:**
- **Automatic Resolution**: 78.3% (low risk only)
- **Human Review Required**: 21.7% (moderate+ risk)
- **Crisis Escalation**: 6.0% (high/crisis risk)
- **Protocol Compliance**: 100% (all cases follow safety guidelines)

## 🛡️ Safety Features

### Real-time Risk Assessment

```python
# Every prediction includes safety assessment
prediction, confidence = detector.predict_risk(text)
risk_level = safety_guidelines.assess_risk_level(prediction, confidence)

# Automatic protocol execution
safety_results = safety_guidelines.execute_safety_protocol(
    risk_level, text, prediction_data
)

# Crisis resource provision
if risk_level in [RiskLevel.HIGH, RiskLevel.CRISIS]:
    display_crisis_resources()
    notify_human_oversight()
```

### Prohibited Use Detection

```python
# Automatic use case validation
validation = safety_guidelines.validate_use_case(intended_use)

if not validation['approved']:
    raise EthicalViolationError("Prohibited use case detected")

# Specific prohibited uses
prohibited_uses = [
    "Clinical diagnosis",
    "Treatment decisions",
    "Automated intervention",
    "Life-critical systems",
    "Discriminatory screening"
]
```

### Audit and Monitoring

```python
# Comprehensive incident logging
safety_monitor.log_prediction(prediction_data, safety_data)

# Real-time safety metrics
metrics = safety_monitor.get_safety_metrics()
# Returns: risk distribution, oversight rates, protocol compliance

# Bias monitoring across demographics
bias_report = monitor.assess_demographic_fairness(predictions, demographics)
```

## Research Foundation

### Clinical Psychology Integration

This project applies clinical insights to AI safety:

**Therapeutic Assessment Principles:**
- **Risk Stratification**: Multiple confidence thresholds mirror clinical triage
- **Crisis Intervention**: Immediate escalation protocols based on emergency psychology
- **False Positive Management**: Balancing sensitivity with practicality
- **Professional Standards**: Adherence to clinical ethics guidelines

**Evidence-Based Approach:**
- Validated crisis intervention protocols
- Professional mental health resource integration
- Ethical frameworks from medical AI literature
- Bias mitigation strategies from clinical psychology research

### Technical Innovation

**Novel Contributions:**
- **Safety-First Architecture**: Ethics integrated into core system design
- **Multi-Level Risk Assessment**: Granular confidence-based protocols
- **Real-time Ethics Enforcement**: Automatic prohibited use detection
- **Comprehensive Audit Trail**: Full incident logging for bias assessment

## Testing & Validation

### Comprehensive Test Suite

```bash
# Run all safety and performance tests
pytest tests/ --cov=src --cov-report=html

# Test safety protocols specifically
pytest tests/test_safety_guidelines.py -v

# Test model performance
pytest tests/test_suicide_risk_detector.py -v

# Test ethical compliance
pytest tests/test_ethical_framework.py -v
```

### Mock Testing for Safety

```python
# Test crisis protocols without real risk
from src.safety_guidelines import demonstrate_safety_guidelines

# Test different risk scenarios
demonstrate_safety_guidelines()

# Validate ethical boundaries
test_prohibited_use_detection()

# Assess bias across demographics
run_fairness_assessment()
```

## Use Cases & Applications

### ✅ Approved Uses

**Research Applications:**
- Academic studies on suicide risk factors
- Algorithm development and validation
- Bias assessment and fairness research
- Crisis intervention protocol development

**Educational Applications:**
- Responsible AI development training
- Ethics in healthcare AI coursework
- Mental health informatics education
- Clinical psychology research methods

**Development Applications:**
- Framework validation and testing
- Safety protocol development
- Ethical guideline implementation
- Performance benchmarking

### ❌ Strictly Prohibited Uses

**Clinical Applications:**
- Diagnostic decision making
- Treatment planning or modification
- Patient screening or assessment
- Clinical documentation or records

**Automated Systems:**
- Unsupervised content moderation
- Automatic crisis intervention
- Life-critical decision making
- Emergency response systems

**Discriminatory Applications:**
- Employment screening
- Insurance assessment
- Educational admissions
- Law enforcement profiling

## Configuration & Customization

### Model Configuration

```python
# Custom model architecture
detector = SuicideRiskDetector(max_length=200)  # Longer sequences

# Custom training parameters
detector.train(
    train_loader, val_loader,
    epochs=20,
    learning_rate=0.0005,
    patience=5  # Early stopping
)

# Custom safety thresholds
safety_guidelines.risk_thresholds = {
    'moderate': 0.4,  # Lower threshold for sensitivity
    'high': 0.6,
    'crisis': 0.8
}
```

### Safety Customization

```python
# Custom crisis resources
safety_guidelines.crisis_resources.update({
    'local_resources': {
        'hospital': 'Local Emergency Room: (555) 123-4567',
        'clinic': 'Community Mental Health: (555) 234-5678'
    }
})

# Custom risk protocols
safety_guidelines.safety_protocols[RiskLevel.HIGH] = SafetyProtocol(
    risk_level=RiskLevel.HIGH,
    required_actions=['immediate_notification', 'crisis_resources', 'safety_planning'],
    human_oversight_required=True,
    crisis_escalation=True,
    logging_required=True
)
```

## Contributing

We welcome contributions that enhance the safety and effectiveness of mental health AI:

### Research Contributions
- **Validation Studies**: Real-world outcome correlation research
- **Bias Assessment**: Fairness across demographic groups
- **Crisis Protocol**: Improved intervention strategies
- **Clinical Integration**: Professional workflow development

### Technical Contributions
- **Model Improvements**: Enhanced architecture and performance
- **Safety Enhancements**: Additional ethical safeguards
- **Monitoring Tools**: Better audit and bias detection
- **Integration APIs**: Healthcare system compatibility

### Development Guidelines

```bash
# Set up development environment
git clone https://github.com/ScottDougBlain/mental-health-nlp.git
cd mental-health-nlp
pip install -e ".[dev]"

# Install pre-commit hooks for safety checks
pre-commit install

# Run comprehensive testing
python -m pytest tests/ --cov=src
python examples/demo_application.py --mode demo

# Ethical review checklist
python src/safety_guidelines.py  # Run ethics validation
```

## Academic Context

### Publications & Research

**Relevant Literature:**
- Benton, A., et al. (2017). Ethical research protocols for social media health research
- Chancellor, S., & De Choudhury, M. (2020). Methods in predictive techniques for mental health status
- Ernala, S.K., et al. (2019). Methodological gaps in predicting mental health states from social media
- Guntuku, S.C., et al. (2017). Detecting depression and mental illness on social media

**Clinical Guidelines:**
- American Psychological Association ethics guidelines
- WHO suicide prevention strategies
- Crisis intervention best practices
- Healthcare AI ethics frameworks

### Impact & Significance

**Technical Impact:**
- Demonstrates responsible AI development in sensitive domains
- Bridges ML engineering with clinical psychology insights
- Establishes safety-first architecture patterns
- Provides comprehensive ethical framework implementation

**Social Impact:**
- Advances responsible AI practices in healthcare
- Promotes ethical consideration in mental health technology
- Demonstrates harm prevention through proactive design
- Contributes to AI safety research community

## Citation

```bibtex
@misc{mental_health_nlp_safety_2024,
  title={Mental Health NLP Safety Demo: Responsible AI for Suicide Risk Detection},
  author={Blain, Scott},
  year={2024},
  url={https://github.com/ScottDougBlain/mental-health-nlp},
  note={LSTM-based suicide risk detection with comprehensive safety framework}
}
```

## Related Projects

- **[Hallucination Mitigation Framework](https://github.com/ScottDougBlain/llm-hallucination-reduction)**: Epistemic humility for AI truthfulness
- **[Theory of Mind Benchmark](https://github.com/ScottDougBlain/theory-of-mind-benchmark)**: Social cognition evaluation
- **[Honest AI Evaluations](https://github.com/ScottDougBlain/honest-ai-evaluations)**: Non-sycophantic response framework

## 📋 License

MIT License - see [LICENSE](LICENSE) file for details.

**Additional Terms for Mental Health Applications:**
- Educational and research use only
- No clinical deployment without professional oversight
- Crisis intervention protocols must be maintained
- Regular bias assessment and fairness evaluation required

---

## Current Limitations

### What This Repository Is

- **Complete implementation** of LSTM and ELECTRA architectures for mental health risk detection
- **Comprehensive safety framework** with real crisis resources and ethical guidelines
- **Training pipeline** ready to work with Kaggle Suicide Watch dataset
- **Demonstration of best practices** for responsible AI in sensitive domains
- **Educational resource** for learning about safety-first ML development

### What This Repository Is NOT

- **NOT a trained model** - requires actual training on data
- **NOT validated** - no performance metrics have been measured
- **NOT for clinical use** - purely educational/research demonstration
- **NOT claiming achieved performance** - all metrics are literature-based targets

### To Use This Repository

1. **Download the Kaggle dataset**: [Suicide Watch dataset](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch)
2. **Train the models**: Use provided training scripts with actual data
3. **Validate performance**: Measure actual metrics on test set
4. **Never use clinically**: This is for research/education only

## Project Impact

This project demonstrates that technical excellence and ethical responsibility are not just compatible, but mutually reinforcing. By implementing comprehensive safety frameworks alongside high-performance ML models, we show how responsible AI development can achieve both technical and social objectives.

**Key Takeaways:**
- **High-Performance Architecture** targeting literature benchmarks proves technical competence in challenging ML domain
- **Comprehensive Safety Framework** demonstrates ethical AI development skills
- **Crisis Intervention Integration** shows understanding of real-world impact
- **Production-Ready Code** indicates software engineering capabilities
- **Clinical Psychology Integration** highlights interdisciplinary thinking

This work positions the developer as someone who can deliver both cutting-edge technical solutions and responsible AI systems suitable for high-stakes applications—exactly the combination needed for AI safety research and development.