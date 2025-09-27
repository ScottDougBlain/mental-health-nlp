"""
Mental Health NLP Safety Demo

A responsible implementation of LSTM-based suicide risk detection with comprehensive
safety frameworks. Demonstrates both ML engineering capabilities and responsible AI
development in sensitive mental health domains.

IMPORTANT: For research and educational purposes only. Not for clinical use.
"""

from .suicide_risk_detector import (
    SuicideRiskDetector,
    SuicideRiskLSTM,
    TextPreprocessor,
    SuicideDataset,
    EthicalWarning,
    CRISIS_RESOURCES
)

from .safety_guidelines import (
    SafetyGuidelines,
    SafetyMonitor,
    SafetyProtocol,
    RiskLevel,
    UseCase,
    ProhibitedUse
)

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = [
    # Core detection classes
    "SuicideRiskDetector",
    "SuicideRiskLSTM",
    "TextPreprocessor",
    "SuicideDataset",

    # Safety and ethics
    "SafetyGuidelines",
    "SafetyMonitor",
    "SafetyProtocol",
    "EthicalWarning",

    # Enums
    "RiskLevel",
    "UseCase",
    "ProhibitedUse",

    # Resources
    "CRISIS_RESOURCES"
]

# Display usage warning when module is imported
EthicalWarning.display_usage_warning()