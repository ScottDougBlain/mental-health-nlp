"""
Safety Guidelines and Ethical Framework for Mental Health NLP

This module provides comprehensive safety guidelines, ethical considerations,
and responsible AI practices for mental health applications.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk levels for mental health content."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRISIS = "crisis"


class UseCase(Enum):
    """Approved use cases for the mental health NLP system."""
    RESEARCH = "research"
    EDUCATION = "education"
    DEVELOPMENT = "development"
    DEMONSTRATION = "demonstration"


class ProhibitedUse(Enum):
    """Explicitly prohibited uses."""
    CLINICAL_DIAGNOSIS = "clinical_diagnosis"
    TREATMENT_DECISIONS = "treatment_decisions"
    AUTOMATED_INTERVENTION = "automated_intervention"
    UNSUPERVISED_MODERATION = "unsupervised_moderation"
    LIFE_CRITICAL_SYSTEMS = "life_critical_systems"
    DISCRIMINATORY_SCREENING = "discriminatory_screening"


@dataclass
class SafetyProtocol:
    """Safety protocol for different risk levels."""
    risk_level: RiskLevel
    required_actions: List[str]
    human_oversight_required: bool
    crisis_escalation: bool
    logging_required: bool
    additional_warnings: List[str]


class SafetyGuidelines:
    """
    Comprehensive safety guidelines for mental health NLP applications.

    This class provides:
    - Ethical use guidelines
    - Risk assessment protocols
    - Crisis intervention procedures
    - Privacy and consent requirements
    - Monitoring and audit recommendations
    """

    def __init__(self):
        self.safety_protocols = self._define_safety_protocols()
        self.crisis_resources = self._load_crisis_resources()

    def _define_safety_protocols(self) -> Dict[RiskLevel, SafetyProtocol]:
        """Define safety protocols for each risk level."""
        return {
            RiskLevel.LOW: SafetyProtocol(
                risk_level=RiskLevel.LOW,
                required_actions=[
                    "Log prediction for monitoring",
                    "Provide general mental health resources"
                ],
                human_oversight_required=False,
                crisis_escalation=False,
                logging_required=True,
                additional_warnings=[]
            ),

            RiskLevel.MODERATE: SafetyProtocol(
                risk_level=RiskLevel.MODERATE,
                required_actions=[
                    "Enhanced logging with context",
                    "Provide targeted mental health resources",
                    "Recommend professional consultation",
                    "Monitor for pattern escalation"
                ],
                human_oversight_required=True,
                crisis_escalation=False,
                logging_required=True,
                additional_warnings=[
                    "Consider seeking professional guidance",
                    "Monitor mental health status"
                ]
            ),

            RiskLevel.HIGH: SafetyProtocol(
                risk_level=RiskLevel.HIGH,
                required_actions=[
                    "Immediate human oversight notification",
                    "Comprehensive logging",
                    "Provide crisis resources prominently",
                    "Recommend immediate professional help",
                    "Consider safety planning"
                ],
                human_oversight_required=True,
                crisis_escalation=True,
                logging_required=True,
                additional_warnings=[
                    "Seek immediate professional help",
                    "Contact mental health crisis services",
                    "Consider emergency services if in immediate danger"
                ]
            ),

            RiskLevel.CRISIS: SafetyProtocol(
                risk_level=RiskLevel.CRISIS,
                required_actions=[
                    "IMMEDIATE crisis protocol activation",
                    "Emergency contact notification",
                    "Comprehensive incident logging",
                    "Provide emergency resources",
                    "Escalate to crisis intervention team"
                ],
                human_oversight_required=True,
                crisis_escalation=True,
                logging_required=True,
                additional_warnings=[
                    "IMMEDIATE ACTION REQUIRED",
                    "Contact emergency services (911)",
                    "Go to nearest emergency room",
                    "Call crisis hotline immediately"
                ]
            )
        }

    def _load_crisis_resources(self) -> Dict:
        """Load comprehensive crisis intervention resources."""
        return {
            "emergency": {
                "us": "911",
                "international": "Check local emergency numbers"
            },
            "suicide_prevention": {
                "us_lifeline": {
                    "number": "988",
                    "description": "National Suicide Prevention Lifeline",
                    "website": "https://suicidepreventionlifeline.org/",
                    "chat": "Available online"
                },
                "crisis_text": {
                    "number": "741741",
                    "text": "HOME",
                    "description": "Crisis Text Line"
                },
                "international": {
                    "website": "https://findahelpline.com",
                    "description": "International suicide prevention resources"
                }
            },
            "mental_health": {
                "samhsa": {
                    "number": "1-800-662-4357",
                    "description": "SAMHSA National Helpline",
                    "website": "https://www.samhsa.gov/find-help/national-helpline"
                },
                "nami": {
                    "website": "https://www.nami.org/help",
                    "description": "National Alliance on Mental Illness"
                }
            },
            "specialized": {
                "veterans": {
                    "number": "1-800-273-8255",
                    "press": "1",
                    "description": "Veterans Crisis Line"
                },
                "lgbtq": {
                    "number": "1-866-488-7386",
                    "description": "The Trevor Project (LGBTQ+ youth)"
                },
                "disaster": {
                    "number": "1-800-985-5990",
                    "description": "Disaster Distress Helpline"
                }
            }
        }

    def assess_risk_level(self, prediction: int, confidence: float, context: Optional[Dict] = None) -> RiskLevel:
        """
        Assess risk level based on model prediction and confidence.

        Args:
            prediction: Model prediction (0=no risk, 1=risk)
            confidence: Prediction confidence score
            context: Additional context information

        Returns:
            Assessed risk level
        """
        if prediction == 0:
            return RiskLevel.LOW

        # Risk prediction (prediction == 1)
        if confidence >= 0.9:
            return RiskLevel.CRISIS
        elif confidence >= 0.7:
            return RiskLevel.HIGH
        elif confidence >= 0.5:
            return RiskLevel.MODERATE
        else:
            return RiskLevel.LOW

    def get_safety_protocol(self, risk_level: RiskLevel) -> SafetyProtocol:
        """Get safety protocol for a given risk level."""
        return self.safety_protocols[risk_level]

    def execute_safety_protocol(self, risk_level: RiskLevel, text: str, prediction_data: Dict) -> Dict:
        """
        Execute appropriate safety protocol based on risk level.

        Args:
            risk_level: Assessed risk level
            text: Original text analyzed
            prediction_data: Model prediction data

        Returns:
            Protocol execution results
        """
        protocol = self.get_safety_protocol(risk_level)

        execution_results = {
            "risk_level": risk_level.value,
            "protocol_executed": True,
            "actions_taken": [],
            "warnings_issued": [],
            "resources_provided": [],
            "escalation_required": protocol.crisis_escalation
        }

        # Execute required actions
        for action in protocol.required_actions:
            if "logging" in action.lower():
                self._log_incident(risk_level, text, prediction_data)
                execution_results["actions_taken"].append("Incident logged")

            elif "oversight" in action.lower():
                self._notify_human_oversight(risk_level, text, prediction_data)
                execution_results["actions_taken"].append("Human oversight notified")

            elif "resources" in action.lower():
                resources = self._provide_resources(risk_level)
                execution_results["resources_provided"].extend(resources)
                execution_results["actions_taken"].append("Resources provided")

        # Issue warnings
        execution_results["warnings_issued"] = protocol.additional_warnings

        return execution_results

    def _log_incident(self, risk_level: RiskLevel, text: str, prediction_data: Dict) -> None:
        """Log safety incident for monitoring and audit."""
        log_entry = {
            "risk_level": risk_level.value,
            "text_hash": hash(text),  # Don't log actual text for privacy
            "text_length": len(text),
            "prediction": prediction_data.get("prediction"),
            "confidence": prediction_data.get("confidence"),
            "timestamp": prediction_data.get("timestamp")
        }

        if risk_level in [RiskLevel.HIGH, RiskLevel.CRISIS]:
            logger.critical(f"HIGH RISK INCIDENT: {log_entry}")
        elif risk_level == RiskLevel.MODERATE:
            logger.warning(f"MODERATE RISK INCIDENT: {log_entry}")
        else:
            logger.info(f"LOW RISK INCIDENT: {log_entry}")

    def _notify_human_oversight(self, risk_level: RiskLevel, text: str, prediction_data: Dict) -> None:
        """Notify human oversight team of high-risk predictions."""
        if risk_level in [RiskLevel.HIGH, RiskLevel.CRISIS]:
            # In a real system, this would send alerts to monitoring systems
            logger.critical("HUMAN OVERSIGHT REQUIRED - High risk prediction detected")

            # Could integrate with:
            # - Slack/Teams notifications
            # - PagerDuty alerts
            # - Email notifications
            # - Dashboard alerts

    def _provide_resources(self, risk_level: RiskLevel) -> List[str]:
        """Provide appropriate resources based on risk level."""
        resources = []

        if risk_level == RiskLevel.CRISIS:
            resources.extend([
                "EMERGENCY: Call 911 if in immediate danger",
                "National Suicide Prevention Lifeline: 988",
                "Crisis Text Line: Text HOME to 741741",
                "Go to nearest emergency room"
            ])
        elif risk_level == RiskLevel.HIGH:
            resources.extend([
                "National Suicide Prevention Lifeline: 988",
                "Crisis Text Line: Text HOME to 741741",
                "SAMHSA National Helpline: 1-800-662-4357",
                "Consider contacting emergency services if in immediate danger"
            ])
        elif risk_level == RiskLevel.MODERATE:
            resources.extend([
                "SAMHSA National Helpline: 1-800-662-4357",
                "NAMI (National Alliance on Mental Illness): nami.org/help",
                "Consider speaking with a mental health professional"
            ])
        else:  # LOW
            resources.extend([
                "Mental Health America: mhanational.org",
                "NAMI: nami.org/help",
                "Psychology Today therapist finder: psychologytoday.com"
            ])

        return resources

    def validate_use_case(self, intended_use: str) -> Dict:
        """
        Validate that the intended use case is appropriate.

        Args:
            intended_use: Description of intended use

        Returns:
            Validation results
        """
        validation = {
            "approved": False,
            "use_case": None,
            "warnings": [],
            "requirements": []
        }

        # Check for prohibited uses
        prohibited_keywords = {
            "clinical": ProhibitedUse.CLINICAL_DIAGNOSIS,
            "diagnosis": ProhibitedUse.CLINICAL_DIAGNOSIS,
            "treatment": ProhibitedUse.TREATMENT_DECISIONS,
            "automated": ProhibitedUse.AUTOMATED_INTERVENTION,
            "unsupervised": ProhibitedUse.UNSUPERVISED_MODERATION,
            "life-critical": ProhibitedUse.LIFE_CRITICAL_SYSTEMS,
            "screening": ProhibitedUse.DISCRIMINATORY_SCREENING
        }

        intended_lower = intended_use.lower()

        for keyword, prohibited_use in prohibited_keywords.items():
            if keyword in intended_lower:
                validation["warnings"].append(f"PROHIBITED USE DETECTED: {prohibited_use.value}")
                validation["approved"] = False
                return validation

        # Check for approved uses
        approved_keywords = {
            "research": UseCase.RESEARCH,
            "education": UseCase.EDUCATION,
            "development": UseCase.DEVELOPMENT,
            "demo": UseCase.DEMONSTRATION,
            "demonstration": UseCase.DEMONSTRATION
        }

        for keyword, use_case in approved_keywords.items():
            if keyword in intended_lower:
                validation["approved"] = True
                validation["use_case"] = use_case
                validation["requirements"] = self._get_use_case_requirements(use_case)
                break

        if not validation["approved"]:
            validation["warnings"].append("Use case not explicitly approved - requires review")

        return validation

    def _get_use_case_requirements(self, use_case: UseCase) -> List[str]:
        """Get requirements for specific use cases."""
        requirements = {
            UseCase.RESEARCH: [
                "IRB approval required",
                "Informed consent from participants",
                "Data anonymization protocols",
                "Secure data storage",
                "Regular ethical review"
            ],
            UseCase.EDUCATION: [
                "Clear educational purpose",
                "Instructor supervision",
                "No real mental health data",
                "Ethical guidelines training",
                "Proper disclaimers"
            ],
            UseCase.DEVELOPMENT: [
                "Development environment only",
                "No production deployment",
                "Synthetic or anonymized data",
                "Code review requirements",
                "Security protocols"
            ],
            UseCase.DEMONSTRATION: [
                "Demo purposes only",
                "Clear limitations disclosure",
                "No real-world decisions",
                "Audience appropriate warnings",
                "Professional supervision"
            ]
        }

        return requirements.get(use_case, [])

    def generate_ethics_checklist(self) -> Dict:
        """Generate comprehensive ethics checklist for mental health NLP."""
        return {
            "data_ethics": {
                "privacy": [
                    "Data anonymization implemented",
                    "Consent obtained for data use",
                    "Data retention policies defined",
                    "Access controls in place"
                ],
                "bias": [
                    "Dataset bias assessment completed",
                    "Fairness across demographics evaluated",
                    "Mitigation strategies implemented",
                    "Regular bias monitoring"
                ]
            },
            "model_ethics": {
                "transparency": [
                    "Model limitations clearly documented",
                    "Uncertainty quantification included",
                    "Decision boundaries explained",
                    "Performance metrics disclosed"
                ],
                "accountability": [
                    "Human oversight requirements defined",
                    "Error handling procedures established",
                    "Audit trail maintained",
                    "Responsibility assignment clear"
                ]
            },
            "deployment_ethics": {
                "safety": [
                    "Safety protocols implemented",
                    "Crisis intervention procedures ready",
                    "Risk assessment protocols active",
                    "Emergency escalation paths defined"
                ],
                "impact": [
                    "Potential harm assessment completed",
                    "Benefit-risk analysis documented",
                    "Stakeholder impact considered",
                    "Monitoring systems in place"
                ]
            },
            "ongoing_ethics": {
                "monitoring": [
                    "Performance monitoring active",
                    "Bias detection ongoing",
                    "User feedback collection",
                    "Regular ethics review scheduled"
                ],
                "improvement": [
                    "Continuous improvement process",
                    "Ethics training for team",
                    "External ethics consultation",
                    "Community engagement"
                ]
            }
        }

    def display_usage_agreement(self) -> str:
        """Display comprehensive usage agreement."""
        agreement = """
        MENTAL HEALTH NLP SYSTEM - USAGE AGREEMENT
        ==========================================

        By using this mental health NLP system, you acknowledge and agree to the following:

        APPROVED USES ONLY:
        ✓ Research with appropriate ethical approval
        ✓ Educational purposes with supervision
        ✓ Development and testing environments
        ✓ Demonstration with clear limitations

        STRICTLY PROHIBITED USES:
        ✗ Clinical diagnosis or treatment decisions
        ✗ Automated intervention without human oversight
        ✗ Life-critical or emergency response systems
        ✗ Discriminatory screening or profiling
        ✗ Unsupervised content moderation

        SAFETY REQUIREMENTS:
        • Human oversight for all high-risk predictions
        • Crisis intervention protocols must be in place
        • Regular monitoring and audit of system performance
        • Immediate escalation of crisis-level predictions

        ETHICAL OBLIGATIONS:
        • Respect user privacy and confidentiality
        • Provide appropriate crisis resources
        • Acknowledge system limitations clearly
        • Obtain informed consent where applicable

        CRISIS RESOURCES:
        • US National Suicide Prevention Lifeline: 988
        • Crisis Text Line: Text HOME to 741741
        • International resources: findahelpline.com

        By proceeding, you confirm that you understand these requirements and will use
        this system responsibly and ethically.

        Do you agree to these terms? (yes/no):
        """

        return agreement


class SafetyMonitor:
    """Monitor and track safety metrics for mental health NLP systems."""

    def __init__(self):
        self.incident_log = []
        self.performance_metrics = {}

    def log_prediction(self, prediction_data: Dict, safety_data: Dict) -> None:
        """Log prediction with safety information."""
        log_entry = {
            "timestamp": prediction_data.get("timestamp"),
            "risk_level": safety_data.get("risk_level"),
            "confidence": prediction_data.get("confidence"),
            "protocol_executed": safety_data.get("protocol_executed"),
            "human_oversight": safety_data.get("escalation_required")
        }

        self.incident_log.append(log_entry)

    def get_safety_metrics(self) -> Dict:
        """Calculate safety metrics from logged incidents."""
        if not self.incident_log:
            return {"error": "No incidents logged"}

        total_incidents = len(self.incident_log)

        # Risk level distribution
        risk_levels = [entry["risk_level"] for entry in self.incident_log]
        risk_distribution = {level: risk_levels.count(level) for level in set(risk_levels)}

        # High-risk incidents
        high_risk_count = sum(1 for entry in self.incident_log
                             if entry["risk_level"] in ["high", "crisis"])

        # Human oversight rate
        oversight_count = sum(1 for entry in self.incident_log
                             if entry.get("human_oversight", False))

        return {
            "total_incidents": total_incidents,
            "risk_distribution": risk_distribution,
            "high_risk_rate": high_risk_count / total_incidents,
            "human_oversight_rate": oversight_count / total_incidents,
            "safety_protocol_compliance": sum(1 for entry in self.incident_log
                                            if entry.get("protocol_executed", False)) / total_incidents
        }


# Example usage and testing
def demonstrate_safety_guidelines():
    """Demonstrate the safety guidelines system."""
    print("Mental Health NLP Safety Guidelines Demo")
    print("=" * 50)

    guidelines = SafetyGuidelines()

    # Test different risk levels
    test_cases = [
        (1, 0.95, "crisis level"),
        (1, 0.75, "high risk"),
        (1, 0.55, "moderate risk"),
        (0, 0.8, "low risk")
    ]

    for prediction, confidence, description in test_cases:
        print(f"\nTesting {description} (prediction={prediction}, confidence={confidence:.2f})")

        risk_level = guidelines.assess_risk_level(prediction, confidence)
        protocol = guidelines.get_safety_protocol(risk_level)

        print(f"Risk Level: {risk_level.value}")
        print(f"Human Oversight Required: {protocol.human_oversight_required}")
        print(f"Crisis Escalation: {protocol.crisis_escalation}")
        print(f"Required Actions: {protocol.required_actions}")

    # Test use case validation
    print("\n" + "=" * 50)
    print("USE CASE VALIDATION TESTS")
    print("=" * 50)

    test_uses = [
        "Research study on mental health",
        "Clinical diagnosis tool",
        "Educational demonstration",
        "Automated treatment system"
    ]

    for use_case in test_uses:
        validation = guidelines.validate_use_case(use_case)
        print(f"\nUse case: {use_case}")
        print(f"Approved: {validation['approved']}")
        if validation['warnings']:
            print(f"Warnings: {validation['warnings']}")


if __name__ == "__main__":
    demonstrate_safety_guidelines()