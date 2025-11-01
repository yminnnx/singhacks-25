"""
Regulatory Rules Engine for AML Monitoring
Configurable rules system for detecting AML violations based on regulatory requirements
"""

from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import re
import json
import logging

class RuleCategory(Enum):
    TRANSACTION_MONITORING = "Transaction Monitoring"
    CUSTOMER_SCREENING = "Customer Screening"
    SANCTIONS_COMPLIANCE = "Sanctions Compliance"
    KYC_COMPLIANCE = "KYC Compliance"
    REPORTING_REQUIREMENTS = "Reporting Requirements"

class RuleSeverity(Enum):
    INFO = "Info"
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"

@dataclass
class RuleViolation:
    rule_id: str
    rule_name: str
    severity: RuleSeverity
    category: RuleCategory
    description: str
    transaction_id: str
    customer_id: str
    violation_details: Dict[str, Any]
    timestamp: datetime
    recommended_action: str

@dataclass
class ComplianceRule:
    rule_id: str
    name: str
    description: str
    category: RuleCategory
    severity: RuleSeverity
    is_active: bool
    effective_date: datetime
    jurisdiction: List[str]  # e.g., ['SG', 'HK', 'CH']
    regulator: List[str]    # e.g., ['MAS', 'HKMA', 'FINMA']
    parameters: Dict[str, Any]
    condition_function: str  # Function name to evaluate
    created_date: datetime
    last_updated: datetime
    version: str

class RegulatoryRulesEngine:
    """
    Configurable rules engine for AML compliance checking
    """
    
    def __init__(self):
        self.rules: Dict[str, ComplianceRule] = {}
        self.rule_violations: List[RuleViolation] = []
        self.setup_logging()
        self.load_default_rules()
    
    def setup_logging(self):
        """Setup logging for audit trail"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def load_default_rules(self):
        """Load default regulatory rules for MAS, HKMA, and FINMA"""
        
        # MAS (Singapore) Rules
        self.add_rule(ComplianceRule(
            rule_id="MAS-TM-001",
            name="Large Cash Transaction Reporting",
            description="Cash transactions above SGD 20,000 require enhanced monitoring",
            category=RuleCategory.TRANSACTION_MONITORING,
            severity=RuleSeverity.HIGH,
            is_active=True,
            effective_date=datetime(2024, 1, 1),
            jurisdiction=["SG"],
            regulator=["MAS"],
            parameters={"cash_threshold": 20000, "currency": "SGD"},
            condition_function="check_large_cash_transaction",
            created_date=datetime.now(),
            last_updated=datetime.now(),
            version="1.0"
        ))
        
        self.add_rule(ComplianceRule(
            rule_id="MAS-SC-001",
            name="Sanctions Screening Requirement",
            description="All transactions must be screened against sanctions lists",
            category=RuleCategory.SANCTIONS_COMPLIANCE,
            severity=RuleSeverity.CRITICAL,
            is_active=True,
            effective_date=datetime(2024, 1, 1),
            jurisdiction=["SG"],
            regulator=["MAS"],
            parameters={},
            condition_function="check_sanctions_screening",
            created_date=datetime.now(),
            last_updated=datetime.now(),
            version="1.0"
        ))
        
        # HKMA (Hong Kong) Rules
        self.add_rule(ComplianceRule(
            rule_id="HKMA-TM-001",
            name="Suspicious Transaction Reporting",
            description="Transactions with unusual patterns require STR filing",
            category=RuleCategory.REPORTING_REQUIREMENTS,
            severity=RuleSeverity.HIGH,
            is_active=True,
            effective_date=datetime(2024, 1, 1),
            jurisdiction=["HK"],
            regulator=["HKMA", "SFC"],
            parameters={"pattern_threshold": 3, "time_window_hours": 24},
            condition_function="check_suspicious_patterns",
            created_date=datetime.now(),
            last_updated=datetime.now(),
            version="1.0"
        ))
        
        self.add_rule(ComplianceRule(
            rule_id="HKMA-KYC-001",
            name="Enhanced Due Diligence for PEPs",
            description="PEP customers require enhanced due diligence",
            category=RuleCategory.KYC_COMPLIANCE,
            severity=RuleSeverity.HIGH,
            is_active=True,
            effective_date=datetime(2024, 1, 1),
            jurisdiction=["HK"],
            regulator=["HKMA", "SFC"],
            parameters={"pep_transaction_limit": 1000000},
            condition_function="check_pep_edd",
            created_date=datetime.now(),
            last_updated=datetime.now(),
            version="1.0"
        ))
        
        # FINMA (Switzerland) Rules
        self.add_rule(ComplianceRule(
            rule_id="FINMA-TM-001",
            name="Cross-border Transaction Monitoring",
            description="Cross-border transactions require enhanced monitoring",
            category=RuleCategory.TRANSACTION_MONITORING,
            severity=RuleSeverity.MEDIUM,
            is_active=True,
            effective_date=datetime(2024, 1, 1),
            jurisdiction=["CH"],
            regulator=["FINMA"],
            parameters={"cross_border_threshold": 15000},
            condition_function="check_cross_border_transaction",
            created_date=datetime.now(),
            last_updated=datetime.now(),
            version="1.0"
        ))
        
        # Universal Rules (Apply to all jurisdictions)
        self.add_rule(ComplianceRule(
            rule_id="UNIVERSAL-001",
            name="High-Value Transaction Monitoring",
            description="Transactions above 1M in any currency require immediate review",
            category=RuleCategory.TRANSACTION_MONITORING,
            severity=RuleSeverity.CRITICAL,
            is_active=True,
            effective_date=datetime(2024, 1, 1),
            jurisdiction=["SG", "HK", "CH"],
            regulator=["MAS", "HKMA", "SFC", "FINMA"],
            parameters={"amount_threshold": 1000000},
            condition_function="check_high_value_transaction",
            created_date=datetime.now(),
            last_updated=datetime.now(),
            version="1.0"
        ))
        
        self.add_rule(ComplianceRule(
            rule_id="UNIVERSAL-002",
            name="Rapid Transaction Sequence",
            description="Multiple transactions in rapid succession may indicate structuring",
            category=RuleCategory.TRANSACTION_MONITORING,
            severity=RuleSeverity.HIGH,
            is_active=True,
            effective_date=datetime(2024, 1, 1),
            jurisdiction=["SG", "HK", "CH"],
            regulator=["MAS", "HKMA", "SFC", "FINMA"],
            parameters={"sequence_count": 3, "time_window_minutes": 60, "amount_threshold": 500000},
            condition_function="check_rapid_sequence",
            created_date=datetime.now(),
            last_updated=datetime.now(),
            version="1.0"
        ))
    
    def add_rule(self, rule: ComplianceRule):
        """Add a new compliance rule"""
        self.rules[rule.rule_id] = rule
        self.logger.info(f"Added rule {rule.rule_id}: {rule.name}")
    
    def update_rule(self, rule_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing rule"""
        if rule_id not in self.rules:
            return False
        
        rule = self.rules[rule_id]
        for key, value in updates.items():
            if hasattr(rule, key):
                setattr(rule, key, value)
        
        rule.last_updated = datetime.now()
        self.logger.info(f"Updated rule {rule_id}")
        return True
    
    def deactivate_rule(self, rule_id: str) -> bool:
        """Deactivate a rule"""
        if rule_id not in self.rules:
            return False
        
        self.rules[rule_id].is_active = False
        self.logger.info(f"Deactivated rule {rule_id}")
        return True
    
    def evaluate_transaction(self, transaction_data: Dict[str, Any]) -> List[RuleViolation]:
        """Evaluate a transaction against all applicable rules"""
        violations = []
        
        # Get applicable rules based on jurisdiction and regulator
        applicable_rules = self._get_applicable_rules(
            transaction_data.get('booking_jurisdiction'),
            transaction_data.get('regulator')
        )
        
        for rule in applicable_rules:
            if self._evaluate_rule_condition(rule, transaction_data):
                violation = self._create_violation(rule, transaction_data)
                violations.append(violation)
                self.rule_violations.append(violation)
        
        return violations
    
    def _get_applicable_rules(self, jurisdiction: str, regulator: str) -> List[ComplianceRule]:
        """Get rules applicable to specific jurisdiction and regulator"""
        applicable_rules = []
        
        for rule in self.rules.values():
            if not rule.is_active:
                continue
            
            # Check if rule applies to this jurisdiction/regulator
            if (jurisdiction in rule.jurisdiction or 
                regulator in rule.regulator or
                'UNIVERSAL' in rule.rule_id):
                applicable_rules.append(rule)
        
        return applicable_rules
    
    def _evaluate_rule_condition(self, rule: ComplianceRule, transaction_data: Dict[str, Any]) -> bool:
        """Evaluate if a transaction violates a specific rule"""
        condition_function = getattr(self, rule.condition_function, None)
        if not condition_function:
            self.logger.warning(f"Condition function {rule.condition_function} not found for rule {rule.rule_id}")
            return False
        
        try:
            return condition_function(rule, transaction_data)
        except Exception as e:
            self.logger.error(f"Error evaluating rule {rule.rule_id}: {e}")
            return False
    
    def _create_violation(self, rule: ComplianceRule, transaction_data: Dict[str, Any]) -> RuleViolation:
        """Create a rule violation record"""
        return RuleViolation(
            rule_id=rule.rule_id,
            rule_name=rule.name,
            severity=rule.severity,
            category=rule.category,
            description=rule.description,
            transaction_id=transaction_data.get('transaction_id', ''),
            customer_id=transaction_data.get('customer_id', ''),
            violation_details=self._extract_violation_details(rule, transaction_data),
            timestamp=datetime.now(),
            recommended_action=self._get_recommended_action(rule, transaction_data)
        )
    
    def _extract_violation_details(self, rule: ComplianceRule, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant details for the violation"""
        details = {
            'rule_parameters': rule.parameters,
            'transaction_amount': transaction_data.get('amount'),
            'transaction_currency': transaction_data.get('currency'),
            'originator_country': transaction_data.get('originator_country'),
            'beneficiary_country': transaction_data.get('beneficiary_country'),
            'channel': transaction_data.get('channel'),
            'product_type': transaction_data.get('product_type')
        }
        return details
    
    def _get_recommended_action(self, rule: ComplianceRule, transaction_data: Dict[str, Any]) -> str:
        """Get recommended action based on rule and transaction"""
        if rule.severity == RuleSeverity.CRITICAL:
            return "Immediate escalation to Legal team and transaction hold"
        elif rule.severity == RuleSeverity.HIGH:
            return "Enhanced due diligence and compliance review required"
        elif rule.severity == RuleSeverity.MEDIUM:
            return "Review transaction details and customer profile"
        else:
            return "Document findings and continue monitoring"
    
    # Rule condition functions
    def check_large_cash_transaction(self, rule: ComplianceRule, transaction_data: Dict[str, Any]) -> bool:
        """Check for large cash transactions"""
        if transaction_data.get('channel') != 'Cash':
            return False
        
        threshold = rule.parameters.get('cash_threshold', 20000)
        amount = transaction_data.get('amount', 0)
        
        return amount > threshold
    
    def check_sanctions_screening(self, rule: ComplianceRule, transaction_data: Dict[str, Any]) -> bool:
        """Check sanctions screening compliance"""
        screening_result = transaction_data.get('sanctions_screening')
        return screening_result == 'potential' or screening_result is None
    
    def check_suspicious_patterns(self, rule: ComplianceRule, transaction_data: Dict[str, Any]) -> bool:
        """Check for suspicious transaction patterns"""
        # This would typically analyze patterns across multiple transactions
        # For demo purposes, checking for round amounts and specific indicators
        amount = transaction_data.get('amount', 0)
        is_round = amount % 10000 == 0 and amount >= 10000
        
        customer_risk = transaction_data.get('customer_risk_rating', '').lower()
        high_risk_customer = customer_risk == 'high'
        
        return is_round and high_risk_customer
    
    def check_pep_edd(self, rule: ComplianceRule, transaction_data: Dict[str, Any]) -> bool:
        """Check PEP enhanced due diligence requirements"""
        is_pep = transaction_data.get('customer_is_pep', False)
        edd_performed = transaction_data.get('edd_performed', False)
        
        if not is_pep:
            return False
        
        # Check if EDD is required but not performed
        amount = transaction_data.get('amount', 0)
        threshold = rule.parameters.get('pep_transaction_limit', 1000000)
        
        return amount > threshold and not edd_performed
    
    def check_cross_border_transaction(self, rule: ComplianceRule, transaction_data: Dict[str, Any]) -> bool:
        """Check cross-border transaction requirements"""
        originator_country = transaction_data.get('originator_country')
        beneficiary_country = transaction_data.get('beneficiary_country')
        booking_jurisdiction = transaction_data.get('booking_jurisdiction')
        
        # Check if transaction crosses borders
        is_cross_border = (originator_country != booking_jurisdiction or 
                          beneficiary_country != booking_jurisdiction)
        
        if not is_cross_border:
            return False
        
        amount = transaction_data.get('amount', 0)
        threshold = rule.parameters.get('cross_border_threshold', 15000)
        
        return amount > threshold
    
    def check_high_value_transaction(self, rule: ComplianceRule, transaction_data: Dict[str, Any]) -> bool:
        """Check for high-value transactions"""
        amount = transaction_data.get('amount', 0)
        threshold = rule.parameters.get('amount_threshold', 1000000)
        
        return amount > threshold
    
    def check_rapid_sequence(self, rule: ComplianceRule, transaction_data: Dict[str, Any]) -> bool:
        """Check for rapid transaction sequences (requires transaction history)"""
        # This is a simplified check - in practice would analyze transaction history
        # For demo, checking if daily transaction count is high
        daily_count = transaction_data.get('daily_cash_txn_count', 0)
        sequence_threshold = rule.parameters.get('sequence_count', 3)
        
        return daily_count >= sequence_threshold
    
    def get_rules_by_category(self, category: RuleCategory) -> List[ComplianceRule]:
        """Get all rules in a specific category"""
        return [rule for rule in self.rules.values() if rule.category == category]
    
    def get_rules_by_jurisdiction(self, jurisdiction: str) -> List[ComplianceRule]:
        """Get all rules applicable to a jurisdiction"""
        return [rule for rule in self.rules.values() if jurisdiction in rule.jurisdiction]
    
    def get_violation_summary(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate violation summary for reporting"""
        period_violations = [
            v for v in self.rule_violations 
            if start_date <= v.timestamp <= end_date
        ]
        
        summary = {
            'total_violations': len(period_violations),
            'by_severity': {},
            'by_category': {},
            'by_rule': {},
            'top_customers': {},
            'trend_analysis': []
        }
        
        # Group by severity
        for violation in period_violations:
            severity = violation.severity.value
            summary['by_severity'][severity] = summary['by_severity'].get(severity, 0) + 1
        
        # Group by category
        for violation in period_violations:
            category = violation.category.value
            summary['by_category'][category] = summary['by_category'].get(category, 0) + 1
        
        # Group by rule
        for violation in period_violations:
            rule_name = violation.rule_name
            summary['by_rule'][rule_name] = summary['by_rule'].get(rule_name, 0) + 1
        
        # Top customers with violations
        customer_counts = {}
        for violation in period_violations:
            customer_id = violation.customer_id
            customer_counts[customer_id] = customer_counts.get(customer_id, 0) + 1
        
        summary['top_customers'] = dict(sorted(customer_counts.items(), 
                                             key=lambda x: x[1], reverse=True)[:10])
        
        return summary
    
    def export_rules_configuration(self) -> str:
        """Export current rules configuration as JSON"""
        rules_data = {}
        for rule_id, rule in self.rules.items():
            rules_data[rule_id] = {
                'name': rule.name,
                'description': rule.description,
                'category': rule.category.value,
                'severity': rule.severity.value,
                'is_active': rule.is_active,
                'effective_date': rule.effective_date.isoformat(),
                'jurisdiction': rule.jurisdiction,
                'regulator': rule.regulator,
                'parameters': rule.parameters,
                'version': rule.version
            }
        
        return json.dumps(rules_data, indent=2)

# Demo function
def demo_rules_engine():
    """Demonstrate the rules engine functionality"""
    engine = RegulatoryRulesEngine()
    
    print("Regulatory Rules Engine Demo")
    print("============================")
    
    # Sample transaction data
    sample_transaction = {
        'transaction_id': 'demo-123',
        'booking_jurisdiction': 'SG',
        'regulator': 'MAS',
        'amount': 1500000,
        'currency': 'SGD',
        'channel': 'Cash',
        'customer_id': 'CUST-123',
        'customer_is_pep': True,
        'edd_performed': False,
        'customer_risk_rating': 'High',
        'sanctions_screening': 'potential',
        'originator_country': 'SG',
        'beneficiary_country': 'CN',
        'daily_cash_txn_count': 5
    }
    
    # Evaluate transaction
    violations = engine.evaluate_transaction(sample_transaction)
    
    print(f"\nFound {len(violations)} violations:")
    for violation in violations:
        print(f"- {violation.rule_name} ({violation.severity.value}): {violation.description}")
    
    # Show rules summary
    print(f"\nTotal active rules: {len([r for r in engine.rules.values() if r.is_active])}")
    print("Rules by jurisdiction:")
    for jurisdiction in ['SG', 'HK', 'CH']:
        rules = engine.get_rules_by_jurisdiction(jurisdiction)
        print(f"  {jurisdiction}: {len(rules)} rules")

if __name__ == "__main__":
    demo_rules_engine()