"""
Transaction Analysis Engine for AML Monitoring
Analyzes transaction data for suspicious patterns and AML risks
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from enum import Enum

class RiskLevel(Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"

class AlertType(Enum):
    LARGE_TRANSACTION = "Large Transaction"
    UNUSUAL_PATTERN = "Unusual Pattern"
    HIGH_RISK_COUNTRY = "High Risk Country"
    ROUND_AMOUNT = "Round Amount"
    RAPID_SUCCESSION = "Rapid Succession"
    SANCTIONS_HIT = "Sanctions Hit"
    PEP_INVOLVEMENT = "PEP Involvement"
    CASH_INTENSIVE = "Cash Intensive"
    STRUCTURING = "Structuring"

@dataclass
class AMLAlert:
    alert_id: str
    transaction_id: str
    alert_type: AlertType
    risk_level: RiskLevel
    description: str
    risk_score: float
    timestamp: datetime
    customer_id: str
    amount: float
    currency: str
    triggered_rules: List[str]
    requires_action: bool
    target_team: str  # Front, Compliance, or Legal

class TransactionAnalysisEngine:
    """
    Core engine for analyzing transactions against AML rules
    """
    
    def __init__(self):
        self.high_risk_countries = [
            'IR', 'KP', 'SY', 'AF', 'MM', 'BY', 'RU', 'CN'  # Sample high-risk countries
        ]
        self.large_amount_threshold = 1000000  # 1M threshold for large amounts
        self.cash_threshold = 10000  # Cash transaction threshold
        self.rapid_succession_window = 3600  # 1 hour in seconds
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging for audit trail"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def load_transactions(self, file_path: str) -> pd.DataFrame:
        """Load and preprocess transaction data"""
        try:
            df = pd.read_csv(file_path)
            df['booking_datetime'] = pd.to_datetime(df['booking_datetime'])
            df['value_date'] = pd.to_datetime(df['value_date'])
            self.logger.info(f"Loaded {len(df)} transactions from {file_path}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading transactions: {e}")
            raise
    
    def calculate_risk_score(self, transaction: pd.Series) -> float:
        """Calculate composite risk score for a transaction"""
        risk_score = 0.0
        
        # Amount-based risk
        if transaction['amount'] > self.large_amount_threshold:
            risk_score += 30
        elif transaction['amount'] > 500000:
            risk_score += 15
        
        # Country risk
        if transaction['originator_country'] in self.high_risk_countries:
            risk_score += 25
        if transaction['beneficiary_country'] in self.high_risk_countries:
            risk_score += 25
        
        # Customer risk
        if transaction['customer_risk_rating'] == 'High':
            risk_score += 20
        elif transaction['customer_risk_rating'] == 'Medium':
            risk_score += 10
        
        # PEP involvement
        if transaction['customer_is_pep']:
            risk_score += 15
        
        # Cash transactions
        if transaction['channel'] == 'Cash':
            risk_score += 20
        
        # Sanctions screening results
        if transaction['sanctions_screening'] == 'potential':
            risk_score += 40
        
        # Round amounts (potential structuring)
        if self._is_round_amount(transaction['amount']):
            risk_score += 10
        
        # Swift field completeness
        if not transaction['swift_f50_present'] or not transaction['swift_f59_present']:
            risk_score += 5
        
        return min(risk_score, 100)  # Cap at 100
    
    def _is_round_amount(self, amount: float) -> bool:
        """Check if amount is suspiciously round"""
        return amount % 10000 == 0 and amount >= 10000
    
    def detect_patterns(self, df: pd.DataFrame) -> List[AMLAlert]:
        """Detect suspicious patterns across transactions"""
        alerts = []
        
        # Group by customer for pattern analysis
        for customer_id, customer_txns in df.groupby('customer_id'):
            alerts.extend(self._analyze_customer_patterns(customer_txns))
        
        return alerts
    
    def _analyze_customer_patterns(self, customer_txns: pd.DataFrame) -> List[AMLAlert]:
        """Analyze patterns for a specific customer"""
        alerts = []
        customer_id = customer_txns.iloc[0]['customer_id']
        
        # Sort by datetime for pattern analysis
        customer_txns = customer_txns.sort_values('booking_datetime')
        
        # 1. Rapid succession transactions
        for i in range(len(customer_txns) - 1):
            time_diff = (customer_txns.iloc[i+1]['booking_datetime'] - 
                        customer_txns.iloc[i]['booking_datetime']).total_seconds()
            
            if time_diff <= self.rapid_succession_window:
                alert = self._create_alert(
                    customer_txns.iloc[i+1],
                    AlertType.RAPID_SUCCESSION,
                    f"Rapid succession transactions within {time_diff/60:.1f} minutes",
                    RiskLevel.MEDIUM,
                    ["rapid_succession_pattern"]
                )
                alerts.append(alert)
        
        # 2. Structuring detection (multiple transactions just under threshold)
        daily_amounts = customer_txns.groupby(customer_txns['booking_datetime'].dt.date)['amount'].sum()
        for date, total_amount in daily_amounts.items():
            daily_txns = customer_txns[customer_txns['booking_datetime'].dt.date == date]
            if len(daily_txns) >= 3 and total_amount > self.large_amount_threshold:
                # Multiple transactions on same day totaling large amount
                for _, txn in daily_txns.iterrows():
                    alert = self._create_alert(
                        txn,
                        AlertType.STRUCTURING,
                        f"Potential structuring: {len(daily_txns)} transactions totaling {total_amount:,.2f}",
                        RiskLevel.HIGH,
                        ["structuring_pattern", "multiple_daily_transactions"]
                    )
                    alerts.append(alert)
        
        return alerts
    
    def analyze_single_transaction(self, transaction: pd.Series) -> List[AMLAlert]:
        """Analyze a single transaction for immediate alerts"""
        alerts = []
        risk_score = self.calculate_risk_score(transaction)
        
        # High-value transaction alert
        if transaction['amount'] > self.large_amount_threshold:
            alert = self._create_alert(
                transaction,
                AlertType.LARGE_TRANSACTION,
                f"Large transaction: {transaction['amount']:,.2f} {transaction['currency']}",
                RiskLevel.HIGH if risk_score >= 70 else RiskLevel.MEDIUM,
                ["large_amount_threshold"]
            )
            alerts.append(alert)
        
        # High-risk country involvement
        if (transaction['originator_country'] in self.high_risk_countries or 
            transaction['beneficiary_country'] in self.high_risk_countries):
            alert = self._create_alert(
                transaction,
                AlertType.HIGH_RISK_COUNTRY,
                f"High-risk country involvement: {transaction['originator_country']} -> {transaction['beneficiary_country']}",
                RiskLevel.HIGH,
                ["high_risk_country"]
            )
            alerts.append(alert)
        
        # PEP involvement
        if transaction['customer_is_pep']:
            alert = self._create_alert(
                transaction,
                AlertType.PEP_INVOLVEMENT,
                "Transaction involves Politically Exposed Person (PEP)",
                RiskLevel.MEDIUM,
                ["pep_involvement"]
            )
            alerts.append(alert)
        
        # Sanctions screening hits
        if transaction['sanctions_screening'] == 'potential':
            alert = self._create_alert(
                transaction,
                AlertType.SANCTIONS_HIT,
                "Potential sanctions screening match",
                RiskLevel.CRITICAL,
                ["sanctions_screening_hit"]
            )
            alerts.append(alert)
        
        # Cash intensive transactions
        if transaction['channel'] == 'Cash' and transaction['amount'] > self.cash_threshold:
            alert = self._create_alert(
                transaction,
                AlertType.CASH_INTENSIVE,
                f"Large cash transaction: {transaction['amount']:,.2f}",
                RiskLevel.HIGH,
                ["cash_transaction_threshold"]
            )
            alerts.append(alert)
        
        return alerts
    
    def _create_alert(self, transaction: pd.Series, alert_type: AlertType, 
                     description: str, risk_level: RiskLevel, 
                     triggered_rules: List[str]) -> AMLAlert:
        """Create an AML alert from transaction data"""
        
        # Determine target team based on alert type and risk level
        target_team = self._determine_target_team(alert_type, risk_level)
        
        alert = AMLAlert(
            alert_id=f"ALT-{datetime.now().strftime('%Y%m%d%H%M%S')}-{transaction['transaction_id'][:8]}",
            transaction_id=transaction['transaction_id'],
            alert_type=alert_type,
            risk_level=risk_level,
            description=description,
            risk_score=self.calculate_risk_score(transaction),
            timestamp=datetime.now(),
            customer_id=transaction['customer_id'],
            amount=transaction['amount'],
            currency=transaction['currency'],
            triggered_rules=triggered_rules,
            requires_action=risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL],
            target_team=target_team
        )
        
        return alert
    
    def _determine_target_team(self, alert_type: AlertType, risk_level: RiskLevel) -> str:
        """Determine which team should receive the alert"""
        if alert_type == AlertType.SANCTIONS_HIT:
            return "Legal"
        elif risk_level == RiskLevel.CRITICAL:
            return "Compliance"
        elif alert_type in [AlertType.LARGE_TRANSACTION, AlertType.PEP_INVOLVEMENT]:
            return "Front"
        elif risk_level == RiskLevel.HIGH:
            return "Compliance"
        else:
            return "Front"
    
    def generate_risk_report(self, df: pd.DataFrame) -> Dict:
        """Generate comprehensive risk analysis report"""
        report = {
            'summary': {
                'total_transactions': len(df),
                'total_amount': df['amount'].sum(),
                'high_risk_transactions': len(df[df.apply(lambda x: self.calculate_risk_score(x) >= 70, axis=1)]),
                'average_risk_score': df.apply(lambda x: self.calculate_risk_score(x), axis=1).mean()
            },
            'risk_distribution': {},
            'country_analysis': {},
            'customer_analysis': {},
            'recommendations': []
        }
        
        # Risk score distribution
        risk_scores = df.apply(lambda x: self.calculate_risk_score(x), axis=1)
        report['risk_distribution'] = {
            'low_risk': len(risk_scores[risk_scores < 30]),
            'medium_risk': len(risk_scores[(risk_scores >= 30) & (risk_scores < 70)]),
            'high_risk': len(risk_scores[risk_scores >= 70])
        }
        
        # Country risk analysis
        country_risks = {}
        for country in df['originator_country'].unique():
            country_txns = df[df['originator_country'] == country]
            avg_risk = country_txns.apply(lambda x: self.calculate_risk_score(x), axis=1).mean()
            country_risks[country] = {
                'transaction_count': len(country_txns),
                'average_risk_score': avg_risk,
                'total_amount': country_txns['amount'].sum()
            }
        report['country_analysis'] = country_risks
        
        # Generate recommendations
        if report['risk_distribution']['high_risk'] > len(df) * 0.1:  # More than 10% high risk
            report['recommendations'].append("High proportion of risky transactions detected. Consider enhanced monitoring.")
        
        return report

def main():
    """Demo function to test the transaction analysis engine"""
    engine = TransactionAnalysisEngine()
    
    # Load transaction data
    df = engine.load_transactions('../data/transactions_mock_1000_for_participants.csv')
    
    # Analyze all transactions
    all_alerts = []
    for _, transaction in df.iterrows():
        alerts = engine.analyze_single_transaction(transaction)
        all_alerts.extend(alerts)
    
    # Add pattern-based alerts
    pattern_alerts = engine.detect_patterns(df)
    all_alerts.extend(pattern_alerts)
    
    # Generate risk report
    risk_report = engine.generate_risk_report(df)
    
    print(f"Generated {len(all_alerts)} alerts")
    print(f"Risk Report Summary: {risk_report['summary']}")
    
    return all_alerts, risk_report

if __name__ == "__main__":
    main()