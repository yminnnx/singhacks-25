"""
Alert System for AML Monitoring
Handles alert generation, routing, and management for different teams
"""

from typing import List, Dict, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json
import logging
from enum import Enum
from transaction_analysis import AMLAlert, RiskLevel, AlertType

class AlertStatus(Enum):
    PENDING = "Pending"
    ACKNOWLEDGED = "Acknowledged"
    INVESTIGATING = "Investigating"
    RESOLVED = "Resolved"
    FALSE_POSITIVE = "False Positive"

class TeamType(Enum):
    FRONT = "Front"
    COMPLIANCE = "Compliance"
    LEGAL = "Legal"

@dataclass
class AlertAction:
    action_id: str
    alert_id: str
    user_id: str
    team: str
    action_type: str  # acknowledge, investigate, resolve, escalate
    comment: str
    timestamp: datetime
    old_status: AlertStatus
    new_status: AlertStatus

class AlertManager:
    """
    Manages AML alerts including routing, status tracking, and workflow management
    """
    
    def __init__(self):
        self.alerts: Dict[str, AMLAlert] = {}
        self.alert_status: Dict[str, AlertStatus] = {}
        self.alert_actions: Dict[str, List[AlertAction]] = {}
        self.team_queues: Dict[str, List[str]] = {
            "Front": [],
            "Compliance": [],
            "Legal": []
        }
        self.escalation_rules = self._setup_escalation_rules()
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging for audit trail"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _setup_escalation_rules(self) -> Dict:
        """Define escalation rules based on alert characteristics"""
        return {
            'time_based': {
                'critical': timedelta(minutes=30),
                'high': timedelta(hours=2),
                'medium': timedelta(hours=8),
                'low': timedelta(hours=24)
            },
            'amount_based': {
                'threshold': 5000000,  # 5M threshold for automatic escalation
                'target_team': 'Legal'
            }
        }
    
    def process_alert(self, alert: AMLAlert) -> str:
        """Process a new alert and route to appropriate team"""
        # Store alert
        self.alerts[alert.alert_id] = alert
        self.alert_status[alert.alert_id] = AlertStatus.PENDING
        self.alert_actions[alert.alert_id] = []
        
        # Route to appropriate team queue
        self.team_queues[alert.target_team].append(alert.alert_id)
        
        # Check for immediate escalation
        self._check_immediate_escalation(alert)
        
        # Log alert creation
        self.logger.info(f"Alert {alert.alert_id} created and routed to {alert.target_team}")
        
        return alert.alert_id
    
    def _check_immediate_escalation(self, alert: AMLAlert):
        """Check if alert needs immediate escalation"""
        # Amount-based escalation
        if alert.amount > self.escalation_rules['amount_based']['threshold']:
            if alert.target_team != 'Legal':
                self._escalate_alert(alert.alert_id, 'Legal', 'High amount threshold exceeded')
        
        # Critical alerts to compliance
        if alert.risk_level == RiskLevel.CRITICAL and alert.target_team != 'Legal':
            self._escalate_alert(alert.alert_id, 'Compliance', 'Critical risk level detected')
    
    def acknowledge_alert(self, alert_id: str, user_id: str, team: str, comment: str = "") -> bool:
        """Acknowledge an alert"""
        if alert_id not in self.alerts:
            return False
        
        old_status = self.alert_status[alert_id]
        self.alert_status[alert_id] = AlertStatus.ACKNOWLEDGED
        
        action = AlertAction(
            action_id=f"ACT-{datetime.now().strftime('%Y%m%d%H%M%S')}-{alert_id[:8]}",
            alert_id=alert_id,
            user_id=user_id,
            team=team,
            action_type="acknowledge",
            comment=comment,
            timestamp=datetime.now(),
            old_status=old_status,
            new_status=AlertStatus.ACKNOWLEDGED
        )
        
        self.alert_actions[alert_id].append(action)
        self.logger.info(f"Alert {alert_id} acknowledged by {user_id} from {team}")
        
        return True
    
    def investigate_alert(self, alert_id: str, user_id: str, team: str, comment: str) -> bool:
        """Start investigation on an alert"""
        if alert_id not in self.alerts:
            return False
        
        old_status = self.alert_status[alert_id]
        self.alert_status[alert_id] = AlertStatus.INVESTIGATING
        
        action = AlertAction(
            action_id=f"ACT-{datetime.now().strftime('%Y%m%d%H%M%S')}-{alert_id[:8]}",
            alert_id=alert_id,
            user_id=user_id,
            team=team,
            action_type="investigate",
            comment=comment,
            timestamp=datetime.now(),
            old_status=old_status,
            new_status=AlertStatus.INVESTIGATING
        )
        
        self.alert_actions[alert_id].append(action)
        self.logger.info(f"Investigation started on alert {alert_id} by {user_id}")
        
        return True
    
    def resolve_alert(self, alert_id: str, user_id: str, team: str, 
                     resolution: str, is_false_positive: bool = False) -> bool:
        """Resolve an alert"""
        if alert_id not in self.alerts:
            return False
        
        old_status = self.alert_status[alert_id]
        new_status = AlertStatus.FALSE_POSITIVE if is_false_positive else AlertStatus.RESOLVED
        self.alert_status[alert_id] = new_status
        
        action = AlertAction(
            action_id=f"ACT-{datetime.now().strftime('%Y%m%d%H%M%S')}-{alert_id[:8]}",
            alert_id=alert_id,
            user_id=user_id,
            team=team,
            action_type="resolve",
            comment=resolution,
            timestamp=datetime.now(),
            old_status=old_status,
            new_status=new_status
        )
        
        self.alert_actions[alert_id].append(action)
        
        # Remove from team queues
        for team_queue in self.team_queues.values():
            if alert_id in team_queue:
                team_queue.remove(alert_id)
        
        self.logger.info(f"Alert {alert_id} resolved by {user_id} as {new_status.value}")
        
        return True
    
    def _escalate_alert(self, alert_id: str, target_team: str, reason: str):
        """Escalate alert to higher team"""
        if alert_id not in self.alerts:
            return
        
        alert = self.alerts[alert_id]
        old_team = alert.target_team
        
        # Update alert target team
        self.alerts[alert_id].target_team = target_team
        
        # Move to new team queue
        if alert_id in self.team_queues[old_team]:
            self.team_queues[old_team].remove(alert_id)
        self.team_queues[target_team].append(alert_id)
        
        # Log escalation
        action = AlertAction(
            action_id=f"ACT-{datetime.now().strftime('%Y%m%d%H%M%S')}-{alert_id[:8]}",
            alert_id=alert_id,
            user_id="SYSTEM",
            team="SYSTEM",
            action_type="escalate",
            comment=f"Escalated from {old_team} to {target_team}: {reason}",
            timestamp=datetime.now(),
            old_status=self.alert_status[alert_id],
            new_status=self.alert_status[alert_id]
        )
        
        self.alert_actions[alert_id].append(action)
        self.logger.info(f"Alert {alert_id} escalated from {old_team} to {target_team}: {reason}")
    
    def check_time_based_escalations(self):
        """Check for alerts that need time-based escalation"""
        current_time = datetime.now()
        
        for alert_id, alert in self.alerts.items():
            if self.alert_status[alert_id] in [AlertStatus.RESOLVED, AlertStatus.FALSE_POSITIVE]:
                continue
            
            time_since_creation = current_time - alert.timestamp
            risk_level_key = alert.risk_level.value.lower()
            
            if risk_level_key in self.escalation_rules['time_based']:
                escalation_threshold = self.escalation_rules['time_based'][risk_level_key]
                
                if time_since_creation > escalation_threshold:
                    # Escalate based on current team
                    if alert.target_team == "Front":
                        self._escalate_alert(alert_id, "Compliance", f"Time threshold exceeded ({time_since_creation})")
                    elif alert.target_team == "Compliance":
                        self._escalate_alert(alert_id, "Legal", f"Time threshold exceeded ({time_since_creation})")
    
    def get_team_queue(self, team: str, status_filter: Optional[AlertStatus] = None) -> List[Dict]:
        """Get alerts for a specific team"""
        team_alerts = []
        
        for alert_id in self.team_queues[team]:
            if status_filter and self.alert_status[alert_id] != status_filter:
                continue
            
            alert = self.alerts[alert_id]
            alert_data = asdict(alert)
            alert_data['status'] = self.alert_status[alert_id].value
            alert_data['actions_count'] = len(self.alert_actions[alert_id])
            alert_data['last_action'] = self.alert_actions[alert_id][-1] if self.alert_actions[alert_id] else None
            
            team_alerts.append(alert_data)
        
        # Sort by risk level and timestamp
        team_alerts.sort(key=lambda x: (
            -self._risk_level_priority(x['risk_level']),
            x['timestamp']
        ))
        
        return team_alerts
    
    def _risk_level_priority(self, risk_level: str) -> int:
        """Get priority number for risk level (higher = more urgent)"""
        priority_map = {
            'Critical': 4,
            'High': 3,
            'Medium': 2,
            'Low': 1
        }
        return priority_map.get(risk_level, 0)
    
    def get_alert_details(self, alert_id: str) -> Optional[Dict]:
        """Get detailed information about a specific alert"""
        if alert_id not in self.alerts:
            return None
        
        alert = self.alerts[alert_id]
        alert_data = asdict(alert)
        alert_data['status'] = self.alert_status[alert_id].value
        alert_data['actions'] = [asdict(action) for action in self.alert_actions[alert_id]]
        
        return alert_data
    
    def generate_team_summary(self, team: str) -> Dict:
        """Generate summary statistics for a team"""
        team_alert_ids = self.team_queues[team]
        
        summary = {
            'total_alerts': len(team_alert_ids),
            'pending_alerts': 0,
            'investigating_alerts': 0,
            'high_priority_alerts': 0,
            'average_age_hours': 0,
            'oldest_alert_age_hours': 0
        }
        
        if not team_alert_ids:
            return summary
        
        current_time = datetime.now()
        ages = []
        
        for alert_id in team_alert_ids:
            status = self.alert_status[alert_id]
            alert = self.alerts[alert_id]
            
            if status == AlertStatus.PENDING:
                summary['pending_alerts'] += 1
            elif status == AlertStatus.INVESTIGATING:
                summary['investigating_alerts'] += 1
            
            if alert.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                summary['high_priority_alerts'] += 1
            
            age_hours = (current_time - alert.timestamp).total_seconds() / 3600
            ages.append(age_hours)
        
        if ages:
            summary['average_age_hours'] = sum(ages) / len(ages)
            summary['oldest_alert_age_hours'] = max(ages)
        
        return summary
    
    def export_audit_trail(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Export audit trail for compliance reporting"""
        audit_records = []
        
        for alert_id, actions in self.alert_actions.items():
            alert = self.alerts[alert_id]
            
            # Include alert creation
            if start_date <= alert.timestamp <= end_date:
                audit_records.append({
                    'timestamp': alert.timestamp.isoformat(),
                    'alert_id': alert_id,
                    'event_type': 'alert_created',
                    'user_id': 'SYSTEM',
                    'team': alert.target_team,
                    'details': {
                        'alert_type': alert.alert_type.value,
                        'risk_level': alert.risk_level.value,
                        'amount': alert.amount,
                        'currency': alert.currency,
                        'customer_id': alert.customer_id
                    }
                })
            
            # Include all actions
            for action in actions:
                if start_date <= action.timestamp <= end_date:
                    audit_records.append({
                        'timestamp': action.timestamp.isoformat(),
                        'alert_id': alert_id,
                        'event_type': action.action_type,
                        'user_id': action.user_id,
                        'team': action.team,
                        'details': {
                            'comment': action.comment,
                            'old_status': action.old_status.value,
                            'new_status': action.new_status.value
                        }
                    })
        
        # Sort by timestamp
        audit_records.sort(key=lambda x: x['timestamp'])
        
        return audit_records

# Example usage and testing
def demo_alert_system():
    """Demonstrate the alert system functionality"""
    from transaction_analysis import TransactionAnalysisEngine
    
    # Create alert manager
    alert_manager = AlertManager()
    
    # Create some sample alerts (normally from transaction analysis)
    engine = TransactionAnalysisEngine()
    
    # Example: Process some alerts
    print("Alert System Demo")
    print("=================")
    
    # Simulate team interactions
    print("\nTeam Queues:")
    for team in ["Front", "Compliance", "Legal"]:
        summary = alert_manager.generate_team_summary(team)
        print(f"{team}: {summary}")

if __name__ == "__main__":
    demo_alert_system()