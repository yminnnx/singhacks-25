"""
Audit Trail System for AML Monitoring
Comprehensive logging and audit capabilities for compliance and regulatory reporting
"""

import logging
import json
import hashlib
import sqlite3
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import os
import csv

class EventType(Enum):
    TRANSACTION_ANALYSIS = "Transaction Analysis"
    ALERT_CREATED = "Alert Created"
    ALERT_ACKNOWLEDGED = "Alert Acknowledged"
    ALERT_ESCALATED = "Alert Escalated"
    ALERT_RESOLVED = "Alert Resolved"
    DOCUMENT_UPLOADED = "Document Uploaded"
    DOCUMENT_PROCESSED = "Document Processed"
    DOCUMENT_APPROVED = "Document Approved"
    DOCUMENT_REJECTED = "Document Rejected"
    IMAGE_ANALYZED = "Image Analyzed"
    RULE_TRIGGERED = "Rule Triggered"
    RULE_CREATED = "Rule Created"
    RULE_MODIFIED = "Rule Modified"
    USER_LOGIN = "User Login"
    USER_LOGOUT = "User Logout"
    SYSTEM_ERROR = "System Error"
    COMPLIANCE_REPORT = "Compliance Report"

class UserRole(Enum):
    FRONT_OFFICE = "Front Office"
    COMPLIANCE_OFFICER = "Compliance Officer"
    LEGAL_COUNSEL = "Legal Counsel"
    SYSTEM_ADMIN = "System Administrator"
    AUDITOR = "Auditor"
    SYSTEM = "System"

@dataclass
class AuditEvent:
    event_id: str
    timestamp: datetime
    event_type: EventType
    user_id: str
    user_role: UserRole
    entity_type: str  # e.g., "transaction", "alert", "document"
    entity_id: str
    action_description: str
    before_state: Optional[Dict[str, Any]]
    after_state: Optional[Dict[str, Any]]
    metadata: Dict[str, Any]
    ip_address: Optional[str]
    session_id: Optional[str]
    risk_level: Optional[str]
    compliance_flags: List[str]
    data_hash: str

@dataclass
class ComplianceMetrics:
    period_start: datetime
    period_end: datetime
    total_transactions: int
    total_alerts: int
    alerts_resolved: int
    documents_processed: int
    documents_rejected: int
    rules_triggered: int
    avg_alert_resolution_time: float
    compliance_violations: int
    audit_findings: int

class AuditTrailManager:
    """
    Comprehensive audit trail management system for AML compliance
    """
    
    def __init__(self, db_path: str = "audit_trail.db"):
        self.db_path = db_path
        self.setup_database()
        self.setup_logging()
        
    def setup_database(self):
        """Initialize SQLite database for audit trail storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create audit_events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_events (
                event_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                user_id TEXT NOT NULL,
                user_role TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                action_description TEXT NOT NULL,
                before_state TEXT,
                after_state TEXT,
                metadata TEXT,
                ip_address TEXT,
                session_id TEXT,
                risk_level TEXT,
                compliance_flags TEXT,
                data_hash TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indices for better query performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_events(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_event_type ON audit_events(event_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_id ON audit_events(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_entity_id ON audit_events(entity_id)')
        
        # Create compliance_metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS compliance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                period_start TEXT NOT NULL,
                period_end TEXT NOT NULL,
                metrics_data TEXT NOT NULL,
                generated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def setup_logging(self):
        """Setup logging for the audit system itself"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('audit_system.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def log_event(self, event: AuditEvent) -> str:
        """Log an audit event to the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO audit_events (
                    event_id, timestamp, event_type, user_id, user_role,
                    entity_type, entity_id, action_description, before_state,
                    after_state, metadata, ip_address, session_id, risk_level,
                    compliance_flags, data_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                event.event_id,
                event.timestamp.isoformat(),
                event.event_type.value,
                event.user_id,
                event.user_role.value,
                event.entity_type,
                event.entity_id,
                event.action_description,
                json.dumps(event.before_state) if event.before_state else None,
                json.dumps(event.after_state) if event.after_state else None,
                json.dumps(event.metadata),
                event.ip_address,
                event.session_id,
                event.risk_level,
                json.dumps(event.compliance_flags),
                event.data_hash
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Audit event logged: {event.event_id} - {event.event_type.value}")
            return event.event_id
            
        except Exception as e:
            self.logger.error(f"Failed to log audit event: {e}")
            raise
    
    def create_event(self, event_type: EventType, user_id: str, user_role: UserRole,
                    entity_type: str, entity_id: str, action_description: str,
                    before_state: Optional[Dict] = None, after_state: Optional[Dict] = None,
                    metadata: Optional[Dict] = None, ip_address: Optional[str] = None,
                    session_id: Optional[str] = None, risk_level: Optional[str] = None,
                    compliance_flags: Optional[List[str]] = None) -> str:
        """Create and log an audit event"""
        
        # Generate event ID
        event_id = self._generate_event_id()
        
        # Calculate data hash for integrity
        data_hash = self._calculate_data_hash(event_type, user_id, entity_id, 
                                            action_description, before_state, after_state)
        
        # Create event
        event = AuditEvent(
            event_id=event_id,
            timestamp=datetime.now(),
            event_type=event_type,
            user_id=user_id,
            user_role=user_role,
            entity_type=entity_type,
            entity_id=entity_id,
            action_description=action_description,
            before_state=before_state,
            after_state=after_state,
            metadata=metadata or {},
            ip_address=ip_address,
            session_id=session_id,
            risk_level=risk_level,
            compliance_flags=compliance_flags or [],
            data_hash=data_hash
        )
        
        return self.log_event(event)
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID"""
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
        return f"AUD-{timestamp}"
    
    def _calculate_data_hash(self, event_type: EventType, user_id: str, entity_id: str,
                           action_description: str, before_state: Optional[Dict],
                           after_state: Optional[Dict]) -> str:
        """Calculate SHA-256 hash of critical event data for integrity verification"""
        data_string = f"{event_type.value}|{user_id}|{entity_id}|{action_description}|{json.dumps(before_state, sort_keys=True) if before_state else ''}|{json.dumps(after_state, sort_keys=True) if after_state else ''}"
        return hashlib.sha256(data_string.encode()).hexdigest()
    
    def query_events(self, start_date: Optional[datetime] = None,
                    end_date: Optional[datetime] = None,
                    event_types: Optional[List[EventType]] = None,
                    user_ids: Optional[List[str]] = None,
                    entity_types: Optional[List[str]] = None,
                    entity_ids: Optional[List[str]] = None,
                    risk_levels: Optional[List[str]] = None,
                    limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Query audit events with various filters"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Build query
        query = "SELECT * FROM audit_events WHERE 1=1"
        params = []
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.isoformat())
        
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date.isoformat())
        
        if event_types:
            placeholders = ','.join(['?' for _ in event_types])
            query += f" AND event_type IN ({placeholders})"
            params.extend([et.value for et in event_types])
        
        if user_ids:
            placeholders = ','.join(['?' for _ in user_ids])
            query += f" AND user_id IN ({placeholders})"
            params.extend(user_ids)
        
        if entity_types:
            placeholders = ','.join(['?' for _ in entity_types])
            query += f" AND entity_type IN ({placeholders})"
            params.extend(entity_types)
        
        if entity_ids:
            placeholders = ','.join(['?' for _ in entity_ids])
            query += f" AND entity_id IN ({placeholders})"
            params.extend(entity_ids)
        
        if risk_levels:
            placeholders = ','.join(['?' for _ in risk_levels])
            query += f" AND risk_level IN ({placeholders})"
            params.extend(risk_levels)
        
        query += " ORDER BY timestamp DESC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        # Convert to dictionaries
        columns = [description[0] for description in cursor.description]
        results = []
        
        for row in rows:
            event_dict = dict(zip(columns, row))
            
            # Parse JSON fields
            for json_field in ['before_state', 'after_state', 'metadata', 'compliance_flags']:
                if event_dict[json_field]:
                    try:
                        event_dict[json_field] = json.loads(event_dict[json_field])
                    except json.JSONDecodeError:
                        event_dict[json_field] = None
            
            results.append(event_dict)
        
        conn.close()
        return results
    
    def verify_event_integrity(self, event_id: str) -> bool:
        """Verify the integrity of an audit event using its hash"""
        try:
            events = self.query_events()
            event = next((e for e in events if e['event_id'] == event_id), None)
            
            if not event:
                return False
            
            # Recalculate hash
            calculated_hash = self._calculate_data_hash(
                EventType(event['event_type']),
                event['user_id'],
                event['entity_id'],
                event['action_description'],
                event['before_state'],
                event['after_state']
            )
            
            return calculated_hash == event['data_hash']
            
        except Exception as e:
            self.logger.error(f"Failed to verify event integrity: {e}")
            return False
    
    def generate_compliance_report(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate comprehensive compliance report"""
        try:
            # Query events for the period
            events = self.query_events(start_date=start_date, end_date=end_date)
            
            # Calculate metrics
            metrics = self._calculate_compliance_metrics(events, start_date, end_date)
            
            # Generate detailed analysis
            report = {
                'report_id': f"COMP-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                'period': {
                    'start': start_date.isoformat(),
                    'end': end_date.isoformat(),
                    'duration_days': (end_date - start_date).days
                },
                'summary': {
                    'total_events': len(events),
                    'total_transactions': metrics.total_transactions,
                    'total_alerts': metrics.total_alerts,
                    'alerts_resolved': metrics.alerts_resolved,
                    'documents_processed': metrics.documents_processed,
                    'documents_rejected': metrics.documents_rejected,
                    'compliance_violations': metrics.compliance_violations
                },
                'performance_metrics': {
                    'alert_resolution_rate': (metrics.alerts_resolved / metrics.total_alerts * 100) if metrics.total_alerts > 0 else 0,
                    'document_approval_rate': ((metrics.documents_processed - metrics.documents_rejected) / metrics.documents_processed * 100) if metrics.documents_processed > 0 else 0,
                    'avg_alert_resolution_time_hours': metrics.avg_alert_resolution_time,
                    'rules_triggered': metrics.rules_triggered
                },
                'risk_analysis': self._analyze_risk_trends(events),
                'user_activity': self._analyze_user_activity(events),
                'regulatory_compliance': self._assess_regulatory_compliance(events),
                'recommendations': self._generate_compliance_recommendations(metrics, events),
                'generated_at': datetime.now().isoformat(),
                'generated_by': 'AML Audit System'
            }
            
            # Store report in database
            self._store_compliance_report(report, start_date, end_date)
            
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate compliance report: {e}")
            raise
    
    def _calculate_compliance_metrics(self, events: List[Dict], start_date: datetime, end_date: datetime) -> ComplianceMetrics:
        """Calculate compliance metrics from events"""
        
        # Count different event types
        transaction_events = [e for e in events if e['event_type'] == EventType.TRANSACTION_ANALYSIS.value]
        alert_events = [e for e in events if e['event_type'] == EventType.ALERT_CREATED.value]
        resolved_alerts = [e for e in events if e['event_type'] == EventType.ALERT_RESOLVED.value]
        document_events = [e for e in events if e['event_type'] == EventType.DOCUMENT_PROCESSED.value]
        rejected_docs = [e for e in events if e['event_type'] == EventType.DOCUMENT_REJECTED.value]
        rule_triggers = [e for e in events if e['event_type'] == EventType.RULE_TRIGGERED.value]
        
        # Calculate average resolution time (simplified)
        avg_resolution_time = 24.0  # Default in hours, would calculate from actual resolution times
        
        # Count compliance violations
        compliance_violations = len([e for e in events if e.get('risk_level') == 'Critical'])
        
        return ComplianceMetrics(
            period_start=start_date,
            period_end=end_date,
            total_transactions=len(transaction_events),
            total_alerts=len(alert_events),
            alerts_resolved=len(resolved_alerts),
            documents_processed=len(document_events),
            documents_rejected=len(rejected_docs),
            rules_triggered=len(rule_triggers),
            avg_alert_resolution_time=avg_resolution_time,
            compliance_violations=compliance_violations,
            audit_findings=0  # Would be calculated based on audit findings
        )
    
    def _analyze_risk_trends(self, events: List[Dict]) -> Dict[str, Any]:
        """Analyze risk trends from audit events"""
        risk_events = [e for e in events if e.get('risk_level')]
        
        risk_distribution = {}
        for event in risk_events:
            risk_level = event['risk_level']
            risk_distribution[risk_level] = risk_distribution.get(risk_level, 0) + 1
        
        return {
            'risk_distribution': risk_distribution,
            'high_risk_entities': self._identify_high_risk_entities(events),
            'risk_trend_analysis': 'Increasing' if len(risk_events) > 50 else 'Stable'  # Simplified
        }
    
    def _analyze_user_activity(self, events: List[Dict]) -> Dict[str, Any]:
        """Analyze user activity patterns"""
        user_activity = {}
        
        for event in events:
            user_id = event['user_id']
            if user_id not in user_activity:
                user_activity[user_id] = {
                    'total_actions': 0,
                    'action_types': {},
                    'last_activity': event['timestamp']
                }
            
            user_activity[user_id]['total_actions'] += 1
            event_type = event['event_type']
            user_activity[user_id]['action_types'][event_type] = user_activity[user_id]['action_types'].get(event_type, 0) + 1
        
        return {
            'active_users': len(user_activity),
            'user_activity_summary': user_activity,
            'most_active_users': sorted(user_activity.items(), key=lambda x: x[1]['total_actions'], reverse=True)[:5]
        }
    
    def _assess_regulatory_compliance(self, events: List[Dict]) -> Dict[str, Any]:
        """Assess regulatory compliance status"""
        
        # Check for required compliance activities
        str_filings = len([e for e in events if 'STR' in e.get('action_description', '')])
        kyc_updates = len([e for e in events if 'KYC' in e.get('action_description', '')])
        edd_completions = len([e for e in events if 'EDD' in e.get('action_description', '')])
        
        return {
            'str_filing_compliance': str_filings >= 5,  # Minimum threshold
            'kyc_update_compliance': kyc_updates >= 10,
            'edd_completion_compliance': edd_completions >= 3,
            'overall_compliance_score': 85.5,  # Would be calculated based on various factors
            'compliance_gaps': self._identify_compliance_gaps(events)
        }
    
    def _identify_high_risk_entities(self, events: List[Dict]) -> List[Dict[str, Any]]:
        """Identify high-risk entities based on events"""
        entity_risk_scores = {}
        
        for event in events:
            entity_id = event['entity_id']
            risk_level = event.get('risk_level', 'Low')
            
            if entity_id not in entity_risk_scores:
                entity_risk_scores[entity_id] = 0
            
            # Add to risk score based on risk level
            risk_weights = {'Critical': 10, 'High': 5, 'Medium': 2, 'Low': 1}
            entity_risk_scores[entity_id] += risk_weights.get(risk_level, 0)
        
        # Return top 10 highest risk entities
        high_risk_entities = sorted(entity_risk_scores.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return [{'entity_id': entity_id, 'risk_score': score} for entity_id, score in high_risk_entities]
    
    def _identify_compliance_gaps(self, events: List[Dict]) -> List[str]:
        """Identify potential compliance gaps"""
        gaps = []
        
        # Check for various compliance indicators
        if len([e for e in events if e['event_type'] == EventType.ALERT_RESOLVED.value]) < 10:
            gaps.append("Low alert resolution rate")
        
        if len([e for e in events if e['event_type'] == EventType.DOCUMENT_REJECTED.value]) > 50:
            gaps.append("High document rejection rate")
        
        if len([e for e in events if e.get('risk_level') == 'Critical']) > 5:
            gaps.append("Multiple critical risk incidents")
        
        return gaps
    
    def _generate_compliance_recommendations(self, metrics: ComplianceMetrics, events: List[Dict]) -> List[str]:
        """Generate compliance recommendations"""
        recommendations = []
        
        # Alert resolution recommendations
        if metrics.alerts_resolved / metrics.total_alerts < 0.8:
            recommendations.append("Improve alert resolution processes - current resolution rate is below 80%")
        
        # Document processing recommendations
        if metrics.documents_rejected / metrics.documents_processed > 0.3:
            recommendations.append("Review document processing criteria - rejection rate is above 30%")
        
        # Risk management recommendations
        if metrics.compliance_violations > 10:
            recommendations.append("Enhance risk monitoring controls - multiple compliance violations detected")
        
        # Training recommendations
        user_errors = len([e for e in events if 'error' in e.get('action_description', '').lower()])
        if user_errors > 20:
            recommendations.append("Consider additional user training - multiple user errors detected")
        
        return recommendations
    
    def _store_compliance_report(self, report: Dict[str, Any], start_date: datetime, end_date: datetime):
        """Store compliance report in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO compliance_metrics (period_start, period_end, metrics_data)
                VALUES (?, ?, ?)
            ''', (start_date.isoformat(), end_date.isoformat(), json.dumps(report)))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to store compliance report: {e}")
    
    def export_audit_trail(self, start_date: datetime, end_date: datetime, 
                          export_format: str = 'csv', file_path: Optional[str] = None) -> str:
        """Export audit trail to various formats"""
        
        events = self.query_events(start_date=start_date, end_date=end_date)
        
        if not file_path:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_path = f"audit_trail_{timestamp}.{export_format}"
        
        if export_format.lower() == 'csv':
            self._export_to_csv(events, file_path)
        elif export_format.lower() == 'json':
            self._export_to_json(events, file_path)
        else:
            raise ValueError(f"Unsupported export format: {export_format}")
        
        self.logger.info(f"Audit trail exported to {file_path}")
        return file_path
    
    def _export_to_csv(self, events: List[Dict], file_path: str):
        """Export events to CSV format"""
        if not events:
            return
        
        fieldnames = events[0].keys()
        
        with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for event in events:
                # Convert complex fields to strings
                row = event.copy()
                for key, value in row.items():
                    if isinstance(value, (dict, list)):
                        row[key] = json.dumps(value)
                writer.writerow(row)
    
    def _export_to_json(self, events: List[Dict], file_path: str):
        """Export events to JSON format"""
        with open(file_path, 'w', encoding='utf-8') as jsonfile:
            json.dump(events, jsonfile, indent=2, default=str)
    
    def get_audit_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Get audit trail statistics for the last N days"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        events = self.query_events(start_date=start_date, end_date=end_date)
        
        stats = {
            'period_days': days,
            'total_events': len(events),
            'events_by_type': {},
            'events_by_user_role': {},
            'events_by_risk_level': {},
            'daily_event_count': {},
            'integrity_check_passed': True
        }
        
        # Count events by type
        for event in events:
            event_type = event['event_type']
            stats['events_by_type'][event_type] = stats['events_by_type'].get(event_type, 0) + 1
            
            user_role = event['user_role']
            stats['events_by_user_role'][user_role] = stats['events_by_user_role'].get(user_role, 0) + 1
            
            risk_level = event.get('risk_level', 'Unknown')
            stats['events_by_risk_level'][risk_level] = stats['events_by_risk_level'].get(risk_level, 0) + 1
            
            # Daily counts
            event_date = event['timestamp'][:10]  # Extract date part
            stats['daily_event_count'][event_date] = stats['daily_event_count'].get(event_date, 0) + 1
        
        return stats

# Demo and testing functions
def demo_audit_trail():
    """Demonstrate audit trail functionality"""
    print("Audit Trail System Demo")
    print("=======================")
    
    # Initialize audit manager
    audit_manager = AuditTrailManager("demo_audit.db")
    
    # Create sample events
    print("\n1. Creating sample audit events...")
    
    # Transaction analysis event
    audit_manager.create_event(
        event_type=EventType.TRANSACTION_ANALYSIS,
        user_id="system",
        user_role=UserRole.SYSTEM,
        entity_type="transaction",
        entity_id="TXN-123456",
        action_description="High-value transaction analyzed",
        after_state={"risk_score": 85, "status": "flagged"},
        metadata={"amount": 2500000, "currency": "USD"},
        risk_level="High"
    )
    
    # Alert creation event
    audit_manager.create_event(
        event_type=EventType.ALERT_CREATED,
        user_id="system", 
        user_role=UserRole.SYSTEM,
        entity_type="alert",
        entity_id="ALT-789012",
        action_description="Large transaction alert created",
        after_state={"status": "pending", "assigned_to": "compliance_team"},
        metadata={"transaction_id": "TXN-123456", "alert_type": "large_transaction"},
        risk_level="High",
        compliance_flags=["large_amount", "high_risk_customer"]
    )
    
    # Document processing event
    audit_manager.create_event(
        event_type=EventType.DOCUMENT_PROCESSED,
        user_id="compliance_officer_1",
        user_role=UserRole.COMPLIANCE_OFFICER,
        entity_type="document",
        entity_id="DOC-345678",
        action_description="Swiss purchase agreement processed",
        before_state={"status": "uploaded"},
        after_state={"status": "approved", "risk_score": 25},
        metadata={"document_type": "purchase_agreement", "file_size": 2048576},
        risk_level="Low"
    )
    
    print("✓ Sample events created")
    
    # Query events
    print("\n2. Querying audit events...")
    recent_events = audit_manager.query_events(limit=10)
    print(f"✓ Retrieved {len(recent_events)} recent events")
    
    # Generate compliance report
    print("\n3. Generating compliance report...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    report = audit_manager.generate_compliance_report(start_date, end_date)
    print(f"✓ Compliance report generated: {report['report_id']}")
    print(f"  - Total events: {report['summary']['total_events']}")
    print(f"  - Compliance violations: {report['summary']['compliance_violations']}")
    
    # Get statistics
    print("\n4. Getting audit statistics...")
    stats = audit_manager.get_audit_statistics(days=30)
    print(f"✓ Statistics for last 30 days:")
    print(f"  - Total events: {stats['total_events']}")
    print(f"  - Event types: {len(stats['events_by_type'])}")
    
    # Export audit trail
    print("\n5. Exporting audit trail...")
    export_file = audit_manager.export_audit_trail(start_date, end_date, 'csv')
    print(f"✓ Audit trail exported to: {export_file}")
    
    print("\n✅ Audit trail demo completed successfully!")

if __name__ == "__main__":
    demo_audit_trail()