# Julius Baer AML Monitoring System

## Project Structure

```
├── src/
│   ├── part1_aml_monitoring/        # Real-time AML monitoring and alerts
│   ├── part2_document_corroboration/ # Document and image verification
│   ├── shared/                      # Shared utilities and models
│   └── frontend/                    # Web interface
├── data/                           # Transaction data and test documents
├── reports/                        # Generated reports and audit trails
└── requirements.txt               # Python dependencies
```

## Getting Started

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
cd src/frontend
streamlit run app.py
```

## Features

### Part 1: Real-Time AML Monitoring
- Transaction risk analysis
- Regulatory compliance checking
- Role-based alerting system
- Audit trail maintenance

### Part 2: Document Corroboration
- Multi-format document processing
- Image authenticity verification
- Format validation
- Risk scoring and reporting