# MLOps Assignment: End-to-End ML Pipeline

This project demonstrates a complete MLOps pipeline for a machine learning model that predicts health outcomes based on patient data. It includes data versioning, model training, testing, API deployment, containerization, orchestration, and cloud deployment.

## 🚀 Features

- **Data Versioning**: Uses DVC for tracking datasets and models
- **Model Training**: Logistic Regression for binary classification
- **Automated Testing**: Comprehensive test suite with pytest
- **REST API**: FastAPI-based inference service
- **Containerization**: Docker support for easy deployment
- **Workflow Orchestration**: Apache Airflow for pipeline automation
- **CI/CD**: GitHub Actions for automated testing
- **Cloud Deployment**: Ready for deployment on AWS EC2

## 🛠 Tech Stack

- **Python 3.11**
- **Machine Learning**: scikit-learn, pandas, numpy
- **API Framework**: FastAPI, Uvicorn
- **Data Versioning**: DVC
- **Orchestration**: Apache Airflow
- **Containerization**: Docker, Docker Compose
- **Testing**: pytest, flake8
- **Encoding Detection**: chardet

## 📁 Project Structure

```
mlops-assignment/
├── data/                    # Dataset storage (versioned with DVC)
│   ├── dataset.csv         # Main dataset
│   └── dataset.csv.dvc     # DVC tracking file
├── models/                  # Trained models
│   └── model.pkl           # Serialized ML model
├── src/                     # Source code
│   └── train.py            # Model training script
├── api/                     # API service
│   └── main.py             # FastAPI application
├── tests/                   # Test suite
│   └── test_train.py       # Unit tests
├── dags/                    # Airflow DAGs
│   └── train_pipeline.py   # Training pipeline DAG
├── .github/workflows/       # CI/CD pipelines
│   └── ci.yml              # GitHub Actions workflow
├── Dockerfile              # Docker image definition
├── docker-compose.yaml     # Airflow setup
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🏁 Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Git
- (Optional) AWS CLI for cloud deployment

### Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/SobanSageer/mlops-assignment.git
   cd mlops-assignment
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up DVC**
   ```bash
   dvc init
   dvc add data/dataset.csv
   git add data/dataset.csv.dvc
   git commit -m "Add dataset"
   ```

5. **Train the model**
   ```bash
   python src/train.py
   ```

6. **Run tests**
   ```bash
   pytest -v
   ```

## 🔧 Usage

### Training Pipeline

Run the training script directly:
```bash
python src/train.py
```

Or use the Airflow DAG:
```bash
# Start Airflow
docker-compose up -d

# Trigger the DAG
docker-compose exec airflow-webserver airflow dags trigger train_pipeline
```

### API Service

Start the FastAPI server:
```bash
uvicorn api.main:app --reload
```

Access the API documentation at: `http://localhost:8000/docs`

### Docker Deployment

Build and run the API container:
```bash
docker build -t mlops-api .
docker run -p 8000:8000 mlops-api
```

## 📡 API Documentation

### Endpoints

#### GET `/health`
Returns the health status of the API.

**Response:**
```json
{
  "status": "healthy"
}
```

#### POST `/predict`
Makes a prediction based on patient data.

**Request Body:**
```json
{
  "age": 45.0,
  "bmi": 27.3,
  "blood_pressure": 130.0,
  "cholesterol": 210.0,
  "glucose": 98.0,
  "smoker": 1
}
```

**Response:**
```json
{
  "prediction": 1,
  "probability": [0.2, 0.8]
}
```

### Testing the API

Using cURL:
```bash
# Health check
curl http://localhost:8000/health

# Make prediction
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "age": 45,
       "bmi": 27.3,
       "blood_pressure": 130,
       "cholesterol": 210,
       "glucose": 98,
       "smoker": 1
     }'
```

## 🚀 Deployment

### Local Docker

```bash
# Build image
docker build -t mlops-api .

# Run container
docker run -p 8000:8000 mlops-api
```

### Cloud Deployment (AWS EC2)

1. **Launch EC2 instance**
   - Ubuntu 20.04 LTS
   - t2.micro (free tier)
   - Security group: Allow SSH (22) and HTTP (80), Custom TCP (8000)

2. **Connect to EC2**
   ```bash
   ssh -i your-key.pem ubuntu@your-ec2-ip
   ```

3. **Install Docker on EC2**
   ```bash
   sudo apt update -y
   sudo apt install docker.io -y
   sudo systemctl start docker
   sudo systemctl enable docker
   sudo usermod -aG docker ubuntu
   # Logout and login again
   ```

4. **Deploy the application**
   ```bash
   # Pull Docker image
   docker pull soban04/mlops-api:v1

   # Run container
   docker run -d -p 8000:8000 soban04/mlops-api:v1
   ```

5. **Verify deployment**
   ```bash
   docker ps
   docker logs <container_id>
   ```

6. **Access the API**
   - Health: `http://your-ec2-ip:8000/health`
   - Docs: `http://your-ec2-ip:8000/docs`

## 🧪 Testing

Run the complete test suite:
```bash
pytest -v
```

Run with coverage:
```bash
pytest --cov=src --cov=api
```

Lint code:
```bash
flake8 src/ api/ tests/
```

## 🔄 CI/CD

The project includes GitHub Actions workflow (`.github/workflows/ci.yml`) that:
- Runs on push/PR to main branch
- Installs dependencies
- Lints code with flake8
- Runs tests with pytest
- Builds Docker image

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make changes and add tests
4. Run tests: `pytest -v`
5. Commit changes: `git commit -m "Add your feature"`
6. Push to branch: `git push origin feature/your-feature`
7. Create a Pull Request

## 📊 Model Details

- **Algorithm**: Logistic Regression
- **Features**: age, bmi, blood_pressure, cholesterol, glucose, smoker
- **Target**: Binary classification (0: healthy, 1: at risk)
- **Preprocessing**: Label encoding for categorical variables
- **Evaluation**: Accuracy, precision, recall metrics

## 📈 Dataset

The dataset contains patient health metrics:
- **patient_id**: Unique identifier
- **age**: Patient age
- **bmi**: Body Mass Index
- **blood_pressure**: Systolic blood pressure
- **cholesterol**: Cholesterol level
- **glucose**: Glucose level
- **smoker**: Smoking status (yes/no)
- **target**: Health outcome (0/1)

## 🐛 Troubleshooting

### Common Issues

1. **DVC errors**: Ensure DVC is initialized and data is committed
2. **Port conflicts**: Change port mapping if 8000 is in use
3. **Permission errors**: Use `sudo` for Docker commands if needed
4. **Encoding issues**: The API auto-detects file encoding

### Logs

- **Airflow logs**: `docker-compose logs airflow-scheduler`
- **API logs**: `docker logs <container_id>`
- **Training logs**: Check console output or Airflow UI

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Soban Sageer** - *Initial work* - [GitHub](https://github.com/SobanSageer)

## 🙏 Acknowledgments

- Apache Airflow documentation
- FastAPI documentation
- DVC documentation
- scikit-learn documentation

---

⭐ Star this repo if you find it helpful!