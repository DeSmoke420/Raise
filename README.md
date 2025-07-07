# 🤖 AI Forecast Intelligence Platform

A powerful, user-friendly web application for generating AI-powered forecasts using multiple machine learning models. Built with Flask, featuring a modern responsive UI and robust forecasting algorithms.

## ✨ Features

### 🎯 **Core Functionality**
- **Multi-Model Forecasting**: Holt-Winters, Prophet, and ARIMA models
- **Automatic Model Selection**: Chooses the best performing model based on statistical criteria
- **Flexible Data Input**: Supports CSV and Excel files with automatic column detection
- **Multiple Time Units**: Daily, weekly, and monthly forecasting
- **Professional Output**: Clean CSV/Excel downloads with model diagnostics
- **User Authentication**: Firebase Auth with Google login and email/password support
- **User Tracking**: Each forecast is associated with a unique user ID

### 🎨 **User Experience**
- **Modern Web Interface**: Responsive design with gradient backgrounds
- **Drag & Drop Upload**: Easy file upload with visual feedback
- **Data Preview**: See your data structure before processing
- **Progress Indicators**: Real-time feedback during forecast generation
- **Error Handling**: Helpful error messages and validation

### 🔧 **Technical Features**
- **Robust Data Processing**: Handles missing data, outliers, and various formats
- **Smart Column Detection**: Works with different column naming conventions
- **Production Ready**: Docker support, deployment guides, and security considerations
- **Diagnostics**: Detailed model performance information
- **Secure Authentication**: JWT tokens and Firebase Admin SDK integration
- **User Management**: Session management and user identification

## 🚀 Quick Start

### **Local Development**

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ai-forecast-platform.git
   cd ai-forecast-platform
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

4. **Set up Firebase Authentication** (Optional)
   ```bash
   # Follow the setup guide in FIREBASE_SETUP.md
   # Or run without authentication for testing
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Open your browser**
   Navigate to `http://localhost:5000`

### **Docker Deployment**

```bash
# Build and run with Docker
docker build -t forecast-app .
docker run -p 5000:5000 forecast-app

# Or use Docker Compose
docker-compose up -d
```

## 📊 Data Requirements

Your CSV/Excel file should contain these columns:

| Column Type | Examples | Required |
|-------------|----------|----------|
| **Date** | `Date`, `period`, `time`, `day` | ✅ |
| **Item ID** | `Item`, `product`, `sku`, `id`, `name` | ✅ |
| **Quantity** | `Quantity`, `sales`, `demand`, `forecast` | ✅ |

### **Example Data Format**
```csv
Date,Product,Quantity
2024-01-01,Product A,100
2024-01-02,Product A,120
2024-01-01,Product B,50
2024-01-02,Product B,55
```

## 🏗️ Architecture

### **Backend (Flask)**
- **Multiple ML Models**: Holt-Winters, Prophet, ARIMA
- **Automatic Model Selection**: Based on AIC/MSE metrics
- **Robust Data Validation**: Handles various data formats and issues
- **RESTful API**: Clean endpoints for forecast generation

### **Frontend (HTML/CSS/JavaScript)**
- **Modern UI**: Professional design with animations
- **Responsive Design**: Works on desktop and mobile
- **Real-time Feedback**: Progress indicators and status updates
- **File Handling**: Drag & drop with data preview

### **Deployment Ready**
- **Docker Support**: Containerized for easy deployment
- **Production WSGI**: Gunicorn for production servers
- **Security**: CORS configuration and input validation
- **Monitoring**: Comprehensive logging and error handling

## 🚀 Deployment Options

### **Cloud Platforms**
- **Railway**: One-click deployment from GitHub
- **Heroku**: Easy deployment with Git integration
- **Render**: Free tier with automatic deployments
- **Google Cloud Run**: Serverless container deployment

### **VPS/Server**
- **DigitalOcean**: $5/month droplets
- **AWS EC2**: Scalable cloud instances
- **Azure**: Microsoft cloud platform

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed deployment instructions.

## 🔧 Configuration

### **Environment Variables**
```bash
FLASK_ENV=production
FLASK_DEBUG=False

# Authentication (Optional)
FIREBASE_CONFIG={"type":"service_account",...}
JWT_SECRET=your-secret-key-here
```

### **Model Parameters**
- **Maximum Periods**: 120 (configurable)
- **File Size Limit**: 10MB
- **Row Limit**: 100,000 rows

## 📈 Model Information

### **Holt-Winters**
- **Best for**: Seasonal data with trends
- **Requirements**: At least 2 seasonal cycles
- **Strengths**: Handles seasonality well

### **Prophet**
- **Best for**: Time series with multiple seasonality
- **Requirements**: Date column, numeric values
- **Strengths**: Robust to missing data, handles holidays

### **ARIMA**
- **Best for**: Stationary time series
- **Requirements**: Sufficient data points
- **Strengths**: Statistical rigor, interpretable

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit your changes**: `git commit -m 'Add amazing feature'`
4. **Push to the branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Flask**: Web framework
- **Prophet**: Facebook's forecasting library
- **Statsmodels**: Statistical models
- **Pandas**: Data manipulation
- **Font Awesome**: Icons

## 📞 Support

- **Issues**: Report bugs on GitHub Issues
- **Documentation**: Check [DEPLOYMENT.md](DEPLOYMENT.md) for deployment help
- **Authentication Setup**: See [FIREBASE_SETUP.md](FIREBASE_SETUP.md) for detailed instructions
- **Questions**: Open a GitHub Discussion

---

**Made with ❤️ for the forecasting community** 