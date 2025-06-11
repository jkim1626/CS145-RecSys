# CS145 Recommender Systems Competition Leaderboard

A Flask-based online leaderboard system for the CS145 Recommender Systems course competition. Students can submit their `submission.py` files and see how they rank against their classmates based on the `test_discounted_revenue` metric.

## Features

- 🚀 **Automated Evaluation**: Submit `submission.py` files and get automatic evaluation
- 🏆 **Real-time Leaderboard**: Live rankings based on test performance metrics
- 📊 **Comprehensive Metrics**: Multiple evaluation metrics including precision, NDCG, MRR, and hit rate
- 🔒 **Secure Execution**: Sandboxed evaluation environment for submitted code
- 📈 **Admin Dashboard**: Monitor all submissions, errors, and statistics
- 📱 **Responsive Design**: Works on desktop and mobile devices

## Quick Start

### Prerequisites

- Python 3.8+
- Java 8+ (required for PySpark)
- All course competition files (`data_generator.py`, `simulator.py`, `config.py`, etc.)

### Installation

1. **Clone the repository and navigate to the directory:**
   ```bash
   cd /path/to/your/competition/directory
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify you have all required course files:**
   - `evaluation.py` ✓
   - `data_generator.py` 
   - `simulator.py`
   - `config.py`
   - `sample_recommenders.py`

4. **Run the Flask application:**
   ```bash
   python app.py
   ```

5. **Open your browser and navigate to:**
   ```
   http://localhost:5431
   ```

## How It Works

### For Students

1. **Implement Your Recommender**: Create a `submission.py` file with your `MyRecommender` class
2. **Submit via Web Interface**: Upload your file with team name and email
3. **Automatic Evaluation**: Server runs your recommender against standardized test data
4. **View Results**: Check the leaderboard for your ranking and performance metrics

### For Instructors

1. **Monitor Progress**: Use the admin dashboard at `/admin` to view all submissions
2. **Debug Issues**: View detailed error messages for failed submissions
3. **Track Statistics**: Monitor success rates, unique teams, and common error patterns

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   File Upload   │───▶│   Evaluation    │───▶│   Database      │
│   (Flask Route) │    │   (Sandboxed)   │    │   (SQLite)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Interface │    │   Results JSON  │    │   Leaderboard   │
│   (HTML/CSS/JS) │    │   Processing    │    │   Display       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Security Features

- **Sandboxed Execution**: Each submission runs in an isolated temporary directory
- **Timeout Protection**: Evaluations are limited to 10 minutes to prevent infinite loops
- **File Validation**: Only `.py` files named exactly `submission.py` are accepted
- **Size Limits**: File uploads are limited to 16MB
- **Error Handling**: Robust error catching and reporting for malicious or broken code

## API Endpoints

### Web Routes
- `GET /` - Main submission page
- `POST /upload` - Handle file uploads
- `GET /leaderboard` - Full leaderboard view
- `GET /admin` - Admin dashboard
- `GET /api/leaderboard` - JSON API for leaderboard data

### Database Schema

The system uses SQLite with the following table structure:

```sql
CREATE TABLE submissions (
    id INTEGER PRIMARY KEY,
    team_name TEXT NOT NULL,
    email TEXT NOT NULL,
    submission_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    filename TEXT NOT NULL,
    test_discounted_revenue REAL,
    -- Additional metrics...
    status TEXT DEFAULT 'pending',
    error_message TEXT,
    raw_results TEXT
);
```

## Configuration

### Environment Variables

You can customize the application by modifying these variables in `app.py`:

- `UPLOAD_FOLDER`: Directory for uploaded files (default: 'uploads')
- `MAX_CONTENT_LENGTH`: Maximum file size (default: 16MB)
- `DATABASE`: SQLite database file (default: 'leaderboard.db')

### Evaluation Parameters

Modify the evaluation configuration in `evaluation.py`:

```python
config['data_generation']['n_users'] = 1000  # Number of users
config['data_generation']['n_items'] = 200   # Number of items
config['data_generation']['seed'] = 42       # Random seed
```

## Troubleshooting

### Common Issues

1. **PySpark Issues**:
   ```bash
   # Ensure Java is installed and JAVA_HOME is set
   export JAVA_HOME=/path/to/java
   ```

2. **Permission Errors**:
   ```bash
   # Make sure uploads directory is writable
   mkdir uploads
   chmod 755 uploads
   ```

3. **Database Errors**:
   ```bash
   # Reset the database if needed
   rm leaderboard.db
   python app.py  # Will recreate the database
   ```

### Logs and Debugging

- Flask debug mode is enabled by default
- Check console output for detailed error messages
- Use the admin dashboard to view submission errors
- PySpark logs are set to WARN level to reduce noise

## Deployment

### Production Deployment

For production deployment, make these changes:

1. **Update the secret key** in `app.py`:
   ```python
   app.secret_key = 'your-secure-random-secret-key'
   ```

2. **Disable debug mode**:
   ```python
   app.run(debug=False, host='0.0.0.0', port=5431)
   ```

3. **Use a production WSGI server** (e.g., Gunicorn):
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5431 app:app
   ```

4. **Set up a reverse proxy** (e.g., Nginx) for better performance

### Docker Deployment

Create a `Dockerfile`:

```dockerfile
FROM python:3.9-slim

# Install Java for PySpark
RUN apt-get update && apt-get install -y openjdk-11-jre-headless

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5431

CMD ["python", "app.py"]
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Support

For questions or issues:
- Check the troubleshooting section
- Review the admin dashboard for error details
- Contact the course staff

## License

This project is created for educational purposes as part of the CS145 course.
