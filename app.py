import os
import sys
import json
import sqlite3
import shutil
import traceback
import tempfile
import subprocess
from datetime import datetime
from werkzeug.utils import secure_filename
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'py'}
DATABASE = 'leaderboard.db'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Add custom Jinja2 filters
@app.template_filter('average')
def average_filter(values):
    """Calculate average of a list of values."""
    if not values:
        return 0
    return sum(values) / len(values)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init_database():
    """Initialize the SQLite database for storing leaderboard results."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS submissions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            team_name TEXT NOT NULL,
            email TEXT NOT NULL,
            submission_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            filename TEXT NOT NULL,
            test_discounted_revenue REAL,
            train_total_revenue REAL,
            test_total_revenue REAL,
            train_avg_revenue REAL,
            test_avg_revenue REAL,
            performance_change REAL,
            train_precision_at_k REAL,
            test_precision_at_k REAL,
            train_ndcg_at_k REAL,
            test_ndcg_at_k REAL,
            train_mrr REAL,
            test_mrr REAL,
            train_hit_rate REAL,
            test_hit_rate REAL,
            status TEXT DEFAULT 'pending',
            error_message TEXT,
            raw_results TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

def safe_evaluate_submission(submission_path, team_name):
    """
    Safely evaluate a submission by copying it and running evaluation in an isolated environment.
    """
    try:
        # Create temporary directory for evaluation
        with tempfile.TemporaryDirectory() as temp_dir:
            # Copy necessary files to temp directory
            temp_submission = os.path.join(temp_dir, 'submission.py')
            shutil.copy2(submission_path, temp_submission)
            
            # Copy evaluation script and dependencies to temp directory
            shutil.copy2('evaluation.py', temp_dir)
            
            # Copy other required files (assuming they exist in current directory)
            required_files = ['data_generator.py', 'simulator.py', 'config.py', 'sample_recommenders.py']
            for file in required_files:
                if os.path.exists(file):
                    shutil.copy2(file, temp_dir)
            
            # Copy required directories recursively
            required_dirs = ['sim4rec']
            for dir_name in required_dirs:
                if os.path.exists(dir_name) and os.path.isdir(dir_name):
                    shutil.copytree(dir_name, os.path.join(temp_dir, dir_name))
            
            # Create a modified evaluation script that returns results as JSON
            eval_script = f'''
import sys
import json
import os
sys.path.insert(0, "{temp_dir}")

# Import the evaluation function
from evaluation import run_evaluation

try:
    results = run_evaluation()
    
    # Extract key metrics
    output = {{
        "status": "success",
        "team_name": "{team_name}",
        "test_discounted_revenue": results.get("test_discounted_revenue", 0.0),
        "train_total_revenue": results.get("train_total_revenue", 0.0),
        "test_total_revenue": results.get("test_total_revenue", 0.0),
        "train_avg_revenue": results.get("train_avg_revenue", 0.0),
        "test_avg_revenue": results.get("test_avg_revenue", 0.0),
        "train_precision_at_k": results.get("train_precision_at_k", 0.0),
        "test_precision_at_k": results.get("test_precision_at_k", 0.0),
        "train_ndcg_at_k": results.get("train_ndcg_at_k", 0.0),
        "test_ndcg_at_k": results.get("test_ndcg_at_k", 0.0),
        "train_mrr": results.get("train_mrr", 0.0),
        "test_mrr": results.get("test_mrr", 0.0),
        "train_hit_rate": results.get("train_hit_rate", 0.0),
        "test_hit_rate": results.get("test_hit_rate", 0.0),
        "raw_results": str(results)
    }}
    
    print(json.dumps(output))
    
except Exception as e:
    error_output = {{
        "status": "error",
        "team_name": "{team_name}",
        "error_message": str(e),
        "traceback": __import__('traceback').format_exc()
    }}
    print(json.dumps(error_output))
'''
            
            # Write the evaluation script
            eval_script_path = os.path.join(temp_dir, 'run_eval.py')
            with open(eval_script_path, 'w') as f:
                f.write(eval_script)
            
            # Run the evaluation with timeout
            result = subprocess.run(
                [sys.executable, eval_script_path],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            if result.returncode == 0:
                # Parse the JSON output
                try:
                    output = json.loads(result.stdout.strip().split('\n')[-1])  # Get last line which should be JSON
                    return output
                except json.JSONDecodeError:
                    return {
                        "status": "error",
                        "team_name": team_name,
                        "error_message": f"Failed to parse evaluation output: {result.stdout[-500:]}"
                    }
            else:
                return {
                    "status": "error",
                    "team_name": team_name,
                    "error_message": f"Evaluation failed with return code {result.returncode}: {result.stderr[-500:]}"
                }
                
    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "team_name": team_name,
            "error_message": "Evaluation timed out after 10 minutes"
        }
    except Exception as e:
        return {
            "status": "error",
            "team_name": team_name,
            "error_message": f"Unexpected error: {str(e)}",
            "traceback": traceback.format_exc()
        }

def store_results(results, email, filename):
    """Store evaluation results in the database."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    if results["status"] == "success":
        # Calculate performance change
        train_avg = results.get("train_avg_revenue", 1)
        test_avg = results.get("test_avg_revenue", 1)
        performance_change = ((test_avg / train_avg) - 1) * 100 if train_avg > 0 else 0
        
        cursor.execute('''
            INSERT INTO submissions (
                team_name, email, filename,
                test_discounted_revenue, train_total_revenue, test_total_revenue,
                train_avg_revenue, test_avg_revenue, performance_change,
                train_precision_at_k, test_precision_at_k,
                train_ndcg_at_k, test_ndcg_at_k,
                train_mrr, test_mrr,
                train_hit_rate, test_hit_rate,
                status, raw_results
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            results["team_name"], email, filename,
            results.get("test_discounted_revenue", 0.0),
            results.get("train_total_revenue", 0.0),
            results.get("test_total_revenue", 0.0),
            results.get("train_avg_revenue", 0.0),
            results.get("test_avg_revenue", 0.0),
            performance_change,
            results.get("train_precision_at_k", 0.0),
            results.get("test_precision_at_k", 0.0),
            results.get("train_ndcg_at_k", 0.0),
            results.get("test_ndcg_at_k", 0.0),
            results.get("train_mrr", 0.0),
            results.get("test_mrr", 0.0),
            results.get("train_hit_rate", 0.0),
            results.get("test_hit_rate", 0.0),
            "success",
            results.get("raw_results", "")
        ))
    else:
        cursor.execute('''
            INSERT INTO submissions (
                team_name, email, filename, status, error_message
            ) VALUES (?, ?, ?, ?, ?)
        ''', (
            results["team_name"], email, filename, "error", results.get("error_message", "Unknown error")
        ))
    
    conn.commit()
    conn.close()

@app.route('/')
def index():
    """Display the main page with upload form and leaderboard."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and evaluation."""
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(request.url)
    
    file = request.files['file']
    team_name = request.form.get('team_name', '').strip()
    email = request.form.get('email', '').strip()
    
    if not team_name or not email:
        flash('Team name and email are required')
        return redirect(url_for('index'))
    
    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        if filename != 'submission.py':
            flash('File must be named "submission.py"')
            return redirect(url_for('index'))
        
        # Create unique filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{team_name}_{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        file.save(filepath)
        
        # Run evaluation
        flash(f'File uploaded successfully! Evaluating submission for {team_name}...')
        
        try:
            results = safe_evaluate_submission(filepath, team_name)
            store_results(results, email, unique_filename)
            
            if results["status"] == "success":
                flash(f'Evaluation completed successfully! Test Discounted Revenue: {results.get("test_discounted_revenue", 0.0):.4f}')
            else:
                flash(f'Evaluation failed: {results.get("error_message", "Unknown error")}')
                
        except Exception as e:
            flash(f'Error during evaluation: {str(e)}')
        
        return redirect(url_for('index'))
    else:
        flash('Invalid file type. Please upload a .py file.')
        return redirect(url_for('index'))

@app.route('/leaderboard')
def leaderboard():
    """Display the leaderboard."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT team_name, email, submission_time, test_discounted_revenue,
               test_total_revenue, test_avg_revenue, performance_change,
               test_precision_at_k, test_ndcg_at_k, test_mrr, test_hit_rate,
               status
        FROM submissions 
        WHERE status = 'success'
        ORDER BY test_discounted_revenue DESC
    ''')
    
    results = cursor.fetchall()
    conn.close()
    
    return render_template('leaderboard.html', results=results)

@app.route('/api/leaderboard')
def api_leaderboard():
    """API endpoint for leaderboard data."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT team_name, test_discounted_revenue, test_total_revenue,
               test_avg_revenue, submission_time, status
        FROM submissions 
        WHERE status = 'success'
        ORDER BY test_discounted_revenue DESC
        LIMIT 50
    ''')
    
    results = cursor.fetchall()
    conn.close()
    
    leaderboard_data = []
    for i, row in enumerate(results):
        leaderboard_data.append({
            'rank': i + 1,
            'team_name': row[0],
            'test_discounted_revenue': row[1],
            'test_total_revenue': row[2],
            'test_avg_revenue': row[3],
            'submission_time': row[4],
            'status': row[5]
        })
    
    return jsonify(leaderboard_data)

@app.route('/admin')
def admin():
    """Admin page to view all submissions including errors."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, team_name, email, submission_time, filename,
               test_discounted_revenue, status, error_message
        FROM submissions 
        ORDER BY submission_time DESC
    ''')
    
    results = cursor.fetchall()
    conn.close()
    
    return render_template('admin.html', results=results)

if __name__ == '__main__':
    init_database()
    app.run(debug=True, host='0.0.0.0', port=5431) 