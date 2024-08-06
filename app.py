from flask import Flask, render_template, request, jsonify, send_file, url_for
from werkzeug.utils import secure_filename
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import io
import base64
import cv2
from PIL import Image
import logging
from flask_sqlalchemy import SQLAlchemy
from matplotlib.figure import Figure
from transformers import pipeline
import yake
from textblob import TextBlob
import pandasql as psql

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///data.db'
db = SQLAlchemy(app)

# Database model
class ExampleModel(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)

with app.app_context():
    db.create_all()

# In-memory data store for uploaded files
data_store = {}

# Initialize the text summarization pipeline
summarizer = pipeline("summarization")

# Home page route
@app.route('/')
def home():
    return render_template('home.html')

# Tabular data page route
@app.route('/tabular_data')
def tabular_data():
    return render_template('tabular_data.html')

# Image processing page route
@app.route('/image_processing')
def image_processing():
    return render_template('image_processing.html')

# Text analysis page route
@app.route('/text_analysis')
def text_analysis():
    return render_template('text_analysis.html')

# Helper function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv', 'xls', 'xlsx'}

# Helper function to create a timestamped filename
def create_timestamped_filename(filename):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    basename, extension = os.path.splitext(filename)
    return f"{basename}_{timestamp}{extension}"

# Helper function to preprocess data
def preprocess_data(df):
    df = df.apply(pd.to_numeric, errors='coerce')  # Convert to numeric, set errors to NaN
    df.ffill(inplace=True)  # Forward fill NaNs
    df.bfill(inplace=True)  # Backward fill NaNs
    return df

# Helper function to calculate statistics
def calculate_statistics(df):
    numeric_df = df.select_dtypes(include=[np.number])
    categorical_df = df.select_dtypes(include=['object', 'category'])

    stats = {'numeric': {}, 'categorical': {}}

    if not numeric_df.empty:
        stats['numeric'] = {
            'mean': numeric_df.mean().to_dict(),
            'std_dev': numeric_df.std().to_dict(),
            'variance': numeric_df.var().to_dict(),
            'skewness': numeric_df.skew().to_dict(),
            'median': numeric_df.median().to_dict(),
            'mode': numeric_df.mode().iloc[0].to_dict(),
            'quartiles': numeric_df.quantile([0.25, 0.5, 0.75]).to_dict(),
            'outliers': calculate_outliers(numeric_df)
        }

    for column in categorical_df.columns:
        stats['categorical'][column] = {
            'mode': categorical_df[column].mode().iloc[0] if not categorical_df[column].mode().empty else None,
            'unique_values': categorical_df[column].nunique()
        }

    return stats

# Helper function to calculate outliers
def calculate_outliers(df):
    outlier_results = {}
    for column in df.columns:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        filter = (df[column] < (q1 - 1.5 * iqr)) | (df[column] > (q3 + 1.5 * iqr))
        outlier_results[column] = df[column][filter].tolist()
    return outlier_results

# Helper function to convert NaNs to None in a dictionary
def convert_nan_to_none(obj):
    if isinstance(obj, dict):
        return {k: convert_nan_to_none(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_nan_to_none(i) for i in obj]
    elif isinstance(obj, float) and (pd.isna(obj) or obj != obj):  # Check for NaN
        return None
    else:
        return obj

# Route for uploading files
@app.route('/upload', methods=['POST'])
def file_upload():
    if 'files' not in request.files:
        return jsonify({'message': 'No files part in the request'}), 400
    files = request.files.getlist('files')
    if not files:
        return jsonify({'message': 'No files uploaded'}), 400

    filenames = []
    for f in files:
        if not allowed_file(f.filename):
            return jsonify({'message': 'File type not allowed'}), 400
        original_filename = secure_filename(f.filename)
        new_filename = create_timestamped_filename(original_filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
        f.save(save_path)
        df = pd.read_csv(save_path) if new_filename.endswith('.csv') else pd.read_excel(save_path)
        
        # Store both original and cleaned dataframes
        data_store[new_filename] = {
            "original": df,
            "cleaned": preprocess_data(df.copy())
        }
        filenames.append(new_filename)
    
    return jsonify({"filenames": filenames, "status": "Files successfully uploaded"}), 200

# Route for processing data
@app.route('/process/<filename>', methods=['GET'])
def process_data(filename):
    if filename not in data_store:
        return jsonify({'error': 'File not found'}), 404

    try:
        df = data_store[filename]["cleaned"]
        stats_type = request.args.get('stats', 'all')
        stats = calculate_statistics(df)

        stats = convert_nan_to_none(stats)

        return jsonify({'filename': filename, 'statistics': stats}), 200
    except Exception as e:
        return jsonify({'error': 'Error processing file: ' + str(e)}), 500

# Route for querying data
@app.route('/query/<filename>', methods=['POST'])
def query_data(filename):
    if filename not in data_store:
        logging.debug(f"Filename {filename} not found in data_store")
        return jsonify({'error': 'File not found'}), 404

    params = request.json
    if not params or 'query' not in params:
        logging.debug("No query provided in request")
        return jsonify({'error': 'No query provided'}), 400

    try:
        df = data_store[filename]["original"]
        query = params['query']
        logging.debug(f"Executing query: {query}")

        # Ensure the query is valid
        if not query.strip().lower().startswith("select"):
            return jsonify({'error': 'Only SELECT queries are allowed'}), 400

        result_df = psql.sqldf(query, locals())
        logging.debug(f"Query result: {result_df}")

        result_json = result_df.to_dict(orient='records')
        return jsonify(result_json), 200
    except Exception as e:
        logging.exception("Error executing query")
        return jsonify({'error': 'Error executing query: ' + str(e)}), 500

# Route for managing data (CRUD operations)
@app.route('/data/<filename>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def manage_data(filename):
    if filename not in data_store:
        return jsonify({'error': 'File not found'}), 404

    df = data_store[filename]["original"]

    if request.method == 'GET':
        return jsonify(convert_nan_to_none(df.to_dict(orient='records'))), 200

    elif request.method == 'POST':
        new_data = request.json
        try:
            new_df = pd.DataFrame([new_data])
            df = pd.concat([df, new_df], ignore_index=True)
            data_store[filename]["original"] = df
            return jsonify(convert_nan_to_none(df.to_dict(orient='records'))), 200
        except Exception as e:
            return jsonify({'error': 'Error adding data: ' + str(e)}), 500

    elif request.method == 'PUT':
        update_data = request.json
        try:
            for index, row in update_data.items():
                df.loc[int(index)] = row
            data_store[filename]["original"] = df
            return jsonify(convert_nan_to_none(df.to_dict(orient='records'))), 200
        except Exception as e:
            return jsonify({'error': 'Error updating data: ' + str(e)}), 500

    elif request.method == 'DELETE':
        delete_indices = request.json.get('indices')
        try:
            df.drop(index=delete_indices, inplace=True)
            data_store[filename]["original"] = df
            return jsonify(convert_nan_to_none(df.to_dict(orient='records'))), 200
        except Exception as e:
            return jsonify({'error': 'Error deleting data: ' + str(e)}), 500

# Route for visualizing data
@app.route('/visualize/<filename>', methods=['GET'])
def visualize_data(filename):
    if filename not in data_store:
        return jsonify({'error': 'File not found'}), 404

    try:
        df = data_store[filename]["original"]
        plot_type = request.args.get('plot_type', 'histogram')
        column = request.args.get('column')

        if column not in df.columns:
            return jsonify({'error': 'Column not found'}), 404

        plt.figure(figsize=(10, 6))

        if plot_type == 'histogram':
            df[column].hist()
            plt.title(f'Histogram of {column}')
        elif plot_type == 'bar':
            df[column].value_counts().plot(kind='bar')
            plt.title(f'Bar Chart of {column}')
        elif plot_type == 'box':
            df[column].plot(kind='box')
            plt.title(f'Box Plot of {column}')
        else:
            return jsonify({'error': 'Invalid plot type'}), 400

        plt.tight_layout()

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

        return jsonify({'plot_url': f'data:image/png;base64,{plot_url}'}), 200
    except Exception as e:
        logging.exception("Error visualizing data")
        return jsonify({'error': 'Error visualizing data: ' + str(e)}), 500

# Route for uploading images
@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'images' not in request.files:
        return jsonify({"error": "No images provided"}), 400

    files = request.files.getlist('images')
    file_info = []

    for file in files:
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            with Image.open(file_path) as img:
                width, height = img.size
            file_info.append({
                "filename": filename,
                "width": width,
                "height": height,
                "path": url_for('uploaded_file', filename=filename),
                "mode": img.mode,
                "format": img.format
            })

    return jsonify({"message": "Images successfully uploaded", "files": file_info}), 200

# Route for accessing uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

# Helper function to generate color histogram
def generate_histogram(image_path):
    image = cv2.imread(image_path)
    color = ('b', 'g', 'r')
    figure = Figure()
    axis = figure.add_subplot(1, 1, 1)

    for i, col in enumerate(color):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        axis.plot(hist, color=col)

    output_path = os.path.join(app.config['RESULT_FOLDER'], os.path.basename(image_path) + '_histogram.png')
    figure.savefig(output_path)

    return output_path

# Route for generating color histogram
@app.route('/histogram/<filename>', methods=['GET'])
def get_histogram(filename):
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    histogram_path = generate_histogram(image_path)
    return send_file(histogram_path, mimetype='image/png')

# Helper function to generate segmentation mask
def generate_segmentation_mask(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    output_path = os.path.join(app.config['RESULT_FOLDER'], os.path.basename(image_path) + '_segmentation.png')
    cv2.imwrite(output_path, mask)
    
    return output_path

# Route for generating segmentation mask
@app.route('/segmentation/<filename>', methods=['GET'])
def get_segmentation_mask(filename):
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    mask_path = generate_segmentation_mask(image_path)
    return send_file(mask_path, mimetype='image/png')

# Route for resizing images
@app.route('/resize/<filename>', methods=['POST'])
def resize_image(filename):
    width = request.json.get('width')
    height = request.json.get('height')
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(image_path):
        return jsonify({"error": "File not found"}), 404

    try:
        image = Image.open(image_path)
        resized_image = image.resize((width, height))
        img_io = io.BytesIO()
        resized_image.save(img_io, 'PNG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route for cropping images
@app.route('/crop/<filename>', methods=['POST'])
def crop_image(filename):
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image = Image.open(image_path)
    left = request.json.get('left')
    top = request.json.get('top')
    right = request.json.get('right')
    bottom = request.json.get('bottom')

    if left < 0 or top < 0 or right > image.width or bottom > image.height or left >= right or top >= bottom:
        logging.error("Invalid crop coordinates")
        return jsonify({"error": "Invalid crop coordinates"}), 400

    cropped_image = image.crop((left, top, right, bottom))
    
    output = io.BytesIO()
    cropped_image.save(output, format='PNG')
    output.seek(0)
    
    return send_file(output, mimetype='image/png')

# Route for converting image formats
@app.route('/convert/<filename>', methods=['POST'])
def convert_image(filename):
    logging.debug(f"Received request to convert {filename}")
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(image_path):
        logging.error(f"File {filename} not found at path: {image_path}")
        return jsonify({"error": "File not found"}), 404

    try:
        image = Image.open(image_path)
        logging.debug(f"Image {filename} opened successfully")
    except Exception as e:
        logging.error(f"Error opening image {filename}: {e}")
        return jsonify({"error": f"Error opening image: {str(e)}"}), 500

    format = request.json.get('format').lower()
    logging.debug(f"Requested format: {format}")

    valid_formats = ["jpeg", "png", "bmp", "gif", "tiff"]
    if format not in valid_formats:
        logging.error(f"Unsupported format: {format}")
        return jsonify({"error": f"Unsupported format: {format}"}), 400

    try:
        output_path = os.path.join(app.config['RESULT_FOLDER'], f"{os.path.splitext(os.path.basename(image_path))[0]}.{format}")
        image.save(output_path, format=format.upper())
        logging.debug(f"Image saved in format {format} at path: {output_path}")
        return send_file(output_path, mimetype=f'image/{format}')
    except KeyError as e:
        logging.error(f"Error saving image in format {format}: {e}")
        return jsonify({"error": f"Error saving image in format: {format}"}), 500
    except Exception as e:
        logging.error(f"General error converting image: {e}")
        return jsonify({"error": f"General error converting image: {str(e)}"}), 500

# Route for text summarization
@app.route('/summarize', methods=['POST'])
def summarize():
    text = request.form['text']
    summary = summarizer(text, max_length=50, min_length=25, do_sample=False)
    return jsonify({'summary': summary[0]['summary_text'], 'input_text_summary': text})

# Route for keyword extraction
@app.route('/keywords', methods=['POST'])
def extract_keywords():
    text = request.form['text']
    kw_extractor = yake.KeywordExtractor()
    keywords = kw_extractor.extract_keywords(text)
    keywords_list = [kw[0] for kw in keywords]
    return jsonify({'keywords': keywords_list, 'input_text_keywords': text})

# Route for sentiment analysis
@app.route('/sentiment', methods=['POST'])
def sentiment_analysis():
    text = request.form['text']
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    if sentiment_score > 0.1:
        sentiment = "Positive"
    elif sentiment_score < -0.1:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    return jsonify({'sentiment': f"{sentiment} ({sentiment_score})", 'input_text_sentiment': text})

# Route for text search
@app.route('/search', methods=['POST'])
def search_text():
    text = request.form['text']
    search_term = request.form['search_term']
    if search_term in text:
        result = "Found"
    else:
        result = "Not Found"
    return jsonify({'result': result, 'input_text_search': text, 'search_term': search_term})

# Route for text categorization
@app.route('/categorize', methods=['POST'])
def categorize_text():
    text = request.form['text']
    # Simple categorization based on the presence of certain keywords
    if 'good' in text or 'happy' in text:
        category = 'Positive'
    elif 'bad' in text or 'sad' in text:
        category = 'Negative'
    else:
        category = 'Neutral'
    return jsonify({'category': category, 'input_text_categorize': text})

# Route for custom text query
@app.route('/custom_query', methods=['POST'])
def custom_query():
    text = request.form['text']
    query = request.form['query']
    # For simplicity, let's just count the number of occurrences of the query in the text
    occurrences = text.count(query)
    return jsonify({'occurrences': occurrences, 'input_text_query': text, 'query': query})

# Error handler for file too large
@app.errorhandler(413)
def too_large(e):
    return jsonify({'message': 'The file is too large'}), 413

# Error handler for internal server error
@app.errorhandler(500)
def internal_error(e):
    return jsonify({'message': 'Internal server error'}), 500

# Main entry point for running the application
if __name__ == "__main__":
    app.run(port=5003, debug=True)
