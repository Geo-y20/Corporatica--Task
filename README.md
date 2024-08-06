
# Corporatica Full-Stack Software Engineering Task

## Overview
This project is a comprehensive software application designed for advanced data analysis and manipulation. It features robust front-end and back-end components, capable of handling various data types including tabular data, RGB images, and textual data. The application includes functionalities for data processing, analysis, and visualization, implemented using Flask for the back-end and React for the front-end. The project is containerized using Docker and deployed with Kubernetes for scalability and resilience.

## Segmentation
This task is segmented for data analysts, computer vision engineers, and NLP engineers. It encompasses different aspects of data handling, including tabular data processing, image processing, and natural language processing (NLP).


## Table of Contents
- [Overview](#overview)
- [Time Allocation](#time-allocation)
- [Methodology](#methodology)
- [Technologies Used](#technologies-used)
- [Directory Structure](#directory-structure)
- [Installation and Setup](#installation-and-setup)
- [Module Details](#module-details)
  - [Tabular Data Processing](#tabular-data-processing)
  - [Image Processing](#image-processing)
  - [Text Analysis](#text-analysis)
- [Running the Application](#running-the-application)
- [Usage](#usage)
- [Additional Information](#additional-information)
  - [Contributions](#contributions)
  - [License](#license)

## Time Allocation
The project was completed over a span of five days, with the following time allocations:
- **Research**: 15 hours
- **Development**: 45 hours
- **Testing**: 24 hours
- **Deployment**: 6 hours

## Methodology
### Development Methodology
We followed an Agile development methodology, allowing for iterative development, continuous feedback, and incremental improvements. This approach ensured that we could quickly adapt to any changes and improve the application based on user feedback.

### Technology Selection Criteria
The selection of technologies, libraries, and frameworks was based on the following criteria:
- **Flask**: For its simplicity and ease of setting up a web server and handling HTTP requests.
- **React**: For creating a responsive and dynamic front-end interface.
- **Docker and Kubernetes**: For containerization and scalable deployment.
- **Pandas and SQLAlchemy**: For efficient data manipulation and ORM capabilities.
- **Pillow and OpenCV**: For comprehensive image processing tasks.
- **Transformers, Yake, and TextBlob**: For advanced NLP functionalities.

## Technologies Used
- **Back-End**: Flask, Python, Pandas, SQLAlchemy
- **Front-End**: React, HTML, CSS, JavaScript
- **Containerization and Deployment**: Docker, Docker Compose, Kubernetes
- **Data Processing and Visualization**: Matplotlib, Pillow, OpenCV
- **NLP Libraries**: Transformers (Hugging Face), Yake, TextBlob

## Directory Structure
```plaintext
Corporatica Back-End Developer Task Phase/
├── __pycache__/
├── instance/
│   ├── data.db
│   └── default.db
├── results/
├── static/
│   ├── scripts.js
│   └── styles.css
├── templates/
│   ├── home.html
│   ├── image_processing.html
│   ├── index.html
│   ├── tabular_data.html
│   └── text_analysis.html
├── uploads/
├── venv/
├── app.py
├── app2.py
├── requirements.txt
├── sample_1.jpg
├── sample_2.jpg
├── sample_3.jpeg
└── Updated_Dockerfile
└── updated_docker-compose.yml
```

## Installation and Setup

### Prerequisites
- Python 3.6 or higher
- Virtual environment tool (venv)
- Docker
- Docker Compose

### Steps
1. Clone the repository:
    ```bash
    git clone https://github.com/Geo-y20/Corporatica--Task
    cd Corporatica--Task
    ```
2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On macOS/Linux
    .\venv\Scripts\activate  # On Windows
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Build and run Docker containers:
    ```bash
    docker-compose up --build
    ```
5. Open your web browser and go to `http://localhost:5000/`.

## Module Details

### Tabular Data Processing

#### Overview
This module handles the uploading, processing, querying, and visualization of tabular data. Users can perform CRUD operations, run queries, and generate visualizations.

#### Routes and Endpoints
- **File Upload**: `/upload` (POST)
    - Uploads tabular data files (CSV, XLS, XLSX) to the server.
- **Process Data**: `/process/<filename>` (GET)
    - Processes the uploaded data and calculates statistics including mean, median, mode, quartiles, and outliers.
- **Query Data**: `/query/<filename>` (POST)
    - Allows users to run SQL-like queries on the data.
- **Manage Data**: `/data/<filename>` (GET, POST, PUT, DELETE)
    - Provides CRUD operations for the data.
- **Visualize Data**: `/visualize/<filename>` (GET)
    - Generates visualizations like histograms, bar charts, and box plots.

#### Detailed Code Snippets
- **File Upload and Preprocessing**
    ```python
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
    ```

- **Processing Data and Calculating Statistics**
    ```python
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
    ```

- **Querying Data**
    ```python
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
    ```

- **Managing Data**
    ```python
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
                    df.loc[int(index

)] = row
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
    ```

- **Visualizing Data**
    ```python
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
    ```

### Image Processing

#### Overview
This module allows users to upload images, generate color histograms, create segmentation masks, and perform image manipulations like resizing, cropping, and format conversion.

#### Routes and Endpoints
- **Image Upload**: `/upload_image` (POST)
    - Uploads image files to the server.
- **Get Histogram**: `/histogram/<filename>` (GET)
    - Generates and returns a color histogram for the uploaded image.
- **Get Segmentation Mask**: `/segmentation/<filename>` (GET)
    - Generates and returns a segmentation mask for the uploaded image.
- **Resize Image**: `/resize/<filename>` (POST)
    - Resizes the image to specified dimensions.
- **Crop Image**: `/crop/<filename>` (POST)
    - Crops the image to specified coordinates.
- **Convert Image Format**: `/convert/<filename>` (POST)
    - Converts the image to the specified format.

### Text Analysis

#### Overview
This module provides text summarization, keyword extraction, and sentiment analysis functionalities. It leverages various NLP libraries and models to process the input text.

#### Routes and Endpoints
- **Text Summarization**: `/summarize` (POST)
    - Summarizes the input text.
- **Keyword Extraction**: `/keywords` (POST)
    - Extracts keywords from the input text.
- **Sentiment Analysis**: `/sentiment` (POST)
    - Analyzes the sentiment of the input text.
 - **Text Search**: /search (POST)
    - Searches for a specific term within the input text.
- **Text Categorization**: /categorize (POST)
    - Categorizes the input text based on the presence of certain keywords.

### Detailed Code Snippets

#### Image Upload and Storage with Batch Processing
```python
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
```

#### Generating Color Histograms
```python
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

@app.route('/histogram/<filename>', methods=['GET'])
def get_histogram(filename):
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    histogram_path = generate_histogram(image_path)
    return send_file(histogram_path, mimetype='image/png')
```

#### Generating Segmentation Masks
```python
def generate_segmentation_mask(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    output_path = os.path.join(app.config['RESULT_FOLDER'], os.path.basename(image_path) + '_segmentation.png')
    cv2.imwrite(output_path, mask)
    
    return output_path

@app.route('/segmentation/<filename>', methods=['GET'])
def get_segmentation_mask(filename):
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    mask_path = generate_segmentation_mask(image_path)
    return send_file(mask_path, mimetype='image/png')
```

#### Image Manipulation Tasks
- **Resizing Image**
```python
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
```

- **Cropping Image**
```python
@app.route('/crop/<filename>', methods=['POST'])
def crop_image(filename):
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image = Image.open(image_path)
    left = request.json.get('left')
    top = request.json.get('top')
    right = request.json.get('right')
    bottom = request.json.get('bottom')

    if left < 0 or top < 0 or right > image.width or bottom > image.height or left >= right or top >= bottom:
        return jsonify({"error": "Invalid crop coordinates"}), 400

    cropped_image = image.crop((left, top, right, bottom))
    
    output = io.BytesIO()
    cropped_image.save(output, format='PNG')
    output.seek(0)
    
    return send_file(output, mimetype='image/png')
```

- **Converting Image Format**
```python
@app.route('/convert/<filename>', methods=['POST'])
def convert_image(filename):
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(image_path):
        return jsonify({"error": "File not found"}), 404

    try:
        image = Image.open(image_path)
    except Exception as e:
        return jsonify({"error": f"Error opening image: {str(e)}"}), 500

    format = request.json.get('format').lower()
    valid_formats = ["jpeg", "png", "bmp", "gif", "tiff"]
    if format not in valid_formats:
        return jsonify({"error": f"Unsupported format: {format}"}), 400

    try:
        output_path = os.path.join(app.config['RESULT_FOLDER'], f"{os.path.splitext(os.path.basename(image_path))[0]}.{format}")
        image.save(output_path, format=format.upper())
        return send_file(output_path, mimetype=f'image/{format}')
    except Exception as e:
        return jsonify({"error": f"Error converting image: {str(e)}"}), 500
```

### Text Summarization
```python
@app.route('/summarize', methods=['POST'])
def summarize():
    text = request.form['text']
    summary = summarizer(text, max_length=50, min_length=25, do_sample=False)
    return render_template('text_analysis.html', summary=summary[0]['summary_text'],

 input_text_summary=text)
```

### Keyword Extraction
```python
@app.route('/keywords', methods=['POST'])
def extract_keywords():
    text = request.form['text']
    kw_extractor = yake.KeywordExtractor()
    keywords = kw_extractor.extract_keywords(text)
    keywords_list = [kw[0] for kw in keywords]
    return render_template('text_analysis.html', keywords=keywords_list, input_text_keywords=text)
```

### Sentiment Analysis
```python
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
    return render_template('text_analysis.html', sentiment=f"{sentiment} ({sentiment_score})", input_text_sentiment=text)
```

## Running the Application
1. Activate the virtual environment:
    ```bash
    source venv/bin/activate  # On macOS/Linux
    .\venv\Scripts\activate  # On Windows
    ```
2. Run the Flask application:
    ```bash
    python app.py
    ```
3. Open your web browser and navigate to `http://localhost:5000/`.

## Usage

### Uploading Images
1. Click on the "Choose files" button and select one or more images to upload.
2. Click on the "Upload" button to upload the selected images.
3. View the details of the uploaded images such as filename, width, height, mode, and format.

### Generating Color Histograms
1. Enter the filename of the uploaded image in the input box.
2. Click on the "Get Histogram" button to generate and display the histogram.

### Generating Segmentation Masks
1. Enter the filename of the uploaded image in the input box.
2. Click on the "Get Segmentation Mask" button to generate and display the segmentation mask.

### Resizing Images
1. Enter the filename of the uploaded image in the input box.
2. Enter the desired width and height in the respective input boxes.
3. Click on the "Resize" button to resize and display the image.

### Cropping Images
1. Enter the filename of the uploaded image in the input box.
2. Enter the desired crop coordinates (left, top, right, bottom) in the respective input boxes.
3. Click on the "Crop" button to crop and display the image.

### Converting Image Formats
1. Enter the filename of the uploaded image in the input box.
2. Select the desired format from the dropdown menu.
3. Click on the "Convert" button to convert and display the image.

### Text Summarization
1. Enter the text to be summarized in the input box.
2. Click on the "Summarize" button to generate and display the summary.

### Keyword Extraction
1. Enter the text for keyword extraction in the input box.
2. Click on the "Extract Keywords" button to generate and display the keywords.

### Sentiment Analysis
1. Enter the text for sentiment analysis in the input box.
2. Click on the "Analyze Sentiment" button to generate and display the sentiment.


### Text Search
1. Enter the text to be searched in the input box.
2. Enter the search term in the input box.
3. Click on the "Search" button to find and display the search result.

### Text Categorization
1. Enter the text to be categorized in the input box.
2. Click on the "Categorize" button to generate and display the category.

## Sample of my work
![Sample of website](Sample%20of%20website/welcome.png)
![Sample of website](Sample%20of%20website/image.png)
![Sample of website](Sample%20of%20website/sample2.png)
![Sample of website](Sample%20of%20website/sample3.png)
