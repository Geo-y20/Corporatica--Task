// Function to handle form submissions with logging and error handling
function handleFormSubmission(formId, url, resultId) {
    $(formId).on('submit', function(e) {
        e.preventDefault();
        $.ajax({
            url: url,
            type: 'POST',
            data: $(this).serialize(),
            success: function(response) {
                $(resultId).html('<h3>Result:</h3><p>' + JSON.stringify(response) + '</p>');
                console.log('Response received:', response);
            },
            error: function(error) {
                console.error('Error occurred:', error);
                $(resultId).html('<h3>Error:</h3><p>An error occurred while processing your request. Please try again later.</p>');
            }
        });
    });
}

// Specific form handling for different functionalities
$(document).ready(function() {
    handleFormSubmission('#summarize-form', '/summarize', '#summary-result');
    handleFormSubmission('#keywords-form', '/keywords', '#keywords-result');
    handleFormSubmission('#sentiment-form', '/sentiment', '#sentiment-result');
    handleFormSubmission('#search-form', '/search', '#search-result');
    handleFormSubmission('#categorize-form', '/categorize', '#categorize-result');
    handleFormSubmission('#custom-query-form', '/custom_query', '#custom-query-result');

    // Handle t-SNE form submission with additional debugging
    $('#tsne-form').on('submit', function(e) {
        e.preventDefault();
        $.ajax({
            url: '/tsne',
            type: 'POST',
            data: $(this).serialize(),
            success: function(response) {
                if (response.error) {
                    $('#tsne-result').html('<h3>Error:</h3><p>' + response.error + '</p>');
                } else {
                    console.log('t-SNE Data Received:', response.tsne);
                    drawTSNE(response.tsne);
                }
            },
            error: function(error) {
                console.error('Error:', error);
                $('#tsne-result').html('<h3>Error:</h3><p>An error occurred while generating the t-SNE visualization. Please try again later.</p>');
            }
        });
    });
});

// Function to draw the t-SNE visualization
function drawTSNE(tsneData) {
    const canvas = document.getElementById('tsneCanvas');
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    const padding = 20;

    const xValues = tsneData.map(point => point[0]);
    const yValues = tsneData.map(point => point[1]);

    const xMin = Math.min(...xValues);
    const xMax = Math.max(...xValues);
    const yMin = Math.min(...yValues);
    const yMax = Math.max(...yValues);

    const normalizedData = tsneData.map(point => {
        return {
            x: ((point[0] - xMin) / (xMax - xMin)) * (width - 2 * padding) + padding,
            y: ((point[1] - yMin) / (yMax - yMin)) * (height - 2 * padding) + padding
        };
    });

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = 'rgba(0, 123, 255, 0.7)';

    normalizedData.forEach(point => {
        ctx.beginPath();
        ctx.arc(point.x, height - point.y, 4, 0, 2 * Math.PI);
        ctx.fill();
    });

    console.log('t-SNE plot drawn successfully');
}

// Function to handle file uploads for tabular data
function uploadFiles() {
    let formData = new FormData();
    let files = document.getElementById('files').files;
    for (let i = 0; i < files.length; i++) {
        formData.append('files', files[i]);
    }

    fetch('/upload', {
        method: 'POST',
        body: formData
    }).then(response => response.json()).then(data => {
        document.getElementById('upload-result').innerText = JSON.stringify(data, null, 2);
    }).catch(error => console.error('Error:', error));
}

// Function to process the uploaded data
function processData() {
    let filename = document.getElementById('process-filename').value;
    let stats = document.getElementById('stats').value;

    fetch(`/process/${filename}?stats=${stats}`, {
        method: 'GET'
    }).then(response => response.json()).then(data => {
        document.getElementById('process-result').innerText = JSON.stringify(data, null, 2);
    }).catch(error => console.error('Error:', error));
}

// Function to query data from the uploaded file
function queryData() {
    let filename = document.getElementById('query-filename').value;
    let query = document.getElementById('query').value;

    fetch(`/query/${filename}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ query: query })
    }).then(response => response.json()).then(data => {
        document.getElementById('query-result').innerText = JSON.stringify(data, null, 2);
    }).catch(error => console.error('Error:', error));
}

// Function to create new data entries in the uploaded file
function createData() {
    let filename = document.getElementById('manage-filename').value;
    let jsonData = document.getElementById('manage-data').value;

    fetch(`/data/${filename}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: jsonData
    }).then(response => response.json()).then(data => {
        document.getElementById('manage-result').innerText = JSON.stringify(data, null, 2);
    }).catch(error => console.error('Error:', error));
}

// Function to read data from the uploaded file
function readData() {
    let filename = document.getElementById('manage-filename').value;

    fetch(`/data/${filename}`, {
        method: 'GET'
    }).then(response => response.json()).then(data => {
        document.getElementById('manage-result').innerText = JSON.stringify(data, null, 2);
    }).catch(error => console.error('Error:', error));
}

// Function to update data entries in the uploaded file
function updateData() {
    let filename = document.getElementById('manage-filename').value;
    let jsonData = document.getElementById('manage-data').value;

    fetch(`/data/${filename}`, {
        method: 'PUT',
        headers: {
            'Content-Type': 'application/json'
        },
        body: jsonData
    }).then(response => response.json()).then(data => {
        document.getElementById('manage-result').innerText = JSON.stringify(data, null, 2);
    }).catch(error => console.error('Error:', error));
}

// Function to delete data entries from the uploaded file
function deleteData() {
    let filename = document.getElementById('manage-filename').value;
    let jsonData = document.getElementById('manage-data').value;

    fetch(`/data/${filename}`, {
        method: 'DELETE',
        headers: {
            'Content-Type': 'application/json'
        },
        body: jsonData
    }).then(response => response.json()).then(data => {
        document.getElementById('manage-result').innerText = JSON.stringify(data, null, 2);
    }).catch(error => console.error('Error:', error));
}

// Function to visualize data from the uploaded file
function visualizeData() {
    let filename = document.getElementById('visualize-filename').value;
    let plotType = document.getElementById('plot-type').value;
    let columnName = document.getElementById('column-name').value;

    fetch(`/visualize/${filename}?plot_type=${plotType}&column=${columnName}`, {
        method: 'GET'
    }).then(response => response.json()).then(data => {
        document.getElementById('visualize-result').innerHTML = `<img src="${data.plot_url}" alt="Plot">`;
    }).catch(error => console.error('Error:', error));
}

// Function to handle image uploads
document.getElementById('uploadForm').addEventListener('submit', function (e) {
    e.preventDefault();
    let formData = new FormData(this);
    fetch('/upload_image', {
        method: 'POST',
        body: formData
    }).then(response => response.json()).then(data => {
        document.getElementById('uploadedImages').innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
    }).catch(error => console.error('Error:', error));
});

// Function to get histogram of an image
function getHistogram() {
    let filename = document.getElementById('filename').value;
    fetch(`/histogram/${filename}`).then(response => response.blob()).then(blob => {
        let url = URL.createObjectURL(blob);
        document.getElementById('histogram').src = url;
        document.getElementById('histogram').style.display = 'block';
    }).catch(error => console.error('Error:', error));
}

// Function to get segmentation mask of an image
function getSegmentation() {
    let filename = document.getElementById('filename').value;
    fetch(`/segmentation/${filename}`).then(response => response.blob()).then(blob => {
        let url = URL.createObjectURL(blob);
        document.getElementById('segmentation').src = url;
        document.getElementById('segmentation').style.display = 'block';
    }).catch(error => console.error('Error:', error));
}

// Function to resize an image
function resizeImage() {
    let filename = document.getElementById('filename').value;
    let width = document.getElementById('resizeWidth').value;
    let height = document.getElementById('resizeHeight').value;
    fetch(`/resize/${filename}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ width: parseInt(width), height: parseInt(height) })
    }).then(response => response.blob()).then(blob => {
        let url = URL.createObjectURL(blob);
        document.getElementById('resized').src = url;
        document.getElementById('resized').style.display = 'block';
    }).catch(error => console.error('Error:', error));
}

// Function to crop an image
function cropImage() {
    let filename = document.getElementById('filename').value;
    let left = document.getElementById('cropLeft').value;
    let top = document.getElementById('cropTop').value;
    let right = document.getElementById('cropRight').value;
    let bottom = document.getElementById('cropBottom').value;
    fetch(`/crop/${filename}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            left: parseInt(left),
            top: parseInt(top),
            right: parseInt(right),
            bottom: parseInt(bottom)
        })
    }).then(response => response.blob()).then(blob => {
        let url = URL.createObjectURL(blob);
        document.getElementById('cropped').src = url;
        document.getElementById('cropped').style.display = 'block';
    }).catch(error => console.error('Error:', error));
}

// Function to convert an image to a different format
function convertImage() {
    let filename = document.getElementById('filename').value;
    let format = document.getElementById('convertFormat').value;
    fetch(`/convert/${filename}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ format: format })
    }).then(response => response.blob()).then(blob => {
        let url = URL.createObjectURL(blob);
        document.getElementById('converted').src = url;
        document.getElementById('converted').style.display = 'block';
    }).catch(error => console.error('Error:', error));
}
