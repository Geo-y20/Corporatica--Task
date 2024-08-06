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

function processData() {
    let filename = document.getElementById('process-filename').value;
    let stats = document.getElementById('stats').value;

    fetch(`/process/${filename}?stats=${stats}`, {
        method: 'GET'
    }).then(response => response.json()).then(data => {
        document.getElementById('process-result').innerText = JSON.stringify(data, null, 2);
    }).catch(error => console.error('Error:', error));
}

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

function readData() {
    let filename = document.getElementById('manage-filename').value;

    fetch(`/data/${filename}`, {
        method: 'GET'
    }).then(response => response.json()).then(data => {
        document.getElementById('manage-result').innerText = JSON.stringify(data, null, 2);
    }).catch(error => console.error('Error:', error));
}

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

function getHistogram() {
    let filename = document.getElementById('filename').value;
    fetch(`/histogram/${filename}`).then(response => response.blob()).then(blob => {
        let url = URL.createObjectURL(blob);
        document.getElementById('histogram').src = url;
        document.getElementById('histogram').style.display = 'block';
    }).catch(error => console.error('Error:', error));
}

function getSegmentation() {
    let filename = document.getElementById('filename').value;
    fetch(`/segmentation/${filename}`).then(response => response.blob()).then(blob => {
        let url = URL.createObjectURL(blob);
        document.getElementById('segmentation').src = url;
        document.getElementById('segmentation').style.display = 'block';
    }).catch(error => console.error('Error:', error));
}

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
