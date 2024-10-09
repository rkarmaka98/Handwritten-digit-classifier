const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let drawing = false;

canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mousemove', draw);

function getMousePos(canvas, event) {
    const rect = canvas.getBoundingClientRect();
    return {
        x: event.clientX - rect.left,
        y: event.clientY - rect.top
    };
}

function startDrawing(event) {
    drawing = true;
    const pos = getMousePos(canvas, event);
    ctx.beginPath();
    ctx.moveTo(pos.x, pos.y);
}

function stopDrawing() {
    drawing = false;
    ctx.beginPath();  // Reset the path to avoid random lines
}

function draw(event) {
    if (!drawing) return;
    const pos = getMousePos(canvas, event);
    ctx.lineWidth = 10;
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'black';
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(pos.x, pos.y);
}

document.getElementById('classifyButton').addEventListener('click', () => {
    canvas.toBlob(blob => {
        const formData = new FormData();
        formData.append('image', blob, 'digit.png');

        fetch('/classify', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            document.getElementById('result').innerText = `Prediction: ${data.prediction}`;
            displayActivations(data.activations);
        });
    });
});

document.getElementById('clearButton').addEventListener('click', () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    document.getElementById('result').innerText = '';
    document.getElementById('activations').innerHTML = '';
});

function displayActivations(activations) {
    const activationsDiv = document.getElementById('activations');
    activationsDiv.innerHTML = '';  // Clear previous activations

    activations.forEach((activation, layerIndex) => {
        const layerDiv = document.createElement('div');
        layerDiv.className = 'activation';
        layerDiv.innerHTML = `<h3>Layer ${layerIndex + 1}</h3>`;
        
        activation.forEach((featureMap, featureIndex) => {
            const featureCanvas = document.createElement('canvas');
            featureCanvas.width = 112;
            featureCanvas.height = 112;
            const featureCtx = featureCanvas.getContext('2d');
            const imageData = featureCtx.createImageData(112, 112);

            // Scale the feature map to fit the canvas
            const scale = 112 / featureMap.length;
            for (let y = 0; y < featureMap.length; y++) {
                for (let x = 0; x < featureMap[y].length; x++) {
                    const value = featureMap[y][x] * 255;
                    const index = (y * 112 + x) * 4;
                    imageData.data[index] = value;
                    imageData.data[index + 1] = value;
                    imageData.data[index + 2] = value;
                    imageData.data[index + 3] = 255;
                }
            }

            featureCtx.putImageData(imageData, 0, 0);
            layerDiv.appendChild(featureCanvas);
        });

        activationsDiv.appendChild(layerDiv);
    });
}