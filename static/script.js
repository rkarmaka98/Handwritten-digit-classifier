const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let drawing = false;

canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mousemove', draw);

function startDrawing(event) {
    drawing = true;
    draw(event);  // Start drawing immediately
}

function stopDrawing() {
    drawing = false;
    ctx.beginPath();  // Reset the path to avoid random lines
}

function draw(event) {
    if (!drawing) return;
    ctx.lineWidth = 10;
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'black';
    ctx.lineTo(event.clientX - canvas.offsetLeft, event.clientY - canvas.offsetTop);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(event.clientX - canvas.offsetLeft, event.clientY - canvas.offsetTop);
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
        });
    });
});

document.getElementById('clearButton').addEventListener('click', () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    document.getElementById('result').innerText = '';
});