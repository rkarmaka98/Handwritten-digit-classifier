document.addEventListener('DOMContentLoaded', function() {
    const canvas = document.getElementById('nnCanvas');
    const ctx = canvas.getContext('2d');

    // Neural network structure
    const layers = [4, 10, 3]; // Example: 3 layers with 4, 10, and 3 neurons respectively

    // Function to draw a neuron
    function drawNeuron(x, y, radius = 20, label = '') {
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, 2 * Math.PI);
        ctx.fillStyle = 'white';
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = 'black';
        ctx.fillText(label, x - radius / 2, y + radius / 2);
    }

    // Function to draw a connection
    function drawConnection(x1, y1, x2, y2, weight = '') {
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
        const midX = (x1 + x2) / 2;
        const midY = (y1 + y2) / 2;
        ctx.fillStyle = 'red';
        ctx.fillText(weight, midX, midY);
    }

    // Draw the neural network
    const layerDistance = canvas.width / (layers.length + 1);
    const neuronDistance = canvas.height / (Math.max(...layers) + 1);

    layers.forEach((layerSize, layerIndex) => {
        const x = (layerIndex + 1) * layerDistance;
        for (let i = 0; i < layerSize; i++) {
            const y = (i + 1) * neuronDistance;
            drawNeuron(x, y, 20, `N${layerIndex + 1}-${i + 1}`);

            // Draw connections to the next layer
            if (layerIndex < layers.length - 1) {
                const nextLayerSize = layers[layerIndex + 1];
                for (let j = 0; j < nextLayerSize; j++) {
                    const nextX = (layerIndex + 2) * layerDistance;
                    const nextY = (j + 1) * neuronDistance;
                    drawConnection(x, y, nextX, nextY, `W${layerIndex + 1}-${i + 1}-${j + 1}`);
                }
            }
        }
    });
});
