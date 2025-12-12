document.addEventListener('DOMContentLoaded', () => {
    const generateBtn = document.getElementById('generateBtn');
    const startIndexInput = document.getElementById('startIndex');
    const allowOverlapCheckbox = document.getElementById('allowOverlap');
    const imageContainer = document.getElementById('imageContainer');
    const totalImagesSpan = document.getElementById('totalImages');
    
    const trainBtn = document.getElementById('trainBtn');
    const modelStructureDiv = document.getElementById('modelStructure');
    const predictBtn = document.getElementById('predictBtn');
    const predictionContainer = document.getElementById('predictionContainer');

    let dataset = [];
    let model;
    const DATASET_SIZE = 1000;
    const GRID_SIZE = 24;
    const PREVIEW_SIZE = 6;

    function generateSample(allowOverlap) {
        const grid = Array(GRID_SIZE).fill(0).map(() => Array(GRID_SIZE).fill(0));
        const numPoints = Math.floor(Math.random() * 9); // 0 to 8 points
        const points = [];

        for (let i = 0; i < numPoints; i++) {
            const radius = Math.random() * 3 + 1; // Radius between 1 and 4
            let x, y, validPosition;

            let attempts = 0;
            do {
                validPosition = true;
                x = Math.floor(Math.random() * GRID_SIZE);
                y = Math.floor(Math.random() * GRID_SIZE);

                if (!allowOverlap) {
                    for (const p of points) {
                        const distance = Math.sqrt((x - p.x) ** 2 + (y - p.y) ** 2);
                        if (distance <= radius + p.radius + 1.5) { // Not allowing adjacency or overlap
                            validPosition = false;
                            break;
                        }
                    }
                }
                attempts++;
            } while (!validPosition && attempts < 100);

            if (validPosition) {
                points.push({ x, y, radius });
                for (let r = 0; r < GRID_SIZE; r++) {
                    for (let c = 0; c < GRID_SIZE; c++) {
                        if (Math.sqrt((c - x) ** 2 + (r - y) ** 2) <= radius) {
                            grid[r][c] = 1;
                        }
                    }
                }
            }
        }
        return { grid, label: points.length };
    }

    function generateDataset() {
        dataset = [];
        const allowOverlap = allowOverlapCheckbox.checked;
        for (let i = 0; i < DATASET_SIZE; i++) {
            dataset.push(generateSample(allowOverlap));
        }
        totalImagesSpan.textContent = `(${dataset.length} samples generated)`;
        renderDatasetPreview();
    }

    function renderDatasetPreview() {
        imageContainer.innerHTML = '';
        const startIndex = parseInt(startIndexInput.value, 10);

        for (let i = 0; i < PREVIEW_SIZE; i++) {
            const imageIndex = startIndex + i;
            if (imageIndex >= dataset.length) break;

            const grid = dataset[imageIndex].grid;
            const canvas = document.createElement('canvas');
            canvas.width = GRID_SIZE * 10;
            canvas.height = GRID_SIZE * 10;
            const ctx = canvas.getContext('2d');
            ctx.imageSmoothingEnabled = false;

            for (let r = 0; r < GRID_SIZE; r++) {
                for (let c = 0; c < GRID_SIZE; c++) {
                    ctx.fillStyle = grid[r][c] === 1 ? 'black' : 'white';
                    ctx.fillRect(c * 10, r * 10, 10, 10);
                }
            }
            imageContainer.appendChild(canvas);
        }
    }

    async function createModel() {
        const model = tf.sequential();
        model.add(tf.layers.conv2d({
            inputShape: [GRID_SIZE, GRID_SIZE, 1],
            kernelSize: 3,
            filters: 16,
            activation: 'relu'
        }));
        model.add(tf.layers.maxPooling2d({poolSize: 2, strides: 2}));
        model.add(tf.layers.conv2d({
            kernelSize: 3,
            filters: 32,
            activation: 'relu'
        }));
        model.add(tf.layers.maxPooling2d({poolSize: 2, strides: 2}));
        model.add(tf.layers.flatten());
        model.add(tf.layers.dense({units: 64, activation: 'relu'}));
        model.add(tf.layers.dense({units: 1}));

        model.compile({
            optimizer: 'adam',
            loss: 'meanSquaredError',
            metrics: ['mse']
        });

        return model;
    }

    async function trainModel() {
        if (dataset.length === 0) return;
        trainBtn.disabled = true;
        trainBtn.textContent = 'Training...';
        
        model = await createModel();
        
        // Show model summary
        modelStructureDiv.innerHTML = '';
        tfvis.show.modelSummary(modelStructureDiv, model);

        // Prepare data
        const inputs = dataset.map(d => d.grid);
        const labels = dataset.map(d => d.label);

        const trainInputs = inputs.slice(0, 800);
        const trainLabels = labels.slice(0, 800);
        const valInputs = inputs.slice(800, 1000);
        const valLabels = labels.slice(800, 1000);

        const trainTensor = tf.tensor(trainInputs).expandDims(3);
        const trainLabelTensor = tf.tensor2d(trainLabels, [800, 1]);
        const valTensor = tf.tensor(valInputs).expandDims(3);
        const valLabelTensor = tf.tensor2d(valLabels, [200, 1]);

        // Train
        await model.fit(trainTensor, trainLabelTensor, {
            epochs: 50,
            validationData: [valTensor, valLabelTensor],
            callbacks: tfvis.show.fitCallbacks(
                document.getElementById('trainingGraph'),
                ['loss', 'val_loss'],
                { height: 200, width: 800, callbacks: ['onEpochEnd'] }
            )
        });
        
        trainTensor.dispose();
        trainLabelTensor.dispose();
        valTensor.dispose();
        valLabelTensor.dispose();
        
        trainBtn.disabled = false;
        trainBtn.textContent = 'Train Model';
        alert('Training Complete!');
    }

    async function renderTestPredictions() {
        if (!model) {
            alert('Please train the model first.');
            return;
        }
        predictionContainer.innerHTML = '';
        const allowOverlap = allowOverlapCheckbox.checked;
        
        for (let i = 0; i < PREVIEW_SIZE; i++) {
            const data = generateSample(allowOverlap);
            const grid = data.grid;
            const actual = data.label;

            // Predict
            const inputTensor = tf.tensor(grid).expandDims(0).expandDims(3);
            const prediction = model.predict(inputTensor).dataSync()[0];
            inputTensor.dispose();

            const container = document.createElement('div');
            container.style.display = 'flex';
            container.style.flexDirection = 'column';
            container.style.alignItems = 'center';

            const canvas = document.createElement('canvas');
            canvas.width = GRID_SIZE * 10;
            canvas.height = GRID_SIZE * 10;
            const ctx = canvas.getContext('2d');
            ctx.imageSmoothingEnabled = false;

            for (let r = 0; r < GRID_SIZE; r++) {
                for (let c = 0; c < GRID_SIZE; c++) {
                    ctx.fillStyle = grid[r][c] === 1 ? 'black' : 'white';
                    ctx.fillRect(c * 10, r * 10, 10, 10);
                }
            }
            
            const info = document.createElement('div');
            info.innerHTML = `Actual: ${actual}<br>Pred: ${Math.round(prediction)} (${prediction.toFixed(2)})`;
            info.style.textAlign = 'center';
            info.style.marginTop = '5px';
            
            container.appendChild(canvas);
            container.appendChild(info);
            predictionContainer.appendChild(container);
        }
    }

    generateBtn.addEventListener('click', generateDataset);
    startIndexInput.addEventListener('change', renderDatasetPreview);
    trainBtn.addEventListener('click', trainModel);
    predictBtn.addEventListener('click', renderTestPredictions);
    
    // Initial generation
    generateDataset();
});
