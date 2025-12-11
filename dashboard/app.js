// --- 1. SENSOR CONFIGURATION ---
const SENSORS = [
    {
        id: 'max30003',
        name: 'ECG/HR',
        unit: 'BPM',
        color: '#37639C',
        min: 60, max: 100,
        notes: "Metrics: Heart rate, HRV\nClinical Notes: Stress load, panic onset, depression relapse, sleep quality, autonomic dysregulation, trauma reactivity, burnout"
    },
    {
        id: 'ad5940',
        name: 'EDA',
        unit: 'µS',
        color: '#558AC4',
        min: 0.5, max: 15,
        notes: "Metrics: Electrodermal activity\nClinical Notes: Anxiety, panic, stress, trauma activation, arousal, withdrawal"
    },
    {
        id: 'imu_resp',
        name: 'Respiratory Rate (IMU)',
        unit: 'rpm',
        color: '#558AC4',
        min: 12, max: 20,
        notes: "Metrics: Respiration rate/pattern\nClinical Notes: Panic, hyperventilation, PTSD hyperarousal, sleep disruption, anxiety spirals, medication side effects"
    },
    {
        id: 'tmp117',
        name: 'Skin Temperature',
        unit: '°C',
        color: '#37639C',
        min: 36.0, max: 37.5,
        notes: "Metrics: Skin temperature\nClinical Notes: Depression relapse, mania onset, insomnia severity, autonomic dysfunction"
    },
    {
        id: 'emg',
        name: 'EMG (Muscle Tension)',
        unit: 'µV',
        color: '#37639C',
        min: 0, max: 50,
        notes: "Metrics: Neck muscle tension\nClinical Notes: Tension, anxiety, panic, bruxism, medication side effects, agitation"
    },
    {
        id: 'bmi270',
        name: 'Micro-motion',
        unit: 'g',
        color: '#558AC4',
        min: 0, max: 2,
        notes: "Metrics: Movement, tremors, pacing, posture\nClinical Notes: Agitation, tremors, sleep position, neck posture → somatic stress"
    },
    {
        id: 'cortisol',
        name: 'Cortisol (Proxy)',
        unit: 'nmol/L',
        color: '#558AC4',
        min: 5, max: 25,
        notes: "Metrics: Local cortisol or proxy\nClinical Notes: Cortisol rhythms, morning cortisol, continuous stress hormone tracking"
    }
];

// --- 2. STATE MANAGEMENT ---
let currentScreen = 'splash-screen';
let currentSensorId = null;
let simulationInterval = null;
const HISTORY_LENGTH = 50; // Points to keep
let sensorData = {}; // Store history for each sensor

// Initialize Sensor Data Structures
SENSORS.forEach(s => {
    sensorData[s.id] = {
        history: Array(HISTORY_LENGTH).fill(s.min), // Init with baseline
        currentValue: s.min,
        chartInstance: null // Will hold the small chart instance
    };
});

let detailChartInstance = null;

// --- 3. DOM ELEMENTS ---
const screens = document.querySelectorAll('.screen');
const splashScreen = document.getElementById('splash-screen');
const sensorGrid = document.getElementById('sensor-grid');
const backBtn = document.getElementById('back-btn');
const detailTitle = document.getElementById('detail-title');
const detailNotes = document.getElementById('detail-notes');
const detailMetrics = document.getElementById('detail-metrics');

// --- 4. NAVIGATION ---
function showScreen(screenId) {
    screens.forEach(s => {
        s.classList.remove('active-screen');
        if (s.id === screenId) s.classList.add('active-screen');
    });
    currentScreen = screenId;
}

// Splash Click
splashScreen.addEventListener('click', () => {
    showScreen('dashboard-screen');
    if (!simulationInterval) startSimulation();
});

// Back Click
backBtn.addEventListener('click', () => {
    showScreen('dashboard-screen');
    currentSensorId = null;
});

// Dashboard Logo Click (Home)
document.getElementById('dashboard-home-btn').addEventListener('click', () => {
    showScreen('splash-screen');
});

// --- 5. INITIALIZATION & RENDERING ---
function init() {
    renderDashboardCards();
}

function renderDashboardCards() {
    sensorGrid.innerHTML = '';
    SENSORS.forEach(sensor => {
        const card = document.createElement('div');
        card.className = 'sensor-card';
        card.onclick = () => openDetail(sensor);

        const header = document.createElement('div');
        header.className = 'sensor-header';

        const name = document.createElement('div');
        name.className = 'sensor-name';
        name.innerText = sensor.name;

        const reading = document.createElement('div');
        reading.className = 'sensor-reading';
        reading.id = `val-${sensor.id}`;
        reading.innerText = `-- ${sensor.unit}`;

        header.appendChild(name);
        header.appendChild(reading);

        // Canvas for Sparkline
        const sparkContainer = document.createElement('div');
        sparkContainer.className = 'sparkline-container';
        const canvas = document.createElement('canvas');
        canvas.id = `chart-${sensor.id}`;
        sparkContainer.appendChild(canvas);

        const subtext = document.createElement('div');
        subtext.className = 'sensor-subtext';
        subtext.innerText = sensor.notes.split('\n')[0]; // First line only

        card.appendChild(header);
        card.appendChild(sparkContainer);
        card.appendChild(subtext);
        sensorGrid.appendChild(card);

        // Init Sparkline
        const ctx = canvas.getContext('2d');
        sensorData[sensor.id].chartInstance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: Array(HISTORY_LENGTH).fill(''),
                datasets: [{
                    data: sensorData[sensor.id].history,
                    borderColor: '#558AC4',
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.4,
                    fill: false
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { display: false },
                    y: { display: false } // Sparkline look
                },
                animation: false
            }
        });
    });
}

function openDetail(sensor) {
    currentSensorId = sensor.id;
    detailTitle.innerText = sensor.name;
    detailNotes.innerText = sensor.notes;

    // Render static metrics for now (could be dynamic)
    detailMetrics.innerHTML = `
        <div class="metric-item">
            <div class="metric-label">Min</div>
            <div class="metric-value">${sensor.min} ${sensor.unit}</div>
        </div>
         <div class="metric-item">
            <div class="metric-label">Max</div>
            <div class="metric-value">${sensor.max} ${sensor.unit}</div>
        </div>
         <div class="metric-item">
            <div class="metric-label">Current</div>
            <div class="metric-value" id="detail-val-current">--</div>
        </div>
         <div class="metric-item">
            <div class="metric-label">Status</div>
            <div class="metric-value" style="color:#22c55e">Normal</div>
        </div>
    `;

    // Init Detail Chart
    const ctx = document.getElementById('detailChart').getContext('2d');
    if (detailChartInstance) detailChartInstance.destroy();

    detailChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array(HISTORY_LENGTH).fill(''),
            datasets: [{
                label: sensor.name,
                data: sensorData[sensor.id].history,
                borderColor: '#37639C',
                backgroundColor: 'rgba(55, 99, 156, 0.1)',
                borderWidth: 3,
                pointRadius: 2,
                tension: 0.3,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { display: false },
                y: {
                    grid: { color: '#E2E8F0' },
                    suggestedMin: sensor.min * 0.9,
                    suggestedMax: sensor.max * 1.1
                }
            },
            animation: false
        }
    });

    showScreen('detail-screen');
}

// --- 6. SIMULATION ---
function startSimulation() {
    simulationInterval = setInterval(() => {
        SENSORS.forEach(sensor => {
            const data = sensorData[sensor.id];

            // Random Walk
            const range = sensor.max - sensor.min;
            let change = (Math.random() - 0.5) * (range * 0.1);
            let newVal = data.currentValue + change;

            // Constrain
            if (newVal < sensor.min) newVal = sensor.min;
            if (newVal > sensor.max) newVal = sensor.max;

            // Spike logic (occasional)
            if (Math.random() > 0.98) newVal = sensor.max * 1.05; // Spike

            data.currentValue = newVal;

            // Update History
            data.history.push(newVal);
            data.history.shift();

            // Update DOM & Chart if visible
            if (currentScreen === 'dashboard-screen') {
                // Update Value Text
                const el = document.getElementById(`val-${sensor.id}`);
                if (el) el.innerText = `${newVal.toFixed(1)} ${sensor.unit}`;

                // Update Sparkline
                if (data.chartInstance) {
                    data.chartInstance.update();
                }
            }
        });

        // Update Detail Chart if Active
        if (currentScreen === 'detail-screen' && currentSensorId) {
            const sData = sensorData[currentSensorId];
            if (detailChartInstance) {
                detailChartInstance.data.datasets[0].data = sData.history;
                detailChartInstance.update();

                // Update current value text in detail
                const detailVal = document.getElementById('detail-val-current');
                if (detailVal) detailVal.innerText = `${sData.currentValue.toFixed(1)} ${SENSORS.find(s => s.id === currentSensorId).unit}`;
            }
        }

    }, 200); // 5Hz update
}

// Start App
init();
