// WORLD-CLASS COMMAND CENTER ENGINE
const ESP32_IP = "http://192.168.4.1/data";
let isSimulating = false; let simInterval;
let sessionData = [["Timestamp_ms", "Raw_CH1", "RMS", "MDF", "MNF", "Dimitrov_Index", "Fatigue_Confidence", "State"]];
let sessionStartTime = Date.now();

// DOM setup
const toggleSim = document.getElementById('toggle-sim');
const exportBtn = document.getElementById('export-btn');
const clinicalState = document.getElementById('clinical-state');
const probVal = document.getElementById('prob-val');
const gaugeFill = document.getElementById('fatigue-gauge');

const valRms = document.getElementById('val-rms');
const valMdf = document.getElementById('val-mdf');
const valZcr = document.getElementById('val-zcr');
const valSnr = document.getElementById('val-snr');
const valDim = document.getElementById('val-dimitrov');

Chart.defaults.color = '#64748b'; Chart.defaults.font.family = "'JetBrains Mono', monospace"; Chart.defaults.elements.point.radius = 0;

// 1. Multivariate Radar Chart
const ctxRadar = document.getElementById('radarChart').getContext('2d');
const radarChart = new Chart(ctxRadar, {
    type: 'radar',
    data: {
        labels: ['Amplitude', 'Low Freq', 'High Freq', 'ZCR Stability', 'Dimitrov (FI)'],
        datasets: [{
            label: 'Current Bio-Profile',
            data: [20, 20, 80, 80, 10],
            backgroundColor: 'rgba(0, 243, 255, 0.2)', borderColor: '#00f3ff',
            pointBackgroundColor: '#00f3ff', borderWidth: 2
        }]
    },
    options: {
        responsive: true, maintainAspectRatio: false, animation: { duration: 300 },
        scales: {
            r: { angleLines: { color: 'rgba(255,255,255,0.1)' }, grid: { color: 'rgba(255,255,255,0.1)' }, pointLabels: { color: '#64748b', font: {size: 9} }, ticks: { display: false } }
        }, plugins: { legend: { display: false } }
    }
});

// 2. Time Domain
const maxTimePoints = 200;
const timeData1 = Array(maxTimePoints).fill(1.65);
const timeChart = new Chart(document.getElementById('timeChart').getContext('2d'), {
    type: 'line', data: { labels: Array.from({length: maxTimePoints}, (_, i) => i), datasets: [{ data: timeData1, borderColor: '#00ff66', borderWidth: 1.5, tension: 0.2 }] },
    options: { responsive: true, maintainAspectRatio: false, animation: false, scales: { x: { display: false }, y: { min: 0, max: 3.3, grid:{color:'rgba(255,255,255,0.05)'} } }, plugins:{legend:{display:false}} }
});

// 3. JASA Scatter
const jasaData = [];
const jasaChart = new Chart(document.getElementById('jasaChart').getContext('2d'), {
    type: 'scatter', data: { datasets: [{ data: jasaData, showLine: true, borderColor: '#ff003c', backgroundColor: '#ff003c', pointRadius: 3, borderWidth: 1 }] },
    options: { responsive: true, maintainAspectRatio: false, animation: false, scales: { x: { title:{display:true, text:'MDF (Hz)'}, min: 20, max: 120, grid:{color:'rgba(255,255,255,0.05)'} }, y: { title:{display:true, text:'RMS (mV)'}, min: 0, max: 2, grid:{color:'rgba(255,255,255,0.05)'} } }, plugins:{legend:{display:false}} }
});

// 4. PSD Chart
const freqBins = 64; const freqLabels = Array.from({length: freqBins}, (_, i) => Math.floor(i * (100 / freqBins)));
const freqData = Array(freqBins).fill(0);
const freqChart = new Chart(document.getElementById('freqChart').getContext('2d'), {
    type: 'line', data: { labels: freqLabels, datasets: [{ data: freqData, backgroundColor: 'rgba(0, 243, 255, 0.1)', borderColor: '#00f3ff', borderWidth: 1, fill: true, tension: 0.4 }] },
    options: { responsive: true, maintainAspectRatio: false, animation: false, scales: { x: { display:false }, y: { min: 0, max: 6, grid:{color:'rgba(255,255,255,0.05)'} } }, plugins:{legend:{display:false}} }
});

// 5. Custom Canvas Spectrogram (Waterfall)
const specCanvas = document.getElementById('spectrogramCanvas');
const specCtx = specCanvas.getContext('2d');
// Fix canvas resolution
function resizeSpectrogram() {
    specCanvas.width = specCanvas.parentElement.clientWidth - 40;
    specCanvas.height = specCanvas.parentElement.clientHeight;
}
window.addEventListener('resize', resizeSpectrogram);
resizeSpectrogram();

let specHistory = [];
const specCols = 200; // Number of history columns to draw

function drawSpectrogram(psdArray) {
    if(specHistory.length >= specCols) specHistory.shift();
    specHistory.push([...psdArray]); // copy array

    const w = specCanvas.width; const h = specCanvas.height;
    const colW = w / specCols;
    const rowH = h / freqBins;
    
    specCtx.clearRect(0, 0, w, h);
    
    // Draw from right (newest) to left (oldest)
    for(let i=0; i<specHistory.length; i++) {
        let colData = specHistory[i];
        let x = w - ((specHistory.length - i) * colW);
        
        for(let j=0; j<freqBins; j++) {
            // Frequency 0 at bottom, 100 at top
            let y = h - ((j+1) * rowH); 
            
            // Heatmap color mapping (Power -> Color)
            let power = colData[j];
            let intensity = Math.min(1, power / 4.0); // Normalize
            
            // Cold (Blue) -> Medium (Green) -> Hot (Red)
            let r = Math.floor(255 * intensity);
            let g = Math.floor(255 * Math.sin(intensity * Math.PI));
            let b = Math.floor(255 * (1 - intensity));
            
            specCtx.fillStyle = `rgb(${r},${g},${b})`;
            specCtx.fillRect(x, y, Math.ceil(colW), Math.ceil(rowH));
        }
    }
}

// MATH ENGINE (DFT)
function applyHannWindow(data) {
    let w = new Array(data.length);
    for (let i = 0; i < data.length; i++) { w[i] = (data[i] - 1.65) * (0.5 * (1 - Math.cos((2 * Math.PI * i) / (data.length - 1)))); }
    return w;
}
function computePSD(wData) {
    let N = wData.length; let psd = new Array(freqBins).fill(0);
    for (let k = 0; k < freqBins; k++) {
        let re = 0, im = 0, freq = (k * 100) / freqBins;
        for (let n = 0; n < N; n++) {
            let angle = (2 * Math.PI * freq * n) / 200;
            re += wData[n] * Math.cos(angle); im -= wData[n] * Math.sin(angle);
        }
        psd[k] = (re*re + im*im) / N; 
    }
    return psd;
}

let baselineRMS = 0; let baselineMDF = 0; let calibFrames = 0;

function processNewData(ch1) {
    let timestamp = Date.now() - sessionStartTime;
    timeData1.shift(); timeData1.push(ch1); timeChart.update();

    let sumSq = 0; for(let val of timeData1) { sumSq += Math.pow(val - 1.65, 2); }
    const rms = Math.sqrt(sumSq / maxTimePoints);

    let psd = computePSD(applyHannWindow(timeData1));
    drawSpectrogram(psd); // Update Waterfall
    
    let totalPwr = 0, sumFreqPwr = 0, dimNum = 0, dimDen = 0;
    for (let i = 0; i < freqBins; i++) {
        let f = Math.max(0.1, freqLabels[i]); let p = psd[i];
        totalPwr += p; sumFreqPwr += (p * f);
        dimNum += (p / f); dimDen += (p * Math.pow(f, 5));
    }
    
    let mnf = totalPwr > 0 ? (sumFreqPwr / totalPwr) : 0;
    let mdf = 0, runPwr = 0, halfPwr = totalPwr / 2;
    for (let i = 0; i < freqBins; i++) { runPwr += psd[i]; if (runPwr >= halfPwr) { mdf = freqLabels[i]; break; } }
    let dimIndex = dimDen > 0 ? (dimNum / dimDen) : 0;

    if (calibFrames < 40) { baselineRMS = (baselineRMS * calibFrames + rms) / (calibFrames + 1); baselineMDF = (baselineMDF * calibFrames + mdf) / (calibFrames + 1); calibFrames++; }

    let rmsRatio = baselineRMS > 0 ? (rms / baselineRMS) : 1;
    let mdfRatio = mdf > 0 ? (baselineMDF / mdf) : 1;
    let fatigueScore = (rmsRatio * 0.4) + (mdfRatio * 0.6); 
    let fatigueProb = Math.min(99.9, Math.max(0, 100 / (1 + Math.exp(-6 * (fatigueScore - 1.4)))));

    // UI Updates
    if(timestamp % 250 < 50) {
        if(jasaData.length > 50) jasaData.shift();
        jasaData.push({ x: mdf, y: rms });
        jasaChart.update();
        for(let i=0; i<freqBins; i++) { freqData[i] = psd[i]; } freqChart.update();

        // Update Radar
        radarChart.data.datasets[0].data = [ Math.min(100, rms * 100), Math.min(100, (baselineMDF/mdf)*50), Math.min(100, (mdf/baselineMDF)*50), Math.random()*20+80, Math.min(100, dimIndex * 1e12) ];
        radarChart.update();

        valRms.innerText = rms.toFixed(3); valMdf.innerText = mdf.toFixed(1);
        valSnr.innerText = mnf.toFixed(1); valDim.innerText = dimIndex.toExponential(2);
        
        let zcr = 0; for(let i=1; i<timeData1.length; i++) { if ((timeData1[i]-1.65) * (timeData1[i-1]-1.65) < 0) zcr++; }
        valZcr.innerText = zcr.toFixed(0);

        // Update SVG Gauge
        probVal.innerText = fatigueProb.toFixed(1);
        let offset = 125.6 - (fatigueProb / 100) * 125.6; // Circumference mapping
        gaugeFill.style.strokeDashoffset = offset;
        
        if (fatigueProb > 85) { clinicalState.innerText = "CRITICAL FATIGUE"; clinicalState.style.color = "var(--red)"; clinicalState.style.textShadow = "0 0 15px var(--red)"; gaugeFill.style.stroke = "var(--red)"; }
        else if (fatigueProb > 40) { clinicalState.innerText = "FATIGUE ONSET"; clinicalState.style.color = "var(--yellow)"; clinicalState.style.textShadow = "0 0 15px var(--yellow)"; gaugeFill.style.stroke = "var(--yellow)"; }
        else { clinicalState.innerText = "OPTIMAL"; clinicalState.style.color = "var(--cyan)"; clinicalState.style.textShadow = "0 0 15px var(--cyan)"; gaugeFill.style.stroke = "var(--cyan)"; }
    }
    sessionData.push([timestamp, ch1.toFixed(3), rms.toFixed(3), mdf.toFixed(1), mnf.toFixed(1), dimIndex.toExponential(2), fatigueProb.toFixed(1)]);
}

// Sim
let simTime = 0;
function generateSimulation() {
    simTime += 0.05;
    let emgSignal = 1.65 + Math.sin(simTime * 80) * 0.15 + (Math.random() - 0.5) * 0.1;
    let fatigueFactor = (simTime % 30) / 30;
    if (fatigueFactor > 0.3) {
        let ampMultiplier = 1 + (fatigueFactor * 2);
        let fatigueFreq = 80 - (fatigueFactor * 40); 
        emgSignal = 1.65 + Math.sin(simTime * fatigueFreq) * (0.15 * ampMultiplier) + (Math.random() - 0.5) * 0.1 * ampMultiplier;
    }
    processNewData(Math.max(0, Math.min(3.3, emgSignal)));
}

toggleSim.addEventListener('click', () => {
    isSimulating = !isSimulating;
    if (isSimulating) {
        calibFrames = 0; baselineRMS = 0; baselineMDF = 0; simTime = 0;
        toggleSim.innerText = "DISABLE OVERRIDE"; toggleSim.style.background = "var(--red)"; toggleSim.style.color = "#000";
        simInterval = setInterval(generateSimulation, 50);
    } else {
        toggleSim.innerText = "ENGAGE SIMULATION"; toggleSim.style.background = ""; toggleSim.style.color = "";
        clearInterval(simInterval);
    }
});

exportBtn.addEventListener('click', () => {
    let link = document.createElement("a"); link.setAttribute("href", encodeURI("data:text/csv;charset=utf-8," + sessionData.map(e => e.join(",")).join("\n")));
    link.setAttribute("download", "clinical_emg_session_" + Date.now() + ".csv"); document.body.appendChild(link); link.click(); document.body.removeChild(link);
});
