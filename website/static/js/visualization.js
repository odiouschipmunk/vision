// Initialize 3D Court Visualization
function init3DCourt() {
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer();
    
    const container = document.getElementById('court3d');
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    // Court dimensions (in meters)
    const courtLength = 9.75;
    const courtWidth = 6.4;
    const courtHeight = 4.57;

    // Create court floor
    const floorGeometry = new THREE.PlaneGeometry(courtWidth, courtLength);
    const floorMaterial = new THREE.MeshBasicMaterial({ color: 0xcccccc, side: THREE.DoubleSide });
    const floor = new THREE.Mesh(floorGeometry, floorMaterial);
    floor.rotation.x = Math.PI / 2;
    scene.add(floor);

    // Create court walls
    const wallMaterial = new THREE.MeshBasicMaterial({ color: 0xeeeeee, transparent: true, opacity: 0.5 });
    
    // Front wall
    const frontWallGeometry = new THREE.PlaneGeometry(courtWidth, courtHeight);
    const frontWall = new THREE.Mesh(frontWallGeometry, wallMaterial);
    frontWall.position.set(0, courtHeight/2, -courtLength/2);
    scene.add(frontWall);

    // Side walls
    const sideWallGeometry = new THREE.PlaneGeometry(courtLength, courtHeight);
    const leftWall = new THREE.Mesh(sideWallGeometry, wallMaterial);
    leftWall.rotation.y = Math.PI / 2;
    leftWall.position.set(-courtWidth/2, courtHeight/2, 0);
    scene.add(leftWall);

    const rightWall = new THREE.Mesh(sideWallGeometry, wallMaterial);
    rightWall.rotation.y = -Math.PI / 2;
    rightWall.position.set(courtWidth/2, courtHeight/2, 0);
    scene.add(rightWall);

    // Add player trails
    const player1Material = new THREE.LineBasicMaterial({ color: 0x0000ff });
    const player2Material = new THREE.LineBasicMaterial({ color: 0xff0000 });
    const ballMaterial = new THREE.LineBasicMaterial({ color: 0x00ff00 });

    function createTrail(positions, material) {
        const geometry = new THREE.BufferGeometry();
        const vertices = new Float32Array(positions.flat());
        geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
        return new THREE.Line(geometry, material);
    }

    if (positionsData.player1.length > 0) {
        const player1Trail = createTrail(positionsData.player1, player1Material);
        scene.add(player1Trail);
    }

    if (positionsData.player2.length > 0) {
        const player2Trail = createTrail(positionsData.player2, player2Material);
        scene.add(player2Trail);
    }

    if (positionsData.ball.length > 0) {
        const ballTrail = createTrail(positionsData.ball, ballMaterial);
        scene.add(ballTrail);
    }

    // Position camera
    camera.position.set(courtWidth, courtHeight * 1.5, courtLength);
    camera.lookAt(0, 0, 0);

    // Animation
    function animate() {
        requestAnimationFrame(animate);
        renderer.render(scene, camera);
    }
    animate();

    // Handle window resize
    window.addEventListener('resize', () => {
        const width = container.clientWidth;
        const height = container.clientHeight;
        renderer.setSize(width, height);
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
    });
}

// Initialize Shot Distribution Chart
function initShotDistribution() {
    const ctx = document.getElementById('shotDistribution').getContext('2d');
    new Chart(ctx, {
        type: 'pie',
        data: {
            labels: Object.keys(shotTypes),
            datasets: [{
                data: Object.values(shotTypes),
                backgroundColor: [
                    '#FF6384',
                    '#36A2EB',
                    '#FFCE56',
                    '#4BC0C0',
                    '#9966FF'
                ]
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'right'
                },
                title: {
                    display: true,
                    text: 'Shot Distribution'
                }
            }
        }
    });
}

// Initialize Player Heatmaps
function initHeatmaps() {
    // Create heatmap for Player 1
    const player1Data = {
        x: positionsData.player1.map(pos => pos[0]),
        y: positionsData.player1.map(pos => pos[1]),
        type: 'histogram2d',
        colorscale: 'Hot'
    };

    const player1Layout = {
        title: 'Player 1 Position Heatmap',
        xaxis: { title: 'Court Width (m)' },
        yaxis: { title: 'Court Length (m)' }
    };

    Plotly.newPlot('player1Heatmap', [player1Data], player1Layout);

    // Create heatmap for Player 2
    const player2Data = {
        x: positionsData.player2.map(pos => pos[0]),
        y: positionsData.player2.map(pos => pos[1]),
        type: 'histogram2d',
        colorscale: 'Hot'
    };

    const player2Layout = {
        title: 'Player 2 Position Heatmap',
        xaxis: { title: 'Court Width (m)' },
        yaxis: { title: 'Court Length (m)' }
    };

    Plotly.newPlot('player2Heatmap', [player2Data], player2Layout);
}

// Initialize all visualizations when the page loads
document.addEventListener('DOMContentLoaded', () => {

    initShotDistribution();
    initHeatmaps();
}); 