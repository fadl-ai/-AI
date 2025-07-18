// DOM Elements
const newsText = document.getElementById('newsText');
const analyzeBtn = document.getElementById('analyzeBtn');
const resultSection = document.getElementById('resultSection');
const confidenceBar = document.getElementById('confidenceBar');
const confidenceValue = document.getElementById('confidenceValue');
const aiAnalysis = document.getElementById('aiAnalysis');
const credibilityScore = document.getElementById('credibilityScore');
const warningSigns = document.getElementById('warningSigns');
const wordCount = document.getElementById('wordCount');
const exampleBtns = document.querySelectorAll('.example-btn');
const aboutLink = document.getElementById('aboutLink');
const aboutModal = document.getElementById('aboutModal');
const closeModal = document.querySelector('.close');
const newsForm = document.getElementById('newsForm');

// Example articles for testing
const examples = {
    real: `NASA's Perseverance Rover Successfully Lands on Mars

In a historic achievement for space exploration, NASA's Perseverance rover successfully landed on Mars on February 18, 2021. The landing took place at Jezero Crater, a 28-mile-wide basin that scientists believe was once home to an ancient river delta.

The rover, which launched from Earth on July 30, 2020, traveled 293 million miles to reach the Red Planet. The landing was confirmed by mission control at NASA's Jet Propulsion Laboratory in Southern California at 3:55 p.m. EST.

"This landing is one of those pivotal moments for NASA, the United States, and space exploration globally," said acting NASA Administrator Steve Jurczyk. "We know we're on the cusp of discovery and sharpening our pencils, so to speak, to rewrite the textbooks."

The Perseverance rover is the most sophisticated rover NASA has ever sent to Mars. It's equipped with 23 cameras, including the first color cameras to capture a Mars landing. The rover also carries a small helicopter named Ingenuity, which will attempt the first powered flight on another planet.

The mission's primary goal is to search for signs of ancient microbial life. The rover will collect rock and soil samples that will be returned to Earth by future missions. Scientists believe Jezero Crater is an ideal location for this search because it contains some of the oldest and most scientifically interesting landscapes on Mars.

"Landing on Mars is always an incredibly difficult task," said Jennifer Trosper, deputy project manager for the mission. "We're excited to be on the ground and ready to roll."`,

    fake: `SHOCKING DISCOVERY: Scientists Find MIRACLE CURE for All Diseases! Big Pharma is HIDING the Truth!

BREAKING NEWS: A group of independent researchers has made a MIND-BLOWING discovery that will change medicine forever! They've found a MIRACLE CURE that can treat ANY disease, but BIG PHARMA is trying to HIDE it from you!

The SECRET treatment, which costs only pennies to make, has been PROVEN to cure cancer, diabetes, heart disease, and even aging! Doctors HATE this simple trick that could save millions of lives!

"Mainstream media won't tell you about this," says Dr. John Smith, a self-proclaimed medical expert. "They're all in the pocket of Big Pharma! The government is covering up this amazing discovery because they don't want you to know the truth!"

The MIRACLE CURE is made from a combination of common household items that you can find in your kitchen! Just mix these ingredients together and take one spoonful daily to experience INCREDIBLE results!

Thousands of people have already tried this SECRET formula and reported AMAZING results! One woman claims she lost 50 pounds in just one week! Another man says his cancer completely disappeared after just three days!

But WAIT! There's more! This MIRACLE CURE also makes you look 20 years younger and gives you unlimited energy! You won't believe the TRANSFORMATION!

HURRY! This information might be REMOVED from the internet soon! Share this with everyone you know before it's too late! The medical establishment doesn't want you to know about this SECRET that could save your life!`
};

// Security configuration
const API_URL = 'http://127.0.0.1:5000';
const MAX_RETRIES = 3;
const RETRY_DELAY = 1000; // 1 second

// Input sanitization
function sanitizeInput(text) {
    return text.replace(/[<>]/g, ''); // Basic XSS prevention
}

// Check server connection
async function checkServerConnection() {
    try {
        const response = await fetch(`${API_URL}/analyze`, {
            method: 'OPTIONS',
            headers: {
                'Content-Type': 'application/json',
                'X-Requested-With': 'XMLHttpRequest'
            }
        });
        return response.ok;
    } catch (error) {
        console.error('Server connection check failed:', error);
        return false;
    }
}

// Error handling
function handleError(error, retryCount = 0) {
    console.error('Error:', error);
    
    if (error.message === 'Failed to fetch') {
        console.error('Connection error: Cannot reach the server. Please make sure the server is running.');
        resultSection.classList.remove('hidden');
        aiAnalysis.textContent = 'Server Connection Error';
        confidenceBar.style.width = '0%';
        confidenceValue.textContent = '0%';
        credibilityScore.textContent = 'Error';
        warningSigns.textContent = 'Cannot connect to the server. Please make sure the server is running at ' + API_URL;
        return;
    }
    
    if (retryCount < MAX_RETRIES) {
        console.log(`Retrying... Attempt ${retryCount + 1} of ${MAX_RETRIES}`);
        setTimeout(() => {
            analyzeNews(retryCount + 1);
        }, RETRY_DELAY);
        return;
    }
    
    // Show error in the UI
    resultSection.classList.remove('hidden');
    aiAnalysis.textContent = 'Error analyzing news. Please try again.';
    confidenceBar.style.width = '0%';
    confidenceValue.textContent = '0%';
    credibilityScore.textContent = 'Error';
    warningSigns.textContent = error.message || 'An error occurred while analyzing the text.';
}

// Analyze news with retry mechanism
async function analyzeNews(retryCount = 0) {
    const text = sanitizeInput(newsText.value.trim());
    
    if (text.length < 50) {
        alert('Please enter at least 50 words for better accuracy.');
        return;
    }

    try {
        // Check server connection first
        const isServerAvailable = await checkServerConnection();
        if (!isServerAvailable) {
            throw new Error('Failed to fetch');
        }

        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';

        console.log('Sending request to analyze text...');
        const response = await fetch(`${API_URL}/analyze`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-Requested-With': 'XMLHttpRequest'
            },
            body: JSON.stringify({ text })
        });

        console.log('Response received:', response.status);
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            if (response.status === 429) {
                throw new Error('Too many requests. Please wait a moment before trying again.');
            }
            throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        console.log('Analysis data received:', data);

        if (data.error) {
            throw new Error(data.error);
        }

        // Update UI with results
        resultSection.classList.remove('hidden');
        
        // Animate confidence bar
        const confidence = Math.round(data.confidence * 100);
        confidenceBar.style.width = `${confidence}%`;
        confidenceValue.textContent = `${confidence}%`;

        // Update analysis details
        aiAnalysis.textContent = data.is_fake ? 
            'The AI model has detected potential signs of fake news.' :
            'The AI model suggests this appears to be legitimate news.';

        credibilityScore.textContent = `Score: ${Math.round((1 + data.credibility_score) * 50)}/100`;

        // Compile warning signs
        const warnings = [];
        if (data.details.has_strong_indicators) warnings.push('Contains strong fake news indicators');
        if (data.details.has_moderate_indicators) warnings.push('Contains moderate fake news indicators');
        if (data.details.has_weak_indicators) warnings.push('Contains weak fake news indicators');
        if (data.details.style_analysis.exclamation_count > 2) warnings.push('Uses excessive punctuation');
        if (data.details.style_analysis.caps_words > 2) warnings.push('Contains excessive capitalization');
        
        warningSigns.textContent = warnings.length > 0 ? 
            warnings.join(', ') : 
            'No significant warning signs detected';

        // Scroll to results
        resultSection.scrollIntoView({ behavior: 'smooth' });

    } catch (error) {
        console.error('Analysis error:', error);
        handleError(error, retryCount);
    } finally {
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = '<i class="fas fa-search"></i> Analyze News';
    }
}

// Form submission with validation
newsForm.addEventListener('submit', (e) => {
    e.preventDefault();
    analyzeNews();
});

// Word count update with sanitization
newsText.addEventListener('input', () => {
    const sanitizedText = sanitizeInput(newsText.value);
    newsText.value = sanitizedText;
    const words = sanitizedText.trim().split(/\s+/).filter(word => word.length > 0);
    wordCount.textContent = words.length;
});

// Example buttons with sanitization
exampleBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        const type = btn.dataset.example;
        const exampleText = sanitizeInput(examples[type]);
        newsText.value = exampleText;
        wordCount.textContent = exampleText.trim().split(/\s+/).filter(word => word.length > 0).length;
    });
});

// Modal handling
aboutLink.addEventListener('click', (e) => {
    e.preventDefault();
    aboutModal.style.display = 'block';
});

closeModal.addEventListener('click', () => {
    aboutModal.style.display = 'none';
});

window.addEventListener('click', (e) => {
    if (e.target === aboutModal) {
        aboutModal.style.display = 'none';
    }
});

// Handle keyboard shortcuts
newsText.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && e.ctrlKey) {
        e.preventDefault();
        analyzeNews();
    }
});

// Simple fake news detection function (rule-based approach)
function detectFakeNews(text) {
    const fakeIndicators = [
        'shocking',
        'unbelievable',
        'miracle',
        'secret',
        'conspiracy',
        'never before seen',
        'doctors hate',
        'you won\'t believe',
        'mind-blowing'
    ];
    
    const textLower = text.toLowerCase();
    let fakeScore = 0;
    
    // Check for sensational words
    fakeIndicators.forEach(word => {
        if (textLower.includes(word)) {
            fakeScore += 1;
        }
    });
    
    // Check for excessive punctuation
    if ((text.match(/[!?]{2,}/g) || []).length > 0) {
        fakeScore += 1;
    }
    
    // Check for ALL CAPS words
    if ((text.match(/\b[A-Z]{4,}\b/g) || []).length > 2) {
        fakeScore += 1;
    }
    
    return fakeScore >= 2;
}
