function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

const csrftoken = getCookie('csrftoken');
// Note: askAgentUrl will be defined globally in a script tag in index.html
// or passed via a data attribute because Django template tags don't work in static JS files.
let askAgentUrl = '/ask/'; // Default, will be overridden if script in HTML sets it

const loadingSpinner = document.getElementById('loadingSpinner');

// Theme toggle functionality
function toggleTheme() {
    const body = document.body;
    const themeToggle = document.getElementById('theme-toggle');
    const isDarkMode = body.classList.contains('dark-theme');
    
    if (isDarkMode) {
        body.classList.remove('dark-theme');
        body.classList.add('light-theme');
        themeToggle.innerHTML = '<i class="fas fa-moon"></i><span>Dark Mode</span>';
        localStorage.setItem('theme', 'light');
    } else {
        body.classList.remove('light-theme');
        body.classList.add('dark-theme');
        themeToggle.innerHTML = '<i class="fas fa-sun"></i><span>Light Mode</span>';
        localStorage.setItem('theme', 'dark');
    }
}

// Apply saved theme from localStorage
function applyTheme() {
    const savedTheme = localStorage.getItem('theme');
    const themeToggle = document.getElementById('theme-toggle');
    
    if (savedTheme === 'dark') {
        document.body.classList.remove('light-theme');
        document.body.classList.add('dark-theme');
        themeToggle.innerHTML = '<i class="fas fa-sun"></i><span>Light Mode</span>';
    } else {
        document.body.classList.remove('dark-theme');
        document.body.classList.add('light-theme');
        themeToggle.innerHTML = '<i class="fas fa-moon"></i><span>Dark Mode</span>';
    }
}

// Modal functionality
function toggleModal() {
    const modal = document.getElementById('aboutModal');
    modal.style.display = modal.style.display === 'block' ? 'none' : 'block';
}

// Handle clicking outside the modal to close
function handleOutsideClick(event) {
    const modal = document.getElementById('aboutModal');
    if (event.target === modal) {
        modal.style.display = 'none';
    }
}

function displayMessage(sender, message, type = 'normal') {
    const chatbox = document.getElementById('chatbox');
    if (!chatbox) {
        console.error("Chatbox element not found!");
        return;
    }
    
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('chat-message');
    
    if (type === 'user') {
        messageDiv.classList.add('user-message');
    } else if (type === 'error') {
        messageDiv.classList.add('error-message');
    } else {
        messageDiv.classList.add('assistant-message');
    }

    // Create message content container
    const contentDiv = document.createElement('div');
    contentDiv.classList.add('message-content');
    messageDiv.appendChild(contentDiv);

    if (sender) {
        const strong = document.createElement('strong');
        strong.textContent = sender + ": ";
        contentDiv.appendChild(strong);
    }

    // Handle inline plot images
    const plotMarker = 'Plot generated at:';
    if (message.includes(plotMarker)) {
        // Split text around the marker
        const [before, after] = message.split(plotMarker);
        const beforeText = before.trim();
        const urlMatch = after.match(/\S+\.(png|jpg|jpeg|gif)/i);
        const imageUrl = urlMatch ? urlMatch[0] : null;
        const afterText = imageUrl ? after.substring(imageUrl.length).trim() : after.trim();

        if (beforeText) {
            contentDiv.appendChild(document.createTextNode(beforeText));
            contentDiv.appendChild(document.createElement('br'));
        }
        if (imageUrl) {
            const img = document.createElement('img');
            img.src = imageUrl;
            img.alt = "Generated plot visualization";
            img.classList.add('plot-image');
            messageDiv.appendChild(img);
        }
        if (afterText) {
            contentDiv.appendChild(document.createElement('br'));
            contentDiv.appendChild(document.createTextNode(afterText));
        }
        
        chatbox.appendChild(messageDiv);
        chatbox.scrollTop = chatbox.scrollHeight;
        return;
    }

    // Sanitize message and handle newlines explicitly
    const lines = message.split('\n');
    lines.forEach((line, index) => {
        contentDiv.appendChild(document.createTextNode(line));
        if (index < lines.length - 1) {
            contentDiv.appendChild(document.createElement('br'));    
        }
    });
    
    chatbox.appendChild(messageDiv);
    chatbox.scrollTop = chatbox.scrollHeight;
}

async function sendRequest(url, bodyData) {
    if (loadingSpinner) loadingSpinner.style.display = 'flex';
    try {
        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': csrftoken
            },
            body: JSON.stringify(bodyData)
        });
        if (!response.ok) {
            // Try to parse error from server if it's a JSON response
            try {
                const errorData = await response.json();
                return { error: errorData.error || `Server error: ${response.status}` };
            } catch (e) {
                return { error: `Server error: ${response.status}` };
            }
        }
        return response.json();
    } catch (error) {
        console.error("Request failed:", error);
        return { error: "Network error or server unreachable." };
    }
    finally {
        if (loadingSpinner) loadingSpinner.style.display = 'none';
    }
}

function resetUI() {
    // Clear user input field
    const userInput = document.getElementById('userInput');
    if (userInput) userInput.value = '';
    
    // Hide SQL approval area
    const sqlApprovalArea = document.getElementById('sqlApprovalArea');
    if (sqlApprovalArea) sqlApprovalArea.style.display = 'none';
    
    // Focus on input field
    if (userInput) userInput.focus();
}

async function sendQuestion(question = null) {
    const userInput = document.getElementById('userInput');
    if (!question && (!userInput || !userInput.value.trim())) return;
    
    const questionText = question || userInput.value;
    displayMessage('You', questionText, 'user');
    resetUI();
    
    const data = { question: questionText };
    if (window.askAgentUrlGlobal) askAgentUrl = window.askAgentUrlGlobal; // Use global var set in HTML
    
    const result = await sendRequest(askAgentUrl, data);

    if (result.sql_to_approve) {
        const sqlToApproveText = document.getElementById('sqlToApproveText');
        if (sqlToApproveText) sqlToApproveText.value = result.sql_to_approve;
        
        const sqlApprovalArea = document.getElementById('sqlApprovalArea');
        if (sqlApprovalArea) {
            sqlApprovalArea.style.display = 'block';
            // Animate the appearance
            sqlApprovalArea.style.animation = 'none';
            sqlApprovalArea.offsetHeight; // Trigger reflow
            sqlApprovalArea.style.animation = 'panel-slide-up 0.3s ease';
        }
    } else if (result.answer) {
        displayMessage('Assistant', result.answer);
    } else if (result.error) {
        displayMessage('Error', result.error, 'error');
    }
}

async function approveSql() {
    const sqlToApproveText = document.getElementById('sqlToApproveText');
    if (!sqlToApproveText) return;
    const approvedSql = sqlToApproveText.value;

    if (!approvedSql.trim()) {
        displayMessage('System', "SQL cannot be empty for approval.", 'error');
        return;
    }
    displayMessage('You', `(Approved SQL)`, 'user'); 

    const data = { approved_sql: approvedSql };
    if (window.askAgentUrlGlobal) askAgentUrl = window.askAgentUrlGlobal; // Use global var set in HTML
    
    const sqlApprovalArea = document.getElementById('sqlApprovalArea');
    if (sqlApprovalArea) sqlApprovalArea.style.display = 'none';
    
    const result = await sendRequest(askAgentUrl, data);

    if (result.answer) {
        displayMessage('Assistant', result.answer);
    } else if (result.error) {
        displayMessage('Error', result.error, 'error');
    }
    
    sqlToApproveText.value = '';
}

function handleKeyPress(event) {
    if (event.key === "Enter") {
        const focusedElement = document.activeElement;
        if (focusedElement && focusedElement.id === 'userInput') {
            sendQuestion();
            event.preventDefault(); // Prevent default Enter key action (e.g., form submission)
        }
        // Using Ctrl+Enter for SQL text area since it's multiline
        if (event.ctrlKey && focusedElement && focusedElement.id === 'sqlToApproveText') {
            approveSql();
            event.preventDefault();
        }
    }
}

function setupSampleQuestions() {
    const sampleQuestions = document.querySelectorAll('.sample-question');
    sampleQuestions.forEach(question => {
        question.addEventListener('click', (e) => {
            e.preventDefault();
            sendQuestion(question.textContent);
        });
    });
}

function newChat() {
    // Reload the page to start a fresh session
    window.location.reload();
}

// Event listeners should be added after the DOM is fully loaded
document.addEventListener('DOMContentLoaded', () => {
    // Apply theme from localStorage
    applyTheme();
    
    // Setup theme toggle
    const themeToggle = document.getElementById('theme-toggle');
    if (themeToggle) {
        themeToggle.addEventListener('click', toggleTheme);
    }
    
    // Setup info modal
    const infoButton = document.querySelector('.info-button');
    if (infoButton) {
        infoButton.addEventListener('click', toggleModal);
    }
    
    const modalClose = document.querySelector('.modal-close');
    if (modalClose) {
        modalClose.addEventListener('click', toggleModal);
    }
    
    // Close modal when clicking outside
    window.addEventListener('click', handleOutsideClick);
    
    // Send message button
    const sendButton = document.getElementById('sendButton');
    if (sendButton) {
        sendButton.addEventListener('click', () => sendQuestion());
    }

    // SQL approval button
    const approveSqlButton = document.getElementById('approveSqlButton');
    if (approveSqlButton) {
        approveSqlButton.addEventListener('click', approveSql);
    }

    // User input field enter key
    const userInput = document.getElementById('userInput');
    if (userInput) {
        userInput.addEventListener('keypress', handleKeyPress);
    }
    
    // SQL textarea Ctrl+Enter
    const sqlToApproveText = document.getElementById('sqlToApproveText');
    if (sqlToApproveText) {
        sqlToApproveText.addEventListener('keydown', handleKeyPress);
    }
    
    // New chat button
    const newChatBtn = document.querySelector('.new-chat-btn');
    if (newChatBtn) {
        newChatBtn.addEventListener('click', newChat);
    }
    
    // Setup sample questions
    setupSampleQuestions();
    
    // Make askAgentUrl available to the script if set by Django template
    if (typeof window.ASK_AGENT_URL_GLOBAL !== 'undefined') {
        askAgentUrl = window.ASK_AGENT_URL_GLOBAL;
    }
    
    // Focus the input field on page load
    if (userInput) {
        userInput.focus();
    }
});
