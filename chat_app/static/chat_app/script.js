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
// For now, we expect it to be globally available or we fetch it dynamically.
let askAgentUrl = '/ask/'; // Default, will be overridden if script in HTML sets it

const loadingSpinner = document.getElementById('loadingSpinner');

// Function to create an HTML table from table data
function createTableElement(tableData) {
    const table = document.createElement('table');
    table.classList.add('data-table');
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    tableData.columns.forEach(col => {
        const th = document.createElement('th');
        th.textContent = col;
        headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);
    table.appendChild(thead);
    const tbody = document.createElement('tbody');
    tableData.rows.forEach(row => {
        const tr = document.createElement('tr');
        row.forEach(cell => {
            const td = document.createElement('td');
            td.textContent = cell;
            tr.appendChild(td);
        });
        tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    return table;
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

    const strong = document.createElement('strong');
    strong.textContent = sender + ": ";
    messageDiv.appendChild(strong);

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
            messageDiv.appendChild(document.createTextNode(beforeText));
            messageDiv.appendChild(document.createElement('br'));
        }
        if (imageUrl) {
            const img = document.createElement('img');
            img.src = imageUrl;
            img.style.maxWidth = '100%';
            messageDiv.appendChild(img);
        }
        if (afterText) {
            messageDiv.appendChild(document.createElement('br'));
            messageDiv.appendChild(document.createTextNode(afterText));
        }
        chatbox.appendChild(messageDiv);
        chatbox.scrollTop = chatbox.scrollHeight;
        return;
    }

    // Handle inline table data (even if it's at the end of a longer message)
    const tableMarker = 'DATA_TABLE:';
    if (message.includes(tableMarker)) {
        const [before, after] = message.split(tableMarker);
        const beforeText = before.trim();
        const jsonStr = after.trim();
        if (beforeText) {
            // Render the text part
            const lines = beforeText.split('\n');
            lines.forEach((line, index) => {
                messageDiv.appendChild(document.createTextNode(line));
                if (index < lines.length - 1) {
                    messageDiv.appendChild(document.createElement('br'));
                }
            });
            messageDiv.appendChild(document.createElement('br'));
        }
        try {
            const tableData = JSON.parse(jsonStr);
            const tableEl = createTableElement(tableData);
            messageDiv.appendChild(tableEl);
        } catch (e) {
            console.error('Invalid table data JSON', e);
            // Optionally show the raw JSON as fallback
            messageDiv.appendChild(document.createTextNode(' [Invalid table data]'));
        }
        chatbox.appendChild(messageDiv);
        chatbox.scrollTop = chatbox.scrollHeight;
        return;
    }

    // Sanitize message and handle newlines explicitly
    const lines = message.split('\n');
    lines.forEach((line, index) => {
        messageDiv.appendChild(document.createTextNode(line));
        if (index < lines.length - 1) {
            messageDiv.appendChild(document.createElement('br'));    
        }
    });
    
    chatbox.appendChild(messageDiv);
    chatbox.scrollTop = chatbox.scrollHeight;
}

// Add a function to display a message loading spinner
function showMessageLoading() {
    const chatbox = document.getElementById('chatbox');
    if (!chatbox) return null;
    
    // Create the message loading container
    const loadingDiv = document.createElement('div');
    loadingDiv.classList.add('message-loading');
    loadingDiv.id = 'message-loading-indicator';
    
    // Create the typing animation
    const typingIndicator = document.createElement('div');
    typingIndicator.classList.add('typing-indicator');
    
    // Add the dots
    for (let i = 0; i < 3; i++) {
        const dot = document.createElement('span');
        typingIndicator.appendChild(dot);
    }
    
    // Add the "Assistant is thinking" text
    const loadingText = document.createElement('div');
    loadingText.classList.add('message-loading-text');
    loadingText.textContent = 'Assistant is thinking...';
    
    // Put it all together
    loadingDiv.appendChild(typingIndicator);
    loadingDiv.appendChild(loadingText);
    chatbox.appendChild(loadingDiv);
    
    // Scroll to bottom of chatbox to ensure the loading indicator is visible
    chatbox.scrollTop = chatbox.scrollHeight;
    
    return loadingDiv;
}

function hideMessageLoading() {
    const loadingDiv = document.getElementById('message-loading-indicator');
    if (loadingDiv && loadingDiv.parentNode) {
        loadingDiv.parentNode.removeChild(loadingDiv);
    }
}

// Modify the sendRequest function to show and hide message loading
async function sendRequest(url, bodyData) {
    if (loadingSpinner) loadingSpinner.style.display = 'inline-block';
    
    // Show the message loading indicator
    const messageLoading = showMessageLoading();
    
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
        // Hide the loader indicators
        if (loadingSpinner) loadingSpinner.style.display = 'none';
        hideMessageLoading();
    }
}

async function sendQuestion() {
    const userInput = document.getElementById('userInput');
    if (!userInput) return;
    const question = userInput.value;
    if (!question.trim()) return;

    displayMessage('You', question, 'user');
    userInput.value = '';
    
    const data = { question: question };
    if (window.askAgentUrlGlobal) askAgentUrl = window.askAgentUrlGlobal; // Use global var set in HTML
    
    const result = await sendRequest(askAgentUrl, data);

    if (result.sql_to_approve) {
        displaySqlApprovalInChat(result.sql_to_approve, result.tool_input);
    } else if (result.answer) {
        displayMessage('Assistant', result.answer);
    } else if (result.error) {
        displayMessage('Error', result.error, 'error');
    }
}

// New function to display SQL approval UI in the chat
function displaySqlApprovalInChat(sqlToApprove, toolInput) {
    const chatbox = document.getElementById('chatbox');
    if (!chatbox) return;
    
    // Create the SQL approval message container
    const approvalDiv = document.createElement('div');
    approvalDiv.classList.add('chat-message', 'assistant-message', 'sql-approval-message');
    approvalDiv.id = 'inline-sql-approval';
    
    // Add the heading
    const heading = document.createElement('strong');
    heading.textContent = 'Assistant: ';
    approvalDiv.appendChild(heading);
    
    // Add the instruction text
    const instructionText = document.createElement('p');
    instructionText.innerHTML = '<i class="fas fa-code"></i> I\'ve generated SQL to answer your question. Please review and approve:';
    instructionText.classList.add('sql-approval-instruction');
    approvalDiv.appendChild(instructionText);
    
    // Add the SQL textarea
    const sqlTextarea = document.createElement('textarea');
    sqlTextarea.id = 'inlineSqlToApproveText';
    sqlTextarea.value = sqlToApprove;
    sqlTextarea.rows = 5;
    sqlTextarea.classList.add('inline-sql-textarea');
    approvalDiv.appendChild(sqlTextarea);
    
    // Add the action buttons
    const buttonContainer = document.createElement('div');
    buttonContainer.classList.add('inline-approval-buttons');
    
    // Approve button
    const approveButton = document.createElement('button');
    approveButton.innerHTML = '<i class="fas fa-check-circle"></i> Approve & Run';
    approveButton.classList.add('inline-approve-button', 'btn-effect');
    approveButton.addEventListener('click', function() {
        approveSqlInline(sqlTextarea.value);
    });
    buttonContainer.appendChild(approveButton);
    
    approvalDiv.appendChild(buttonContainer);
    
    // Add the approval UI to the chat
    chatbox.appendChild(approvalDiv);
    
    // Scroll to the new element
    chatbox.scrollTop = chatbox.scrollHeight;
    
    // Store the tool input in a data attribute (to avoid creating a new global variable)
    approvalDiv.dataset.toolInput = toolInput;
}

// New function to handle inline SQL approval
async function approveSqlInline(approvedSql) {
    if (!approvedSql.trim()) {
        displayMessage('System', "SQL cannot be empty for approval.", 'error');
        return;
    }
    
    // Remove the SQL approval UI from chat
    const approvalElement = document.getElementById('inline-sql-approval');
    if (approvalElement) {
        approvalElement.remove();
    }
    
    // Display a simple confirmation message
    displayMessage('You', `(Approved SQL)`, 'user');
    
    const data = { approved_sql: approvedSql };
    if (window.askAgentUrlGlobal) askAgentUrl = window.askAgentUrlGlobal;
    const result = await sendRequest(askAgentUrl, data);
    
    if (result.answer) {
        displayMessage('Assistant', result.answer);
    } else if (result.error) {
        displayMessage('Error', result.error, 'error');
    }
}

// Modified to remove the approveSql function since we'll use approveSqlInline instead
// Keep the original if there are other references to it that we want to maintain
async function approveSql() {
    const sqlToApproveText = document.getElementById('sqlToApproveText');
    if (!sqlToApproveText) return;
    const approvedSql = sqlToApproveText.value;

    if (!approvedSql.trim()) {
        displayMessage('System', "SQL cannot be empty for approval.", 'error');
        return;
    }
    displayMessage('You', `(Approved SQL)`); 

    const data = { approved_sql: approvedSql };
    if (window.askAgentUrlGlobal) askAgentUrl = window.askAgentUrlGlobal; // Use global var set in HTML
    const result = await sendRequest(askAgentUrl, data);

    if (result.answer) {
        displayMessage('Assistant', result.answer);
    } else if (result.error) {
        displayMessage('Error', result.error, 'error');
    }
    const sqlApprovalArea = document.getElementById('sqlApprovalArea');
    if (sqlApprovalArea) sqlApprovalArea.style.display = 'none';
    sqlToApproveText.value = '';
}

function handleKeyPress(event) {
    if (event.key === "Enter") {
        const focusedElement = document.activeElement;
        if (focusedElement && focusedElement.id === 'userInput') {
            sendQuestion();
            event.preventDefault(); // Prevent default Enter key action (e.g., form submission)
        }
        // If you want Enter to also approve SQL when textarea is focused:
        // else if (focusedElement && focusedElement.id === 'sqlToApproveText') {
        //     approveSql();
        //     event.preventDefault();
        // }
    }
}

// Event listeners should be added after the DOM is fully loaded
document.addEventListener('DOMContentLoaded', () => {
    const sendButton = document.getElementById('sendButton');
    if (sendButton) {
        sendButton.addEventListener('click', sendQuestion);
    }

    // We'll keep this for backward compatibility, but it won't be used in the new flow
    const approveSqlButton = document.getElementById('approveSqlButton');
    if (approveSqlButton) {
        approveSqlButton.addEventListener('click', approveSql);
    }

    const userInput = document.getElementById('userInput');
    if (userInput) {
        userInput.addEventListener('keypress', handleKeyPress);
    }
    
    // Make askAgentUrl available to the script if set by Django template
    if (typeof window.ASK_AGENT_URL_GLOBAL !== 'undefined') {
        askAgentUrl = window.ASK_AGENT_URL_GLOBAL;
    }
});
