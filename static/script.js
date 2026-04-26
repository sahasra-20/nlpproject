const chatForm = document.getElementById('chat-form');
const userInput = document.getElementById('user-input');
const chatHistory = document.getElementById('chat-history');
const loadingIndicator = document.getElementById('loading');
const clearBtn = document.getElementById('clear-btn');

const HISTORY_KEY = 'agriqa_chat_history';

// Load history from localStorage on startup
document.addEventListener('DOMContentLoaded', () => {
    const savedHistory = localStorage.getItem(HISTORY_KEY);
    if (savedHistory) {
        // Clear the default welcome message if we have history
        chatHistory.innerHTML = '';
        const messages = JSON.parse(savedHistory);
        messages.forEach(msg => {
            appendMessage(msg.role, msg.content, false);
        });
        scrollToBottom();
    }
});

// Clear history
clearBtn.addEventListener('click', () => {
    if(confirm("Are you sure you want to clear your chat history?")) {
        localStorage.removeItem(HISTORY_KEY);
        chatHistory.innerHTML = `
            <div class="message bot welcome-message">
                <div class="avatar">🤖</div>
                <div class="bubble">History cleared. How can I help you today?</div>
            </div>
        `;
    }
});

chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const text = userInput.value.trim();
    if (!text) return;

    // Append user message
    appendMessage('user', text);
    userInput.value = '';
    
    // Show loading
    loadingIndicator.style.display = 'flex';
    scrollToBottom();

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: text })
        });
        
        const data = await response.json();
        
        loadingIndicator.style.display = 'none';
        
        if (data.error) {
            appendMessage('bot', `Error: ${data.error}`);
        } else {
            appendMessage('bot', data.answer);
        }
    } catch (error) {
        loadingIndicator.style.display = 'none';
        appendMessage('bot', 'Sorry, I could not connect to the server. Is it running?');
        console.error(error);
    }
});

function appendMessage(role, text, save=true) {
    const msgDiv = document.createElement('div');
    msgDiv.classList.add('message', role);
    
    const avatar = document.createElement('div');
    avatar.classList.add('avatar');
    avatar.textContent = role === 'user' ? '🧑‍🌾' : '🤖';
    
    const bubble = document.createElement('div');
    bubble.classList.add('bubble');
    bubble.textContent = text;
    
    msgDiv.appendChild(avatar);
    msgDiv.appendChild(bubble);
    
    chatHistory.appendChild(msgDiv);
    scrollToBottom();

    if (save) {
        saveMessageToHistory(role, text);
    }
}

function saveMessageToHistory(role, content) {
    let history = [];
    const saved = localStorage.getItem(HISTORY_KEY);
    if (saved) {
        history = JSON.parse(saved);
    }
    history.push({ role, content });
    localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
}

function scrollToBottom() {
    chatHistory.scrollTop = chatHistory.scrollHeight;
}
