import React, { useState, useRef, useEffect } from 'react';
import './ChatWindow.css';
import { FaUser, FaRobot, FaInfoCircle, FaPaperPlane, FaMicrophone, FaTrash } from 'react-icons/fa';

const ChatWindow = ({ filename, isNewFile, onNewFileHandled }) => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [typingText, setTypingText] = useState('');
  const [isListening, setIsListening] = useState(false);
  const messagesEndRef = useRef(null);
  const messagesContainerRef = useRef(null);
  const recognitionRef = useRef(null);

  // Load/save messages with localStorage persistence
  useEffect(() => {
    if (filename) {
      const savedMessages = localStorage.getItem(`chat_${filename}`);
      if (savedMessages && !isNewFile) {
        setMessages(JSON.parse(savedMessages));
      } else {
        const newMessages = [{
          type: 'system',
          text: isNewFile ? `New document "${filename}" loaded. Previous chat cleared.` : `Document "${filename}" loaded. Ask me anything about it!`
        }];
        setMessages(newMessages);
        if (isNewFile && onNewFileHandled) {
          onNewFileHandled();
        }
      }
    }
  }, [filename, isNewFile]);

  useEffect(() => {
    if (messages.length > 0 && filename) {
      localStorage.setItem(`chat_${filename}`, JSON.stringify(messages));
    }
  }, [messages, filename]);

  // Initialize speech recognition
  useEffect(() => {
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      recognitionRef.current = new SpeechRecognition();
      recognitionRef.current.continuous = false;
      recognitionRef.current.interimResults = false;
      recognitionRef.current.lang = 'en-US';
      
      recognitionRef.current.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        setInput(transcript);
        setIsListening(false);
      };
      
      recognitionRef.current.onerror = () => {
        setIsListening(false);
      };
      
      recognitionRef.current.onend = () => {
        setIsListening(false);
      };
    }
  }, []);



  const scrollToBottom = () => {
    if (messagesContainerRef.current) {
      messagesContainerRef.current.scrollTop = messagesContainerRef.current.scrollHeight;
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    if (typingText) {
      scrollToBottom();
    }
  }, [typingText]);



  const typeText = (text, metadata) => {
    let index = 0;
    setTypingText('');
    
    const interval = setInterval(() => {
      if (index < text.length) {
        setTypingText(text.substring(0, index + 1));
        index++;
      } else {
        clearInterval(interval);
        setMessages(prev => [...prev, {
          type: 'bot',
          text: text,
          ...metadata
        }]);
        setTypingText('');
        setLoading(false);
      }
    }, 20);
  };

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage = { type: 'user', text: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const response = await fetch('http://localhost:8000/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: input, filename }),
      });

      const data = await response.json();

      if (response.ok) {
        const answer = data.answer || 'No response received';
        typeText(answer, {
          confidence: data.confidence,
          category: data.category
        });
      } else {
        setMessages(prev => [...prev, {
          type: 'error',
          text: `Error: ${data.detail}`
        }]);
        setLoading(false);
      }
    } catch (error) {
      setMessages(prev => [...prev, {
        type: 'error',
        text: `Failed to get response: ${error.message}`
      }]);
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const clearChat = () => {
    const newMessages = [{
      type: 'system',
      text: filename ? `Chat cleared. Document "${filename}" is still loaded.` : 'Chat cleared.'
    }];
    setMessages(newMessages);
    if (filename) {
      localStorage.setItem(`chat_${filename}`, JSON.stringify(newMessages));
    }
  };

  const startListening = () => {
    if (recognitionRef.current && !isListening) {
      setIsListening(true);
      recognitionRef.current.start();
    }
  };

  return (
    <div className="chat-container">
      <div className="chat-header">
        <h2>Medical Assistant</h2>
        <div className="header-controls">
          <button className="clear-chat-btn" onClick={clearChat} title="Clear Chat">
            <FaTrash /> Clear
          </button>
          {filename && <span className="active-doc"><FaInfoCircle /> {filename}</span>}
        </div>
      </div>

      <div className="messages-container" ref={messagesContainerRef}>
        {messages.map((msg, idx) => (
          <div key={idx} className={`message ${msg.type}`}>
            {msg.type === 'user' && <span className="avatar user-avatar"><FaUser /></span>}
            {msg.type === 'bot' && <span className="avatar bot-avatar"><FaRobot /></span>}
            {msg.type === 'system' && <span className="avatar system-avatar"><FaInfoCircle /></span>}
            
            <div className="message-content">
              <div className="message-text">
                {msg.text.split('\n').map((line, i) => {
                  // Remove ** markers and detect bold text
                  const cleanLine = line.replace(/\*\*/g, '');
                  const isBold = line.includes('**');
                  
                  return (
                    <React.Fragment key={i}>
                      {line.includes('|') && line.split('|').length > 2 ? (
                        <div className="inline-table">
                          {line.split('|').map((cell, j) => (
                            <span key={j} className="table-cell">{cell.trim()}</span>
                          ))}
                        </div>
                      ) : isBold ? (
                        <div className="bold-line">{cleanLine}</div>
                      ) : line.startsWith('•') || line.startsWith('-') ? (
                        <div className="bullet-line">{cleanLine.substring(1).trim()}</div>
                      ) : cleanLine.trim() ? (
                        <div className="text-line">{cleanLine}</div>
                      ) : null}
                    </React.Fragment>
                  );
                })}
              </div>
              {msg.confidence && (
                <small className="metadata">
                  Confidence: {(msg.confidence * 100).toFixed(0)}% | Category: {msg.category}
                </small>
              )}
            </div>
          </div>
        ))}
        
        {loading && !typingText && (
          <div className="message bot">
            <span className="avatar bot-avatar"><FaRobot /></span>
            <div className="message-content">
              <div className="typing-indicator">
                <span></span><span></span><span></span>
              </div>
            </div>
          </div>
        )}
        
        {typingText && (
          <div className="message bot typing-message">
            <span className="avatar bot-avatar"><FaRobot /></span>
            <div className="message-content">
              <div className="message-text">
                {typingText.split('\n').map((line, i) => (
                  <React.Fragment key={i}>
                    {line}
                    {i < typingText.split('\n').length - 1 && <br />}
                  </React.Fragment>
                ))}
                <span className="cursor">|</span>
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      <div className="input-container">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Ask about medications, diagnosis, instructions... (or use voice)"
          rows="2"
          disabled={!filename || loading}
        />
        <button
          className={`mic-btn ${isListening ? 'listening' : ''}`}
          onClick={startListening}
          disabled={!filename || loading || isListening}
          title="Voice Input"
        >
          <FaMicrophone />
        </button>
        <button
          onClick={handleSend}
          disabled={!input.trim() || !filename || loading}
        >
          <FaPaperPlane />
        </button>
      </div>
    </div>
  );
};

export default ChatWindow;
