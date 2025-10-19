import React, { useState, useEffect } from 'react';
import './App.css';
import UploadFile from './components/UploadFile';
import ChatWindow from './components/ChatWindow';
import SearchPanel from './components/SearchPanel';
import { FaHospitalAlt, FaComments, FaFileAlt } from 'react-icons/fa';

function App() {
  const [currentFile, setCurrentFile] = useState(null);
  const [extractedData, setExtractedData] = useState(null);
  const [activeTab, setActiveTab] = useState('chat');
  const [fileHistory, setFileHistory] = useState([]);
  const [isNewFile, setIsNewFile] = useState(false);

  // Load file history from localStorage on mount
  useEffect(() => {
    const savedHistory = localStorage.getItem('file_history');
    if (savedHistory) {
      setFileHistory(JSON.parse(savedHistory));
    }
  }, []);

  // Save file history to localStorage
  useEffect(() => {
    if (fileHistory.length > 0) {
      localStorage.setItem('file_history', JSON.stringify(fileHistory));
    }
  }, [fileHistory]);

  const handleUploadSuccess = (filename, data) => {
    // Clear previous file data
    if (currentFile) {
      localStorage.removeItem(`chat_${currentFile}`);
      localStorage.removeItem(`data_${currentFile}`);
    }
    
    setCurrentFile(filename);
    setExtractedData(data);
    setIsNewFile(true); // Always clear chat for new file
    
    // Clear file history and add new file
    const newFile = { filename, data, uploadedAt: new Date().toISOString() };
    setFileHistory([newFile]);
    
    // Store new data
    localStorage.setItem(`data_${filename}`, JSON.stringify(data));
  };

  return (
    <div className="App">
      <header className="app-header">
        <div className="header-content">
          <FaHospitalAlt className="header-icon" />
          <div>
            <h1>Patient Discharge Assistant</h1>
            <p>AI-Powered Medical Document Analysis</p>
          </div>
        </div>
      </header>

      <main className="app-main">
        <div className="main-row">
          <UploadFile onUploadSuccess={handleUploadSuccess} />
          
          <div className="right-panel">
            {currentFile && (
              <div className="nav-tabs">
                <button 
                  className={`tab ${activeTab === 'chat' ? 'active' : ''}`}
                  onClick={() => setActiveTab('chat')}
                >
                  <FaComments /> Chat
                </button>
                <button 
                  className={`tab ${activeTab === 'document' ? 'active' : ''}`}
                  onClick={() => {
                    setActiveTab('document');
                    // Load extracted data from localStorage if not in state
                    if (!extractedData) {
                      const savedData = localStorage.getItem(`data_${currentFile}`);
                      if (savedData) {
                        setExtractedData(JSON.parse(savedData));
                      }
                    }
                  }}
                >
                  <FaFileAlt /> Document
                </button>
              </div>
            )}
            
            {activeTab === 'chat' ? (
              <ChatWindow filename={currentFile} isNewFile={isNewFile} onNewFileHandled={() => setIsNewFile(false)} />
            ) : (
              <SearchPanel extractedData={extractedData} />
            )}
          </div>
        </div>
      </main>

      <footer className="app-footer">
        <p>© 2024 Patient Discharge Assistant | Powered by AI</p>
      </footer>
    </div>
  );
}

export default App;
