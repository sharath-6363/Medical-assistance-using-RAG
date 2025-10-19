import React, { useState } from 'react';
import './UploadFile.css';
import { FaCloudUploadAlt, FaFileAlt, FaCheckCircle, FaExclamationCircle } from 'react-icons/fa';

const UploadFile = ({ onUploadSuccess }) => {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [message, setMessage] = useState('');

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setMessage('Please select a file first');
      return;
    }

    setUploading(true);
    setMessage('Uploading...');

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/upload', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (response.ok) {
        setMessage('Upload successful! Extracting text...');
        
        // Poll for processing status with faster interval
        let pollCount = 0;
        const maxPolls = 60; // 60 seconds max
        
        const checkStatus = setInterval(async () => {
          pollCount++;
          
          if (pollCount > maxPolls) {
            clearInterval(checkStatus);
            setMessage('Processing timeout. Please try again.');
            setUploading(false);
            return;
          }
          
          try {
            const statusRes = await fetch(`http://localhost:8000/processing-status/${data.filename}`);
            const statusData = await statusRes.json();
            
            console.log('Processing status:', statusData);
            
            if (statusData.status === 'completed') {
              clearInterval(checkStatus);
              
              // Check if data was actually extracted
              const extractedData = statusData.extracted_data || {};
              const sectionCount = Object.keys(extractedData).length;
              
              if (sectionCount > 0) {
                setMessage(`✅ Document processed! Extracted ${sectionCount} sections`);
                console.log('Extracted data:', extractedData);
                onUploadSuccess(data.filename, extractedData);
              } else {
                setMessage('⚠️ Document uploaded but no data extracted. Check file format.');
                console.log('No data extracted from:', data.filename);
              }
              
              setFile(null);
              setUploading(false);
            } else if (statusData.status === 'error') {
              clearInterval(checkStatus);
              setMessage(`Error: ${statusData.message}`);
              setUploading(false);
            } else {
              // Show more detailed processing status
              if (statusData.message) {
                setMessage(`Processing... ${statusData.message}`);
              } else {
                setMessage(`Processing... (${pollCount}s)`);
              }
            }
          } catch (err) {
            console.error('Status check error:', err);
          }
        }, 1000); // Check every 1 second for faster response
      } else {
        setMessage(`Error: ${data.detail}`);
        setUploading(false);
      }
    } catch (error) {
      setMessage(`Upload failed: ${error.message}`);
      setUploading(false);
    }
  };

  return (
    <div className="upload-container">
      <h2>📄 Upload Medical Document</h2>
      
      <div
        className={`drop-zone ${dragActive ? 'active' : ''}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <input
          type="file"
          id="file-input"
          onChange={handleChange}
          accept=".pdf,.docx,.doc,.txt,.png,.jpg,.jpeg"
          style={{ display: 'none' }}
        />
        <label htmlFor="file-input" className="file-label">
          {file ? (
            <div className="file-info">
              <FaFileAlt className="file-icon" />
              <span className="file-name">{file.name}</span>
              <span className="file-size">({(file.size / 1024).toFixed(2)} KB)</span>
            </div>
          ) : (
            <div className="upload-prompt">
              <FaCloudUploadAlt className="upload-icon" />
              <p>Drag & drop your file here or click to browse</p>
              <small>Supported: PDF, DOCX, DOC, TXT, PNG, JPG (Max 50MB)</small>
            </div>
          )}
        </label>
      </div>

      <button
        className="upload-btn"
        onClick={handleUpload}
        disabled={!file || uploading}
      >
        <FaCloudUploadAlt />
        {uploading ? 'Processing...' : 'Upload & Process'}
      </button>

      {message && (
        <div className={`message ${message.includes('Error') ? 'error' : 'success'}`}>
          {message.includes('Error') ? <FaExclamationCircle /> : <FaCheckCircle />}
          <span>{message}</span>
        </div>
      )}
    </div>
  );
};

export default UploadFile;
