import React, { useState } from 'react';
import './SearchPanel.css';
import { FaSearch, FaChevronDown, FaChevronRight, FaFileAlt } from 'react-icons/fa';

const SearchPanel = ({ extractedData }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [expandedSections, setExpandedSections] = useState({});

  // Auto-expand all sections on mount
  React.useEffect(() => {
    if (extractedData) {
      const allSections = {};
      Object.keys(extractedData).forEach(key => {
        allSections[key] = true;
      });
      setExpandedSections(allSections);
    }
  }, [extractedData]);

  const toggleSection = (section) => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }));
  };

  const filterData = (data) => {
    if (!searchTerm) return data;
    
    const filtered = {};
    Object.keys(data).forEach(key => {
      const value = data[key];
      if (typeof value === 'string' && value.toLowerCase().includes(searchTerm.toLowerCase())) {
        filtered[key] = value;
      } else if (typeof value === 'object') {
        const nestedFiltered = filterData(value);
        if (Object.keys(nestedFiltered).length > 0) {
          filtered[key] = nestedFiltered;
        }
      }
    });
    return filtered;
  };

  const renderValue = (value) => {
    if (!value) return <span className="data-value">N/A</span>;
    
    if (Array.isArray(value)) {
      return (
        <ul className="data-list">
          {value.map((item, idx) => (
            <li key={idx}>{typeof item === 'object' ? JSON.stringify(item) : item}</li>
          ))}
        </ul>
      );
    }
    if (typeof value === 'object' && value !== null) {
      return (
        <div className="nested-data">
          {Object.entries(value).map(([k, v]) => (
            <div key={k} className="data-item">
              <strong>{k.replace(/_/g, ' ')}:</strong> {renderValue(v)}
            </div>
          ))}
        </div>
      );
    }
    
    const text = String(value);
    
    // Check if it's table data (contains | separators)
    if (text.includes('|') && text.split('\n').length > 1) {
      const rows = text.split('\n').filter(row => row.trim());
      const tableData = rows.map(row => 
        row.split('|').map(cell => cell.trim()).filter(cell => cell)
      );
      
      return (
        <div className="table-container">
          <table className="data-table">
            <tbody>
              {tableData.map((row, rowIdx) => (
                <tr key={rowIdx}>
                  {row.map((cell, cellIdx) => (
                    <td key={cellIdx}>{cell}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
    }
    
    // Handle long text with line breaks
    if (text.length > 200) {
      return (
        <div className="data-value long-text">
          {text.split('\n').map((line, idx) => (
            <p key={idx}>{line}</p>
          ))}
        </div>
      );
    }
    return <span className="data-value">{text}</span>;
  };

  const filteredData = extractedData ? filterData(extractedData) : {};
  const sections = Object.keys(filteredData);

  return (
    <div className="search-panel">
      <div className="panel-header">
        <h2>Document Data</h2>
        <div className="search-box">
          <FaSearch className="search-icon" />
          <input
            type="text"
            placeholder="Search in document..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="search-input"
          />
        </div>
      </div>

      <div className="data-sections">
        {sections.length === 0 ? (
          <div className="empty-state">
            {extractedData ? (
              <p>No data found matching "{searchTerm}"</p>
            ) : (
              <>
                <FaFileAlt className="empty-icon" />
                <p>Upload a document to view extracted data</p>
              </>
            )}
          </div>
        ) : (
          sections.map(section => (
            <div key={section} className="section">
              <div
                className="section-header"
                onClick={() => toggleSection(section)}
              >
                <span className="section-icon">
                  {expandedSections[section] ? <FaChevronDown /> : <FaChevronRight />}
                </span>
                <h3>{section.replace(/_/g, ' ').toUpperCase()}</h3>
              </div>
              
              {expandedSections[section] && (
                <div className="section-content">
                  {renderValue(filteredData[section])}
                </div>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default SearchPanel;
