#!/usr/bin/env python3
"""
YOLOv7 Rice Quality Classification Streamlit App - Unified Full Version
Supports image upload, video processing, and live classification
Classes: white_rice, thi_rice, brown_rice, black_rice
"""

import os
import sys
import streamlit as st
from pathlib import Path
import tempfile
from PIL import Image
import numpy as np
import cv2
import torch
import time
import threading
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import subprocess
import json

# Page configuration
st.set_page_config(
    page_title="🌾 YOLOv7 Rice Quality Classifier",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enterprise-grade professional styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Reset & Base Styles */
    .stApp {
        background: #f7f9fc;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #1a1a1a;
        line-height: 1.6;
    }
    
    /* Remove default Streamlit padding */
    .main .block-container {
        padding: 2rem 1rem 3rem 1rem;
        max-width: 100%;
    }
    
    /* Professional Header */
    .enterprise-header {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 3rem 2rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.04);
        position: relative;
        overflow: hidden;
    }
    
    .enterprise-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #065f46 0%, #059669 25%, #10b981 50%, #fbbf24 100%);
    }
    
    .main-title {
        font-size: 2.75rem;
        font-weight: 800;
        color: #111827;
        text-align: center;
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }
    
    .main-subtitle {
        font-size: 1.125rem;
        color: #6b7280;
        text-align: center;
        font-weight: 400;
        margin: 0;
    }
    
    /* Status Bar */
    .status-bar {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1rem;
        margin: 1.5rem 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    
    .status-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.875rem;
        font-weight: 500;
        color: #374151;
    }
    
    .status-badge {
        background: #dcfce7;
        color: #166534;
        padding: 0.25rem 0.75rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Professional Cards */
    .pro-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .pro-card:hover {
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
        transform: translateY(-1px);
    }
    
    .card-header {
        font-size: 1.125rem;
        font-weight: 600;
        color: #111827;
        margin: 0 0 1rem 0;
        padding: 0 0 0.75rem 0;
        border-bottom: 2px solid #f3f4f6;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Navigation Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 0.25rem;
        margin-bottom: 2rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        gap: 0.25rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #6b7280;
        border: none;
        border-radius: 8px;
        padding: 0.875rem 1.5rem;
        font-weight: 500;
        font-size: 0.875rem;
        transition: all 0.15s ease;
        white-space: nowrap;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: #f9fafb;
        color: #374151;
    }
    
    .stTabs [aria-selected="true"] {
        background: #10b981 !important;
        color: white !important;
        font-weight: 600;
        box-shadow: 0 2px 4px rgba(16, 185, 129, 0.2);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 0.875rem;
        transition: all 0.2s ease;
        box-shadow: 0 2px 4px rgba(16, 185, 129, 0.2);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
        transform: translateY(-1px);
    }
    
    /* Form Inputs */
    .stSelectbox > div > div,
    .stNumberInput > div > div > input,
    .stTextInput > div > div > input {
        background: white;
        border: 1px solid #d1d5db;
        border-radius: 8px;
        font-size: 0.875rem;
        transition: all 0.2s ease;
    }
    
    .stSelectbox > div > div:focus-within,
    .stNumberInput > div > div > input:focus,
    .stTextInput > div > div > input:focus {
        border-color: #10b981;
        box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.1);
    }
    
    /* File Uploader */
    .stFileUploader > div > div > div {
        background: white;
        border: 2px dashed #d1d5db;
        border-radius: 12px;
        padding: 3rem 2rem;
        text-align: center;
        transition: all 0.2s ease;
    }
    
    .stFileUploader > div > div > div:hover {
        border-color: #10b981;
        background: #f0fdf4;
    }
    
    /* Enhanced Slider Styling for Configuration Cards */
    .stSlider {
        padding: 0;
        background: transparent;
        border: none;
        margin: 0.75rem 0;
    }
    
    /* Slider Track */
    .stSlider > div > div > div {
        background: rgba(255, 255, 255, 0.7);
        border-radius: 10px;
        height: 6px;
        box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Slider Fill - Dynamic based on slider context */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #059669 0%, #047857 100%);
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(5, 150, 105, 0.4);
        height: 6px;
    }
    
    /* Slider Thumb */
    .stSlider > div > div > div > div > div {
        background: white;
        border: 3px solid #059669;
        box-shadow: 0 2px 6px rgba(5, 150, 105, 0.3);
        width: 18px;
        height: 18px;
        border-radius: 50%;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        top: -6px;
    }
    
    .stSlider > div > div > div > div > div:hover {
        transform: scale(1.15);
        box-shadow: 0 3px 8px rgba(5, 150, 105, 0.5);
        border-width: 4px;
    }
    
    .stSlider > div > div > div > div > div:active {
        transform: scale(0.95);
        box-shadow: 0 1px 3px rgba(5, 150, 105, 0.6);
    }
    
    /* Slider Labels - Hidden since we use custom labels */
    .stSlider > label {
        display: none;
    }
    
    /* Remove any default margins */
    .stSlider > div {
        margin: 0;
        padding: 0;
    }
    
    /* Metrics */
    .css-1629p8f [data-testid="metric-container"] {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    
    .css-1629p8f [data-testid="metric-container"] [data-testid="metric-value"] {
        color: #10b981;
        font-weight: 700;
        font-size: 2rem;
    }
    
    .css-1629p8f [data-testid="metric-container"] [data-testid="metric-label"] {
        color: #6b7280;
        font-weight: 500;
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Alert Messages */
    .stAlert > div {
        border-radius: 8px;
        border: none;
        font-weight: 500;
    }
    
    .stSuccess > div {
        background: #f0fdf4;
        color: #166534;
        border-left: 4px solid #10b981;
    }
    
    .stWarning > div {
        background: #fefce8;
        color: #a16207;
        border-left: 4px solid #f59e0b;
    }
    
    .stError > div {
        background: #fef2f2;
        color: #b91c1c;
        border-left: 4px solid #ef4444;
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        color: #111827;
        font-weight: 600;
        line-height: 1.3;
    }
    
    h1 { font-size: 2.25rem; }
    h2 { font-size: 1.875rem; }
    h3 { font-size: 1.5rem; }
    h4 { font-size: 1.25rem; }
    
    .stMarkdown {
        color: #374151;
        line-height: 1.6;
    }
    
    /* Utility Classes */
    .text-success { color: #10b981; }
    .text-warning { color: #f59e0b; }
    .text-error { color: #ef4444; }
    .text-muted { color: #6b7280; }
    
    .bg-success {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .bg-warning {
        background: #fefce8;
        border: 1px solid #fde68a;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Results Display */
    .result-container {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        overflow: hidden;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    
    .result-header {
        background: #f9fafb;
        padding: 1rem 1.5rem;
        border-bottom: 1px solid #e5e7eb;
        font-weight: 600;
        color: #111827;
    }
    
    .result-content {
        padding: 1.5rem;
    }
    
    /* Detection Cards */
    .detection-card {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: all 0.2s ease;
    }
    
    .detection-card:hover {
        background: #f3f4f6;
        border-color: #10b981;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #10b981 0%, #f59e0b 100%);
        border-radius: 4px;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2rem;
        }
        
        .enterprise-header {
            padding: 2rem 1rem;
        }
        
        .status-bar {
            flex-direction: column;
            gap: 0.5rem;
            align-items: stretch;
        }
        
        .pro-card {
            padding: 1rem;
        }
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        color: white;
        padding: 1rem;
    }
    
    .sidebar-header {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 16px rgba(16, 185, 129, 0.2);
    }
    
    .sidebar-section {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(10px);
    }
    
    .sidebar-section h3 {
        color: #10b981;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(16, 185, 129, 0.2);
    }
    
    /* Bento Box Dashboard Grid */
    .bento-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        grid-auto-rows: minmax(200px, auto);
        gap: 1.5rem;
    }
    
    .bento-card {
        background: white;
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(0, 0, 0, 0.05);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .bento-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12);
    }
    
    .bento-card.large {
        grid-column: span 2;
    }
    
    .bento-card.tall {
        grid-row: span 2;
    }
    
    .bento-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding-bottom: 1rem;
    }
    
    .bento-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1e293b;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .bento-subtitle {
        font-size: 0.875rem;
        color: #64748b;
        margin-top: 0.25rem;
    }
    
    .bento-metric {
        font-size: 2.5rem;
        font-weight: 800;
        color: #10b981;
        line-height: 1;
        margin: 1rem 0;
    }
    
    .bento-badge {
        background: white;
        color: #10b981;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Dashboard Status Cards */
    .status-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .status-card {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .status-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #10b981 0%, #3b82f6 50%, #f59e0b 100%);
    }
    
    .status-value {
        font-size: 2rem;
        font-weight: 800;
        color: #1e293b;
        margin: 0.5rem 0;
    }
    
    .status-label {
        font-size: 0.875rem;
        color: #64748b;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Analysis Timeline */
    .timeline-item {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 1rem 0;
        border-bottom: 1px solid #f1f5f9;
    }
    
    .timeline-item:last-child {
        border-bottom: none;
    }
    
    .timeline-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background: #10b981;
        flex-shrink: 0;
    }
    
    .timeline-content {
        flex: 1;
    }
    
    .timeline-time {
        font-size: 0.75rem;
        color: #64748b;
        font-weight: 500;
    }
    
    /* Remove Streamlit branding and default styling */
    #MainMenu, .stDeployButton, footer, header {
        visibility: hidden !important;
    }
    
    /* Remove Streamlit default emotion cache styling */
    .st-emotion-cache-1wf904r,
    .e1gk92lc0,
    .st-emotion-cache-17lr0tt,
    .e1lln2w81,
    .st-emotion-cache-1r4qj8v,
    .e1akgbir4 {
        margin: 0 !important;
        padding: 0 !important;
        border: none !important;
        background: transparent !important;
    }
    
    /* Style specific emotion cache classes with white background */
    .st-emotion-cache-8atqhb,
    .e1mlolmg0,
    .en45cdb5 {
        background: white !important;
        margin: 0 !important;
        border-radius: 0 0 12px 12px !important;
        padding: 1.5rem !important;
        border: none !important;
        border-bottom: none !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08) !important;
        outline: none !important;
    }
    
    .st-emotion-cache-mv7iac,
    .e16xj5sw0 {
        background: white !important;
        border-radius: 0 0 20px 20px !important;
        margin: 0 !important;
        padding: 0 !important;
        border: none !important;
        border-bottom: none !important;
        border-top: none !important;
    }
    
    /* Hide labels in file uploader */
    .st-emotion-cache-mv7iac label,
    .e16xj5sw0 label,
    .st-emotion-cache-1wf904r,
    .e1gk92lc0,
    .st-emotion-cache-1xgtwnd,
    .edtmxes10 {
        display: none !important;
    }
    
    /* Hide specific empty containers */
    .stElementContainer.element-container.st-emotion-cache-17lr0tt.e1lln2w81:empty {
        display: none !important;
    }
    
    /* Remove unnecessary borders and lines within file uploader */
    .st-emotion-cache-8atqhb hr,
    .e1mlolmg0 hr,
    .st-emotion-cache-mv7iac hr,
    .e16xj5sw0 hr {
        display: none !important;
    }
    
    /* Remove any bottom borders from containers */
    .st-emotion-cache-8atqhb > div,
    .e1mlolmg0 > div,
    .st-emotion-cache-mv7iac > div,
    .e16xj5sw0 > div {
        border-bottom: none !important;
        border-top: none !important;
    }
    
    /* Ensure file uploader container has smooth bottom radius */
    [data-testid="stFileUploader"] {
        border-bottom: none !important;
    }
    
    [data-testid="stFileUploader"] > div {
        border-bottom: none !important;
        border-radius: 0 0 20px 20px !important;
    }
    
    /* Clean up container spacing */
    .element-container {
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* Override Streamlit's default container styles */
    .stElementContainer {
        margin: 0 !important;
        padding: 0 !important;
        background: transparent !important;
        border: none !important;
    }
    
    /* Remove default block container styling */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        max-width: 100% !important;
    }
    
    /* Force file uploader to be inside bento cards */
    .bento-card .stFileUploader {
        margin: 0 !important;
        padding: 0 !important;
        width: 100% !important;
    }
    
    /* Ensure all streamlit components stay within bento card bounds */
    .bento-card > div > [data-testid="element-container"] {
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* Override any absolute positioning that might take elements out of flow */
    .bento-card * {
        position: relative !important;
    }
    
    /* Force all Streamlit file uploaders to stay within their parent containers */
    [data-testid="stFileUploader"] {
        background: transparent !important;
        border: none !important;
        margin: 0 !important;
        padding: 0 !important;
        width: 100% !important;
        display: block !important;
    }
    
    /* Hide the main app container's padding to let bento cards control layout */
    .main > div {
        padding: 0 !important;
    }
    
    /* Ensure all elements stay within the bento grid */
    .bento-card .stFileUploader,
    .bento-card .stButton,
    .bento-card .stImage,
    .bento-card .stMarkdown {
        width: 100% !important;
        max-width: 100% !important;
        box-sizing: border-box !important;
    }
    
    /* Specifically target file uploader containers */
    .bento-card [data-testid="stFileUploader"] > div {
        margin: 0 !important;
        padding: 1rem !important;
        border: 2px dashed #d1d5db !important;
        border-radius: 12px !important;
        background: white !important;
    }
    
    /* Super aggressive file uploader containment */
    .bento-card.large [data-testid="stFileUploader"],
    .bento-card.large .st-emotion-cache-8atqhb,
    .bento-card.large .e1mlolmg0 {
        position: relative !important;
        display: block !important;
        width: 100% !important;
        margin: 1rem 0 !important;
        padding: 0 !important;
        float: none !important;
        clear: both !important;
        box-sizing: border-box !important;
        z-index: 1 !important;
    }
    
    /* Show all file uploaders but ensure they're positioned correctly */
    
    /* New approach - Style containers to look like bento cards */
    .bento-card-header.large-card-header {
        background: linear-gradient(135deg, #065f46, #059669, #10b981, #fbbf24);
        border-radius: 20px 20px 0 0;
        padding: 1.5rem 1.5rem 0 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        margin-bottom: 0;
        position: relative;
        color: white;
    }
    
    .bento-card-header.large-card-header:first-child {
        grid-column: span 2;
    }
    
    .bento-card-header.large-card-header .bento-title {
        color: white !important;
    }
    
    .bento-card-header.large-card-header .bento-subtitle {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    /* Style the container that follows the header to complete the card */
    .large-card-header + div {
        background: white !important;
        border-radius: 0 0 20px 20px !important;
        padding: 1.5rem !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08) !important;
        border: 1px solid rgba(0, 0, 0, 0.05) !important;
        border-top: none !important;
        margin-top: 0 !important;
        margin-bottom: 2rem !important;
    }
    
    /* Ensure file uploader fits nicely within the styled container */
    .large-card-header + div [data-testid="stFileUploader"] {
        background: transparent !important;
        border: none !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    
    .large-card-header + div [data-testid="stFileUploader"] > div > div > div {
        background: #f8fafc !important;
        border: 2px dashed #d1d5db !important;
        border-radius: 12px !important;
        padding: 2rem !important;
        text-align: center !important;
        transition: all 0.2s ease !important;
    }
    
    .large-card-header + div [data-testid="stFileUploader"] > div > div > div:hover {
        border-color: #10b981 !important;
        background: #f0fdf4 !important;
    }
    
    /* Force the file uploader to display within padding area */
    .bento-card div[style*="padding: 1rem"] [data-testid="stFileUploader"] {
        position: static !important;
        display: block !important;
        margin: 1rem 0 !important;
        width: 100% !important;
        max-width: 100% !important;
    }
    
    /* Override any absolute positioning on file uploader elements */
    .bento-card * {
        position: relative !important;
        float: none !important;
    }
    
    .bento-card .element-container,
    .bento-card .stElementContainer {
        position: relative !important;
        display: block !important;
        width: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* Image Display Controls - Fix oversized images */
    [data-testid="stImage"] {
        max-width: 100% !important;
        height: auto !important;
    }
    
    /* Limit image height to prevent layout breaking */
    [data-testid="stImage"] > img {
        max-height: 400px !important;
        width: auto !important;
        object-fit: contain !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
    }
    
    /* For result images in columns, make them fit better */
    .stColumn [data-testid="stImage"] > img {
        max-height: 350px !important;
        max-width: 100% !important;
    }
    
    /* Ensure image containers don't exceed their bounds */
    .stColumn > div > div > [data-testid="stImage"] {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        overflow: hidden !important;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #94a3b8;
    }
    
    /* Enhanced Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%);
        border-right: 2px solid #e2e8f0;
        box-shadow: 4px 0 20px rgba(0, 0, 0, 0.05);
        padding-top: 2rem;
    }
    
    /* Simplified sidebar sections */
    .sidebar-section {
    }
    
    .sidebar-header {
        background: linear-gradient(135deg, #065f46, #10b981);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0;
        font-weight: 600;
        font-size: 1rem;
    }
    
    .analytics-metrics {
        display: grid;
        grid-template-columns: 1fr;
        gap: 0.75rem;
    }
    
    .metric-mini {
        background: #f8fafc;
        padding: 0.75rem;
        border-radius: 6px;
        font-size: 0.85rem;
    }
    
    .metric-mini .metric-label {
        color: #6b7280;
        font-size: 0.75rem;
        margin-bottom: 0.25rem;
    }
    
    .metric-mini .metric-value {
        font-weight: 600;
        color: #1f2937;
    }
    
    /* Enhanced expander/collapse button styling */
    .streamlit-expander .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%) !important;
        border: 2px solid #bbf7d0 !important;
        border-radius: 8px !important;
        padding: 0.75rem 1rem !important;
        font-weight: 600 !important;
        color: #065f46 !important;
        transition: all 0.2s ease !important;
    }
    
    .streamlit-expander .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        color: white !important;
        border-color: #059669 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.3) !important;
    }
    
    /* Make the expand/collapse arrow more prominent */
    .streamlit-expander .streamlit-expanderHeader svg {
        width: 1.25rem !important;
        height: 1.25rem !important;
        color: #10b981 !important;
    }
    
    .streamlit-expander .streamlit-expanderHeader:hover svg {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Global variables
classifier = None

# H.264 conversion function for better video compatibility
def to_h264(input_path, output_path=None):
    """Convert video to H.264 format for better browser compatibility with enhanced error handling"""
    if output_path is None:
        base_name = os.path.splitext(input_path)[0]
        output_path = f"{base_name}_h264.mp4"
    
    try:
        # Check if ffmpeg is available
        ffmpeg_check = subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        print(f"✅ ffmpeg available: {ffmpeg_check.returncode == 0}")
        
        # Get video info first to check dimensions
        probe_cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', input_path
        ]
        
        try:
            probe_result = subprocess.run(probe_cmd, capture_output=True, check=True, text=True)
            import json
            video_info = json.loads(probe_result.stdout)
            
            # Find video stream
            video_stream = None
            for stream in video_info.get('streams', []):
                if stream.get('codec_type') == 'video':
                    video_stream = stream
                    break
            
            if video_stream:
                width = int(video_stream.get('width', 0))
                height = int(video_stream.get('height', 0))
                print(f"📐 原始尺寸: {width}x{height}")
                
                # Check if dimensions are odd (H.264 requirement: must be even)
                if width % 2 != 0 or height % 2 != 0:
                    # Force even dimensions
                    width = width + (width % 2)
                    height = height + (height % 2)
                    print(f"🔧 調整為偶數尺寸: {width}x{height}")
                    scale_filter = f"scale={width}:{height}"
                else:
                    scale_filter = None
                    print("✅ 尺寸已為偶數，無需調整")
                    
        except Exception as probe_error:
            print(f"⚠️ 無法獲取視頻信息，使用默認設置: {probe_error}")
            scale_filter = "scale=trunc(iw/2)*2:trunc(ih/2)*2"  # Force even dimensions
        
        # Build ffmpeg command with enhanced settings
        cmd = [
            'ffmpeg', '-y',  # -y to overwrite output file
            '-i', input_path,
            '-c:v', 'libx264',  # H.264 codec
            '-preset', 'fast',  # Fast encoding preset
            '-crf', '26',  # Higher CRF (lower quality) to reduce file size
            '-maxrate', '2M',  # Limit maximum bitrate
            '-bufsize', '4M',  # Buffer size
            '-movflags', '+faststart',  # Optimize for web streaming
            '-pix_fmt', 'yuv420p',  # Ensure compatibility
        ]
        
        # Add scale filter if needed (force even dimensions)
        if scale_filter:
            cmd.extend(['-vf', scale_filter])
        else:
            # Ensure even dimensions even if probe failed
            cmd.extend(['-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2'])
            
        cmd.append(output_path)
        
        print(f"🔄 執行 ffmpeg 命令: {' '.join(cmd[:8])}...")
        
        # Run conversion with detailed error capture
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ H.264 轉換成功")
            return output_path
        else:
            print(f"❌ ffmpeg 轉換失敗 (返回碼: {result.returncode})")
            print(f"📋 stderr: {result.stderr[:500]}...")  # Show first 500 chars of error
            print(f"📋 stdout: {result.stdout[:500]}...")
            return None
        
    except subprocess.CalledProcessError as e:
        print(f"❌ ffmpeg 命令執行失敗: {e}")
        print(f"📋 返回碼: {e.returncode}")
        if e.stderr:
            print(f"📋 錯誤輸出: {e.stderr.decode()[:500]}")
        return None
    except FileNotFoundError:
        print("❌ ffmpeg 未安裝或無法找到")
        return None
    except Exception as e:
        print(f"❌ H.264 轉換出現意外錯誤: {e}")
        return None

def ffprobe_json(path: str) -> dict:
    """Get detailed video metadata using ffprobe for browser compatibility analysis"""
    try:
        out = subprocess.check_output([
            "ffprobe", "-v", "error", "-print_format", "json",
            "-show_streams", "-show_format", path
        ])
        return json.loads(out.decode("utf-8"))
    except Exception as e:
        st.warning(f"ffprobe 失敗：{e}")
        return {}

class RiceClassifierStreamlit:
    def __init__(self, weights_path='models/best.pt', device='', img_size=640, conf_thres=0.25, iou_thres=0.45):
        """Initialize YOLOv7 Rice Classifier for Streamlit"""
        self.weights_path = weights_path
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        # Class names for rice quality
        self.names = ['white_rice', 'thi_rice', 'brown_rice', 'black_rice']
        self.colors = [(255, 255, 255), (255, 215, 0), (139, 69, 19), (0, 0, 0)]  # White, Gold, Brown, Black
        
        # Initialize device
        self.device = self._select_device(device)
        
        # Load model
        self.model = self._load_model()
        
        # Create output directory
        self.output_dir = Path('runs/detect')
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _select_device(self, device=''):
        """Select computation device"""
        if device.lower() == 'cpu':
            return torch.device('cpu')
        elif torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    
    def _load_model(self):
        """Load YOLOv7 model with trained weights"""
        try:
            # Add YOLOv7 to path
            yolo_path = Path('./yolov7')
            if yolo_path.exists():
                sys.path.insert(0, str(yolo_path))
            
            # Import YOLOv7 modules
            from models.experimental import attempt_load
            from utils.general import check_img_size, non_max_suppression, scale_coords
            from utils.plots import plot_one_box
            from utils.torch_utils import select_device
            from utils.datasets import letterbox
            
            # Store functions for later use
            self.attempt_load = attempt_load
            self.check_img_size = check_img_size
            self.non_max_suppression = non_max_suppression
            self.scale_coords = scale_coords
            self.plot_one_box = plot_one_box
            self.letterbox = letterbox
            
            # Load model
            model = attempt_load(self.weights_path, map_location=self.device)
            model.eval()
            
            # Check image size
            self.img_size = check_img_size(self.img_size, s=model.stride.max())
            
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    
    def preprocess_image(self, img):
        """Preprocess image for inference with letterbox"""
        # Apply letterbox resize (maintains aspect ratio with padding)
        # IMPORTANT: Keep ALL return values for proper coordinate transformation
        img_letterbox, ratio, pad = self.letterbox(img, self.img_size, stride=int(self.model.stride.max()), auto=True)
        
        # Convert BGR to RGB and transpose
        img_rgb = img_letterbox[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3xHxW
        img_rgb = np.ascontiguousarray(img_rgb)
        
        # Convert to tensor and normalize
        img_tensor = torch.from_numpy(img_rgb).to(self.device)
        img_tensor = img_tensor.float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
        
        # Return tensor and letterbox parameters for coordinate mapping
        return img_tensor, img_letterbox, ratio, pad
    
    def predict_video_frame(self, frame):
        """Run inference on a single video frame"""
        if self.model is None:
            return frame, []
        
        try:
            result_img, detections = self.predict_image(frame)
            return result_img if result_img is not None else frame, detections
        except Exception as e:
            print(f"[ERROR] predict_video_frame: {e}")
            return frame, []
    
    def process_video(self, video_path, output_path=None, progress_callback=None):
        """Process entire video file with progress tracking"""
        if self.model is None:
            return None, "Model not loaded"
        
        print(f"Starting video processing: {video_path}")
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Failed to open video: {video_path}")
                return None, f"Error opening video: {video_path}"
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"Video properties: {total_frames} frames, {fps} FPS, {width}x{height}")
            
            # Ensure even dimensions for VideoWriter compatibility (especially for H.264)
            original_width, original_height = width, height
            if width % 2 != 0:
                width = width + 1
                print(f"🔧 Adjusted width from {original_width} to {width} (must be even)")
            if height % 2 != 0:
                height = height + 1
                print(f"🔧 Adjusted height from {original_height} to {height} (must be even)")
            
            if width != original_width or height != original_height:
                print(f"📐 VideoWriter will use dimensions: {width}x{height} (adjusted from {original_width}x{original_height})")
            
            out = None
            if output_path:
                # Try different codecs optimized for cloud deployment and browser compatibility
                codecs_to_try = [
                    ('mp4v', '.mp4'),  # MPEG-4 - most widely supported
                    ('XVID', '.avi'),  # Xvid - good fallback
                    ('avc1', '.mp4'),  # H.264 - best quality but may not be available in cloud
                    ('H264', '.mp4'),  # Alternative H.264
                ]
                
                for codec, ext in codecs_to_try:
                    try:
                        # Adjust output path extension based on codec
                        current_output_path = output_path
                        if not output_path.endswith(ext):
                            current_output_path = output_path.rsplit('.', 1)[0] + ext
                        
                        fourcc = cv2.VideoWriter_fourcc(*codec)
                        # Use the adjusted even dimensions
                        out = cv2.VideoWriter(current_output_path, fourcc, fps, (width, height))
                        if out.isOpened():
                            print(f"✅ Successfully created video writer with codec: {codec}, dimensions: {width}x{height}, file: {current_output_path}")
                            # Update output_path to the successful one
                            output_path = current_output_path
                            break
                        else:
                            out.release()
                    except Exception as codec_error:
                        print(f"❌ Failed to create video writer with codec {codec}: {codec_error}")
                        continue
                
                if out is None or not out.isOpened():
                    print(f"Failed to create output video writer with any codec")
                    cap.release()
                    return None, f"Error creating output video: {output_path}"
            
            all_detections = []
            frame_count = 0
            start_time = time.time()
            
            print("Starting frame processing...")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                try:
                    # Process frame
                    result_frame, detections = self.predict_video_frame(frame)
                    
                    # Store detections with frame info
                    for det in detections:
                        det['frame'] = frame_count
                        det['timestamp'] = frame_count / fps
                    all_detections.extend(detections)
                    
                    # Write frame if output specified
                    if out is not None:
                        # Convert RGB back to BGR for video output
                        bgr_frame = cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR)
                        
                        # Ensure frame dimensions match VideoWriter dimensions
                        frame_h, frame_w = bgr_frame.shape[:2]
                        if frame_w != width or frame_h != height:
                            # Resize frame to match VideoWriter dimensions
                            bgr_frame = cv2.resize(bgr_frame, (width, height), interpolation=cv2.INTER_LINEAR)
                            if frame_count == 0:  # Log only once
                                print(f"🔧 Resizing frames from {frame_w}x{frame_h} to {width}x{height} for VideoWriter consistency")
                        
                        out.write(bgr_frame)
                    
                    frame_count += 1
                    
                    # Update progress if callback provided
                    if progress_callback and total_frames > 0:
                        progress = frame_count / total_frames
                        elapsed_time = time.time() - start_time
                        if frame_count > 0:
                            eta = (elapsed_time / frame_count) * (total_frames - frame_count)
                            progress_callback(progress, frame_count, total_frames, elapsed_time, eta)
                    
                    # Print progress every 10 frames
                    if frame_count % 10 == 0:
                        progress_pct = (frame_count / total_frames) * 100
                        print(f"Processed {frame_count}/{total_frames} frames ({progress_pct:.1f}%)")
                        
                except Exception as frame_error:
                    print(f"Error processing frame {frame_count}: {frame_error}")
                    frame_count += 1
                    continue
            
            cap.release()
            if out is not None:
                out.release()
            
            total_time = time.time() - start_time
            print(f"Video processing completed: {frame_count} frames in {total_time:.1f}s")
            return all_detections, f"Processed {frame_count} frames in {total_time:.1f}s"
            
        except Exception as e:
            print(f"Error in process_video: {e}")
            return None, f"Error processing video: {e}"
    
    def predict_image(self, image):
        """Run inference on a single image"""
        if self.model is None:
            return None, "Model not loaded"
        
        try:
            # Convert PIL Image to OpenCV format
            if isinstance(image, Image.Image):
                img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            else:
                img = image
            
            original_img = img.copy()
            
            # Preprocess with letterbox (now returns ratio and pad for alignment)
            img_tensor, img_letterbox, ratio, pad = self.preprocess_image(img)
            
            # Inference
            with torch.no_grad():
                pred = self.model(img_tensor, augment=False)[0]
                pred = self.non_max_suppression(pred, self.conf_thres, self.iou_thres)
            
            detection_results = []
            
            # Process detections
            for i, det in enumerate(pred):
                if len(det):
                    # Rescale boxes from img_size to im0 size (same as YOLOv7 detect.py)
                    det[:, :4] = self.scale_coords(img_tensor.shape[2:], det[:, :4], original_img.shape).round()
                    
                    # Draw boxes and labels
                    for *xyxy, conf, cls in reversed(det):
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        color = self.colors[int(cls)]
                        
                        # Draw bounding box on ORIGINAL image (not letterbox)
                        # Ensure coordinates are integers
                        x1, y1, x2, y2 = [int(x) for x in xyxy]
                        
                        # Draw rectangle directly using cv2 for consistency
                        cv2.rectangle(original_img, (x1, y1), (x2, y2), color, 2)
                        
                        # Add label
                        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        label_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
                        
                        # Draw label background
                        cv2.rectangle(original_img, 
                                    (x1, label_y - label_size[1] - 3),
                                    (x1 + label_size[0], label_y + 3),
                                    color, -1)
                        
                        # Draw label text
                        cv2.putText(original_img, label, (x1, label_y),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        # Store detection info
                        x1, y1, x2, y2 = [int(x) for x in xyxy]
                        detection_results.append({
                            'class': self.names[int(cls)],
                            'confidence': float(conf),
                            'bbox': [x1, y1, x2, y2]
                        })
            
            # Convert back to RGB for display
            # IMPORTANT: Return the original image with boxes drawn on it
            # Do NOT return the letterbox processed image
            result_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            
            return result_img_rgb, detection_results
            
        except Exception as e:
            return None, f"Error during inference: {e}"

@st.cache_resource
def load_classifier():
    """Load the rice classifier with caching"""
    weights_path = 'models/best.pt'
    if not os.path.exists(weights_path):
        return None, f"Model file not found at {weights_path}"
    
    try:
        # Suppress PyTorch warnings
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        
        print(f"Loading model from {weights_path}...")
        classifier = RiceClassifierStreamlit(weights_path=weights_path)
        if classifier.model is None:
            return None, "Failed to load model"
        print("Model loaded successfully!")
        return classifier, "Model loaded successfully!"
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, f"Error initializing classifier: {e}"

def predict_image_interface(image, conf_threshold, iou_threshold):
    """Main prediction interface"""
    global classifier
    
    if classifier is None:
        classifier_obj, status = load_classifier()
        if classifier_obj is None:
            return None, status
        classifier = classifier_obj
    
    # Update thresholds
    classifier.conf_thres = conf_threshold
    classifier.iou_thres = iou_threshold
    
    # Run prediction
    result_img, detections = classifier.predict_image(image)
    
    if result_img is None:
        return None, detections
    
    return result_img, detections

def create_detection_summary(detections):
    """Create a summary of detections"""
    if not detections:
        return "No rice grains detected. Try adjusting the confidence threshold."
    
    # Count by class
    class_counts = {'white_rice': 0, 'thi_rice': 0, 'brown_rice': 0, 'black_rice': 0}
    for det in detections:
        class_counts[det['class']] += 1
    
    total = len(detections)
    summary = f"**Total detected: {total} rice grains**\n\n"
    
    # Add percentages
    for class_name, count in class_counts.items():
        percentage = (count / total * 100) if total > 0 else 0
        emoji = {'white_rice': '⚪', 'thi_rice': '🟡', 'brown_rice': '🟤', 'black_rice': '⚫'}[class_name]
        summary += f"{emoji} **{class_name.capitalize()}**: {count} ({percentage:.1f}%)\n"
    
    return summary

class VideoTransformer(VideoProcessorBase):
    """Video transformer for webcam processing"""
    
    def __init__(self):
        self.classifier = None
        self.conf_threshold = 0.25
        self.iou_threshold = 0.45
    
    def set_classifier(self, classifier, conf_threshold, iou_threshold):
        self.classifier = classifier
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
    
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        if self.classifier is not None:
            # Update thresholds
            self.classifier.conf_thres = self.conf_threshold
            self.classifier.iou_thres = self.iou_threshold
            
            # Process frame
            result_img, detections = self.classifier.predict_video_frame(img)
            
            # Convert back to BGR for video output
            if result_img is not None:
                result_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
                return av.VideoFrame.from_ndarray(result_bgr, format="bgr24")
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def process_video_interface(video_file, conf_threshold, iou_threshold, progress_placeholder=None, status_placeholder=None):
    """Video processing interface with progress tracking"""
    global classifier
    
    # Add debug logging
    if status_placeholder:
        status_placeholder.info("🔧 載入模型中...")
    
    if classifier is None:
        classifier_obj, status = load_classifier()
        if classifier_obj is None:
            return None, status, None
        classifier = classifier_obj
    
    # Update thresholds
    classifier.conf_thres = conf_threshold
    classifier.iou_thres = iou_threshold
    
    temp_video_path = None
    output_video_path = None
    
    try:
        if status_placeholder:
            status_placeholder.info("💾 保存視頻文件...")
        
        # Save uploaded video to temp file
        video_bytes = video_file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(video_bytes)
            temp_video_path = tmp_file.name
        
        # Create output video path
        output_video_path = temp_video_path.replace('.mp4', '_processed.mp4')
        
        if status_placeholder:
            status_placeholder.info("📹 讀取視頻信息...")
        
        # Check video can be opened
        import cv2
        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            return None, "Error: 無法打開視頻文件", None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        if status_placeholder:
            status_placeholder.info(f"📊 視頻信息: {total_frames} frames, {fps:.1f} FPS")
        
        # Progress callback function
        def update_progress(progress, current_frame, total_frames, elapsed_time, eta):
            try:
                if progress_placeholder is not None:
                    progress_placeholder.progress(progress)
                if status_placeholder is not None:
                    mins_elapsed = int(elapsed_time // 60)
                    secs_elapsed = int(elapsed_time % 60)
                    mins_eta = int(eta // 60)
                    secs_eta = int(eta % 60)
                    
                    status_text = f"""🎬 處理進度: {current_frame}/{total_frames} frames ({progress:.1%})
⏱️ 已用時間: {mins_elapsed:02d}:{secs_elapsed:02d}
⏳ 預估剩餘: {mins_eta:02d}:{secs_eta:02d}
🔄 處理速度: {current_frame/elapsed_time:.1f} frames/sec"""
                    status_placeholder.info(status_text)
            except Exception as e:
                # Ignore progress update errors to prevent breaking the main process
                pass
        
        if status_placeholder:
            status_placeholder.info("🚀 開始處理視頻...")
        
        # Process video with progress tracking and output video
        try:
            detections, status = classifier.process_video(temp_video_path, output_path=output_video_path, progress_callback=update_progress)
        except TypeError as e:
            # Fallback for environments that don't support progress_callback
            if status_placeholder:
                status_placeholder.warning(f"⚠️ Progress callback not supported, falling back... ({str(e)})")
            detections, status = classifier.process_video(temp_video_path, output_path=output_video_path)
        
        # Check processed video file (but don't read bytes to memory yet)
        processed_video_bytes = None
        if output_video_path and os.path.exists(output_video_path):
            try:
                if status_placeholder:
                    status_placeholder.info("📤 視頻檔案準備完成...")
                
                # Check file size
                file_size = os.path.getsize(output_video_path)
                print(f"Output video file size: {file_size} bytes")
                
                if file_size > 0:
                    # Don't read the entire file into memory - just verify it exists and has content
                    # The video preview will use file path directly
                    print(f"Output video ready for preview: {output_video_path}")
                    if status_placeholder:
                        status_placeholder.success(f"✅ 視頻處理完成 (檔案大小: {file_size/1024/1024:.1f}MB)")
                else:
                    print("Output video file is empty")
                    if status_placeholder:
                        status_placeholder.error("❌ 輸出視頻檔案為空")
            except Exception as e:
                print(f"Error checking processed video: {e}")
                if status_placeholder:
                    status_placeholder.warning(f"⚠️ 視頻檔案檢查失敗: {e}")
        
        return detections, status, processed_video_bytes, output_video_path
        
    except Exception as e:
        print(f"Error in process_video_interface: {e}")
        return None, f"Error processing video: {str(e)}", None, None
    
    finally:
        # Clean up temp input files only - KEEP processed output video for preview/download
        if temp_video_path and os.path.exists(temp_video_path):
            try:
                os.unlink(temp_video_path)
                print(f"✅ Cleaned up temp input video: {temp_video_path}")
            except Exception as e:
                print(f"⚠️ Failed to cleanup temp input video: {e}")
        
        # DO NOT clean up output_video_path here - let Streamlit handle it
        # The processed video and H.264 converted video will be kept for preview
        # Streamlit will clean up temp files when the session ends
        if output_video_path:
            print(f"📁 Keeping processed video for preview: {output_video_path}")

def main():
    """Main Streamlit application"""
    
    # Sidebar Configuration
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-header">
            <h2 style="margin: 0; font-size: 1.25rem;">⚙️ Model Configuration</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Configuration Section
        with st.container():
            st.markdown("""
            <div class="sidebar-section">
                <h3>🎯 Detection Settings</h3>
            """, unsafe_allow_html=True)
            
            conf_threshold = st.slider(
                "Detection Confidence", 
                min_value=0.1, 
                max_value=1.0, 
                value=0.25, 
                step=0.05,
                help="Minimum confidence score for valid object detection"
            )
            
            iou_threshold = st.slider(
                "IoU Threshold", 
                min_value=0.1, 
                max_value=1.0, 
                value=0.45, 
                step=0.05,
                help="Non-maximum suppression threshold for overlapping detections"
            )
            
            st.markdown("</div>", unsafe_allow_html=True)
            
        # Model Status Section
        with st.container():
            st.markdown("""
            <div class="sidebar-section">
                <h3>📊 System Status</h3>
            """, unsafe_allow_html=True)
            
            # Check if model can be loaded
            classifier_obj, status = load_classifier()
            
            if classifier_obj is not None:
                st.success("✅ Model Loaded")
                st.markdown(f"""
                <div style="font-size: 0.875rem; color: rgba(255,255,255,0.8); margin-top: 1rem;">
                    <strong>Version:</strong> v2024.11.04<br>
                    <strong>Model:</strong> YOLOv7<br>
                    <strong>Classes:</strong> 4 Types<br>
                    <strong>Conf:</strong> {conf_threshold:.2f}<br>
                    <strong>IoU:</strong> {iou_threshold:.2f}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error("❌ Model Error")
                st.markdown(f"<small>{status}</small>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # System Analytics Section (收合式)
        with st.expander("📈 System Analytics", expanded=False):
            st.markdown("""
            <div class="sidebar-section">
                <div class="sidebar-header">📈 System Analytics</div>
                <div class="analytics-metrics">
            """, unsafe_allow_html=True)
            
            # System metrics
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                <div class="metric-mini">
                    <div class="metric-label">Model Status</div>
                    <div class="metric-value">{}</div>
                </div>
                """.format("✅ Ready" if classifier_obj is not None else "❌ Error"), unsafe_allow_html=True)
                
                st.markdown("""
                <div class="metric-mini">
                    <div class="metric-label">Framework</div>
                    <div class="metric-value">YOLOv7</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="metric-mini">
                    <div class="metric-label">Version</div>
                    <div class="metric-value">v2024.11.04</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class="metric-mini">
                    <div class="metric-label">Classes</div>
                    <div class="metric-value">4 Types</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Configuration display
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%);
                border: 1px solid #bbf7d0;
                border-radius: 8px;
                padding: 1rem;
                margin-top: 1rem;
                font-size: 0.875rem;
            ">
                <strong>🎯 Current Settings:</strong><br>
                Confidence: {conf_threshold:.2f} | IoU: {iou_threshold:.2f}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("</div></div>", unsafe_allow_html=True)
    
    # Main Dashboard Header
    st.markdown("""
    <div class="enterprise-header">
        <div class="main-title">🌾 YOLOv7 Rice Quality Classifier</div>
        <div class="main-subtitle">Enterprise-grade AI solution for rice quality analysis and classification</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Bento Box Dashboard Grid
    st.markdown("""
    <div class="bento-grid">
    """, unsafe_allow_html=True)
    
    # Check if model can be loaded (simplified for dashboard)
    classifier_obj, status = load_classifier()
    model_loaded = classifier_obj is not None
    
    # Bento Card 1: Image Analysis - 完整包裝
    with st.container():
        # 使用特殊的 CSS class 來模擬 bento card 外觀
        st.markdown("""
        <div class="bento-card-header large-card-header">
            <div class="bento-header">
                <div>
                    <div class="bento-title">📷 Image Analysis</div>
                    <div class="bento-subtitle">Upload and analyze rice grain images</div>
                </div>
                <div class="bento-badge">Core Feature</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if model_loaded:
            # Image Analysis Upload
            uploaded_file = st.file_uploader(
                "Upload Image File",
                type=['png', 'jpg', 'jpeg'],
                help="Select clear images of rice grains for AI analysis"
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(image, caption="Original Image", use_container_width=True)
                
                with col2:
                    with st.spinner("🔍 Analyzing..."):
                        result_img, detections = predict_image_interface(
                            image, conf_threshold, iou_threshold
                        )
                    
                    if result_img is not None:
                        st.image(result_img, caption="Analysis Results", use_container_width=True)
                        summary = create_detection_summary(detections)
                        st.markdown(summary)
                    else:
                        st.error(f"Analysis failed: {detections}")
        else:
            st.info("🚫 Model not loaded. Check sidebar for status.")
    
    # Bento Card 2: Video Processing
    with st.container():
        st.markdown("""
        <div class="bento-card-header large-card-header">
            <div class="bento-header">
                <div>
                    <div class="bento-title">📹 Video Processing</div>
                    <div class="bento-subtitle">Process videos with detection overlay</div>
                </div>
                <div class="bento-badge">Advanced</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if model_loaded:
            uploaded_video = st.file_uploader(
                "Upload Video File",
                type=['mp4', 'avi', 'mov'],
                help="Upload video for rice grain detection"
            )
            
            if uploaded_video is not None:
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.video(uploaded_video)
                
                with col2:
                    if st.button("🎬 Process Video", use_container_width=True):
                        with st.spinner("Processing video..."):
                            detections, status, _, output_path = process_video_interface(
                                uploaded_video, conf_threshold, iou_threshold
                            )
                        
                        if detections is not None:
                            st.success(f"✅ {status}")
                            if output_path and os.path.exists(output_path):
                                h264_path = to_h264(output_path)
                                if h264_path and os.path.exists(h264_path):
                                    st.video(h264_path)
                                    with open(h264_path, 'rb') as f:
                                        st.download_button(
                                            "📥 Download",
                                            data=f.read(),
                                            file_name="processed_video.mp4",
                                            mime="video/mp4"
                                        )
                        else:
                            st.error(f"❌ {status}")
        else:
            st.info("🚫 Model not loaded")
    
    # Bento Card 3: Live Camera
    with st.container():
        st.markdown("""
        <div class="bento-card-header large-card-header">
            <div class="bento-header">
                <div>
                    <div class="bento-title">📸 Live Camera</div>
                    <div class="bento-subtitle">Real-time detection via webcam</div>
                </div>
                <div class="bento-badge">Real-time</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if model_loaded:
            # Camera settings
            st.subheader("⚙️ Camera Settings")
            col1, col2 = st.columns(2)
            with col1:
                camera_conf = st.slider("Camera Confidence", 0.1, 1.0, conf_threshold, 0.05, key="camera_conf")
            with col2:
                camera_iou = st.slider("Camera IoU", 0.1, 1.0, iou_threshold, 0.05, key="camera_iou")
            
            # WebRTC streamer  
            video_transformer = VideoTransformer()
            if classifier is not None:
                video_transformer.set_classifier(classifier, camera_conf, camera_iou)
            
            st.info("🎥 Live camera detection with YOLOv7")
            webrtc_ctx = webrtc_streamer(
                key="rice-detection",
                video_processor_factory=lambda: video_transformer,
                rtc_configuration=RTCConfiguration(
                    ice_servers=[{"urls": ["stun:stun.l.google.com:19302"]}]
                ),
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )
            
            st.markdown("""
            **📱 How to use Live Camera:**
            1. Click "START" to begin camera detection
            2. Allow browser camera access when prompted
            3. Position rice grains in front of camera
            4. Adjust confidence/IoU thresholds as needed
            5. Click "STOP" when finished
            """)
            
            # Fallback simple camera
            st.subheader("📷 Alternative: Photo Capture")
            simple_camera = st.camera_input("📸 Take a photo (if WebRTC doesn't work)")
            if simple_camera is not None:
                image = Image.open(simple_camera)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="Captured", use_container_width=True)
                
                with col2:
                    with st.spinner("Analyzing..."):
                        result_img, detections = predict_image_interface(
                            image, conf_threshold, iou_threshold
                        )
                    
                    if result_img is not None:
                        st.image(result_img, caption="Results", use_container_width=True)
                        st.markdown(create_detection_summary(detections))
                    else:
                        st.error("Analysis failed")
        else:
            st.info("🚫 Model not loaded")
    
    # Bento Card 4: Batch Analysis
    with st.container():
        st.markdown("""
        <div class="bento-card-header large-card-header">
            <div class="bento-header">
                <div>
                    <div class="bento-title">📊 Batch Analysis</div>
                    <div class="bento-subtitle">Process multiple images at once</div>
                </div>
                <div class="bento-badge">Batch</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if model_loaded:
            uploaded_files = st.file_uploader(
                "Upload Multiple Images",
                type=['png', 'jpg', 'jpeg'],
                accept_multiple_files=True
            )
            
            if uploaded_files:
                st.write(f"📁 {len(uploaded_files)} files uploaded")
                
                if st.button("🔄 Process All", use_container_width=True):
                    progress_bar = st.progress(0)
                    results = []
                    
                    for i, file in enumerate(uploaded_files):
                        image = Image.open(file)
                        result_img, detections = predict_image_interface(
                            image, conf_threshold, iou_threshold
                        )
                        
                        if result_img is not None:
                            results.append({
                                'filename': file.name,
                                'detections': len(detections),
                                'white_rice': len([d for d in detections if d['class'] == 'white_rice']),
                                'thi_rice': len([d for d in detections if d['class'] == 'thi_rice']),
                                'brown_rice': len([d for d in detections if d['class'] == 'brown_rice']),
                                'black_rice': len([d for d in detections if d['class'] == 'black_rice'])
                            })
                        
                        progress_bar.progress((i + 1) / len(uploaded_files))
                    
                    if results:
                        import pandas as pd
                        df = pd.DataFrame(results)
                        st.dataframe(df, use_container_width=True)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("⚪ White", df['white_rice'].sum())
                        with col2:
                            st.metric("🟡 Thi", df['thi_rice'].sum())
                        with col3:
                            st.metric("🟤 Brown", df['brown_rice'].sum())
                        with col4:
                            st.metric("⚫ Black", df['black_rice'].sum())
        else:
            st.info("🚫 Model not loaded")
    
    
    # Close Bento Grid
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()