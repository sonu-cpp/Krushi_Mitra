# Krushi Mitra 🌾
Krushi Mitra is an AI-Powered agricultural decision support system that is built to assist Indian farmers. Recommends crops, detects plant diseases and suggests fertilizers. 

## Disclaimer 
The following project is an academic project and is just in a working demo version. Please consult any local agricultural officers before implementing this in real life scenarios.

## Features
- 🌱 Crop Recommendation 
- 🧪 Fertilizer Recommendation
- 🔬 Plant Disease Detection
- 🌐 Multilingual: English, Hindi (हिंदी), Odia (ଓଡ଼ିଆ)

## Setup & Run

### 1. Install Python

- Recommended Version: **Python 3.10 or above**

### 2. Install Git

- Git is required to clone the repository.
- In case of Windows users, Github desktop can be used to clone the repository locally.

### 3. Verify Installation

Open terminal / command prompt and run:

```bash
python --version
pip --version
```

### 4. Clone the repository

For Github desktop users
Open this link in your browser

```bash
https://github.com/sonu-cpp/Krushi_Mitra
```
Click on the top right -> <>Code -> Click on "Open with GitHub Desktop" 
<img width="671" height="590" alt="image" src="https://github.com/user-attachments/assets/737e1590-7901-4373-97d1-21d7f27adb3e" />

For git users 
Open Terminal/Command Prompt

```bash
git clone https://github.com/sonu-cpp/Krushi_Mitra
```
  
### Activate the Virtual Environment

For Windows users: 
-  The Cloned repositories are usually present in Documents -> GitHub Desktop 
-  Right click -> Open with Terminal
-  Run the following commands

#### Create a Python Virtual Enviroment
```bash
python3 -m venv venv
```

#### Activate the Virtual Enviroment
#### Windows (PowerShell)

```bash
.\venv\Scripts\Activate.ps1
```

#### Windows (Command Prompt)

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```
### Install Project Dependencies

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

### Run the Application

Start the Streamlit application using:

```bash
streamlit run App.py
```
