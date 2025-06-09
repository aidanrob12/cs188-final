# Robosuite Robot with Text Commands

### Intro to Robotics CS 188: Final Project  
**Team Members**  
- Aidan Robinson
- Arthur Zhou
- Olana Abraham

---

## 🧠 Project Overview

This project enables a robot in the **Robosuite** simulation environment to be controlled entirely through **natural language commands**. The robot's task is to move a cube from point A to point B based on short, user-issued text instructions. These commands can be **imperative** (e.g., "move left") or **declarative** (e.g., "grab the cube"), and are parsed using Natural Language Processing (NLP) techniques.

Our main objective was to build a robust NLP-to-action pipeline where the robot can understand different phrasings for the same command, such as "go left" and "move left," and execute them reliably using a PID controller.

---

## 🧪 Project Goals and Success Criteria

- ✅ Interpret and map a variety of natural language commands to predefined robotic actions.
- ✅ Move a cube from a start location to a target location **using only text input**.
- ✅ Ensure **90%+ success rate** on complete tasks without manual intervention.
- ✅ Handle unknown or malformed commands gracefully (error handling).
  
---

## 🛠️ Technical Approach

- **Robosuite**: Used as the simulation environment for robotic manipulation.
- **PID Controller**: Guides the robot’s motion to perform low-level control actions.
- **Natural Language Processing (NLP)**: Converts user input into actionable commands using:
  - `SpaCy` / `NLTK` / `HuggingFace Transformers` for intent recognition
  - Command mapping strategies (intent normalization, synonyms, etc.)
- **Policy Engine**: Maps high-level intents to specific sequences of PID-controlled actions.

---

## 📁 File Structure

| File              | Description |
|-------------------|-------------|
| `environment.py`  | Initializes and manages the Robosuite simulation environment. |
| `pid.py`          | PID controller for robot movement and manipulation tasks. |
| `nlp.py`          | NLP interface to parse text commands and infer intent. |
| `policies.py`     | Contains policies for translating parsed commands into robotic actions. |
| `final_project.pdf` | Full project report detailing design, methodology, and results. |
| `.gitignore`      | Standard Git exclusions. |
| `README.md`       | This documentation file. |

---

## 🚀 Getting Started

### ✅ Requirements

- Python 3.8+
- `robosuite` (install from https://github.com/ARISE-Initiative/robosuite)
- NLP Libraries:
  - `spaCy`
  - `transformers`
  - `nltk`

### 📦 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/cs188-final-main.git
cd cs188-final-main

# Create a virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

## NLP Packages
- https://anaconda.org/conda-forge/sentence-transformers

## Installation
`conda install conda-forge::sentence-transformers`
- Note: you may have to run `conda install numpy=2.1` if you get the Numpy version error