# Interview Questions

## Overview
Generates interview questions and case studies for a given topic


## Setup Instructions
1. Clone the repository:
   ```
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```
   cd haystack-project
   ```
3. Install the required dependencies:
   ```
   uv sync
   ```

## Usage
To generate questions, execute the following command:
```
python src/questions.py
```

To generate case studies, execute the following command:
```
python src/case_studies.py
```

## TODO

No particular order

* Add config file control, rather than modifying the scripts
* Case studies v2.0
   - Separate context, question, and answers into separate LLMs+prompts
* Refactor to make use of better pipeline functionality
* Add guardrails for LLM responses, including:
   - Filtering empty lines
   - Self-correcting loop
* Play with the rephrasing prompt to get better results
* Create some C4 diagrams

## License
This project is licensed under the MIT License. See the LICENSE file for details.