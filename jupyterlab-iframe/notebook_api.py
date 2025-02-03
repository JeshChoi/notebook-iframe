from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import os
import json

app = Flask(__name__)
CORS(app)  # Allow all origins

NOTEBOOK_PATH = "/home/jovyan/work"

@app.route('/set_notebook', methods=['POST'])
def set_notebook():
    try:
        data = request.json
        notebook_name = data.get('notebookName', 'example.ipynb')
        notebook_data = data.get('notebookData')

        if not notebook_data:
            return jsonify({"error": "Notebook data is required"}), 400

        # Save the notebook JSON file
        notebook_file_path = os.path.join(NOTEBOOK_PATH, notebook_name)
        with open(notebook_file_path, 'w', encoding='utf-8') as f:
            json.dump(notebook_data, f, indent=4)

        return jsonify({
            "message": "Notebook saved successfully",
            "notebookPath": notebook_file_path
        })

    except Exception as e:
        print(f"Unexpected server error: {e}")
        return jsonify({"error": f"Unexpected server error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
