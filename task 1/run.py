
#!/usr/bin/env python3
from app import app

if __name__ == "__main__":
    print("Starting Sentiment Analysis web app on http://127.0.0.1:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
