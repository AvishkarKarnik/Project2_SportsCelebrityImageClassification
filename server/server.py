from flask import Flask, jsonify, request
import util

# Python flask server is like a micro server/very lightweight web server that you can write in python

app = Flask(__name__)


@app.route('/classify_image', methods=['GET', 'POST'])
# 'GET method is used to send message to server and server returns data.
# 'POST' method is used to return HTML form of data to the server.
def classify_image():
    image_data = request.form['image_data']
    response = jsonify(util.classify_image(image_data))

    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

# This Python code sets up a simple Flask web application for sports celebrity image classification. It defines a single
# endpoint /classify_image that can be accessed using both GET and POST methods to classify sports celebrity images. The
# classification process is carried out using a utility function from the util module.
# Here's a breakdown of the code:
# 1. app = Flask(__name__):
# This line creates a new Flask web application instance and assigns it to the variable app. Flask is a lightweight web
# framework used for building web applications in Python.
# 2. @app.route('/classify_image', methods=['GET', 'POST']):
# This is a decorator that associates the function classify_image() with the URL path /classify_image and specifies that
# it can handle both GET and POST HTTP methods. When a client sends a request to the server with the /classify_image
# path, Flask will call the classify_image() function to handle the request.
# 3. def classify_image():
# This defines the classify_image() function, which will be called when a request is made to the /classify_image
# endpoint.
# 4. image_data = request.form['image_data']:
# Inside the classify_image() function, it retrieves the value of the 'image_data' field from the request's form data.
# This is how the client will send the image data to the server for classification.
# 5. response = jsonify(util.classify_image(image_data)):
# The image data retrieved from the request is then passed to the classify_image() function in the util module. The
# jsonify() function converts the classification results into a JSON response.
# 6. response.headers.add('Access-Control-Allow-Origin', '*'):
# This line sets the Access-Control-Allow-Origin header in the response, allowing the server to respond to requests from
# any origin. This is often used for handling Cross-Origin Resource Sharing (CORS) to ensure that the server can be
# accessed from different domains.
# 7. return response:
# The JSON response containing the classification results is returned to the client.
# 8. if __name__ == '__main__'::
# This block checks if the script is being run directly (as opposed to being imported as a module). It is a common
# practice to ensure that the following code is only executed when the script is run directly and not when it's imported
# as a module.
# 9. print("Starting Python Flask Server For Sports Celebrity Image Classification"):
# A message is printed to the console indicating that the Flask server is starting up.
# 10. util.load_saved_artifacts():
# This line calls a function load_saved_artifacts() from the util module. This function likely loads any pre-trained
# machine learning model or other necessary artifacts required for the sports celebrity image classification.
# 11. app.run(port=5000):
# Finally, the Flask application is started with app.run(), and it will listen for incoming requests on port 5000. The
# server will be up and running, ready to classify sports celebrity images as requested.


if __name__ == '__main__':
    print("Starting Python Flask Server For Sports Celebrity Image Classification")
    util.load_saved_artifacts()
    app.run(port=5000)
