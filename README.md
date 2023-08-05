# Chatbot Deployment with Flask

## Description

This project demonstrates the deployment of a chatbot using Flask. It allows users to interact with the chatbot and receive responses generated by an AI model.
## Dependencies

To run this project, you need the following dependencies:

- **flask**
- **python-dotenv**

## Setting up the Project

Follow these steps to set up the project:

1. Download the project code.

2. Generate a Hugging Face token from [Hugging Face website](https://huggingface.co/settings/tokens).

3. Create a `.env` file in the project directory.

4. Paste the Hugging Face token in the `.env` file as follows:

`HUGGINGFACE_TOKEN=<YOUR_HUGGING_FACE_TOKEN>`

5. Install the required dependencies by running the following commands:

`pip install flask dotenv`

6. Run the `server.py` file using the following command:

`python server.py`


7. Once the server is running, access the chatbot by opening your web browser and navigating to `http://localhost:5000`.

8. Interact with the chatbot and enjoy!

Feel free to explore and modify the code to suit your specific requirements.

Happy chatting! 😊

