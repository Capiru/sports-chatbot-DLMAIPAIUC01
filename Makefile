hooks:  # Install pre-commit hooks
	poetry run pre-commit install


pre-commit: # Runs the pre-commit over entire repo
	poetry run pre-commit run --all-files


install:  # Install packages locally using poetry
	poetry install --all-extras


check:  # Run unit tests
	poetry run py.test tests/sports_chatbot

app:  # Runs the streamlit chatbot app
	poetry run streamlit run src/chat_app/chat.py
