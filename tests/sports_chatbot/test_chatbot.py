from sports_chatbot.chatbot import SportsChatbot


def test_chatbot():
    chatbot = SportsChatbot("./data/sports_chatbot/Day_1.csv")
    context = chatbot.get_contextual_response(
        "Which team is playing the Red Sox? Please remembe to always answer fully in a nic and educated manner."
    )
    assert "canucks" in context.lower()
    query = chatbot.query(
        "Which team is playing the Red Sox? Please remember "
        + "to always answer fully in a nic and educated manner. Ex:  "
        + "'The {team_here} team is playing against {other_team} today!'"
    )
    assert "canucks" in query["answer"].lower()
