from flask import Flask, render_template, request

import os

from utils.pdf_loader import load_pdf
from utils.vector_store import create_vector_store

from utils.agents import (
    trip_planner_agent,
    hotel_agent,
    budget_agent
)

app = Flask(__name__)

vectorstore = None


# Home Page
@app.route("/")
def home():

    return render_template("index.html")



# Upload PDF
@app.route("/upload", methods=["POST"])
def upload():

    global vectorstore

    pdf = request.files["pdf"]

    # Save PDF
    pdf_path = os.path.join(
        "uploads",
        pdf.filename
    )

    pdf.save(pdf_path)


    # Read PDF
    text = load_pdf(pdf_path)


    # Create FAISS Vector Store
    vectorstore = create_vector_store(text)


    return render_template(
        "index.html",
        response="PDF Uploaded Successfully!"
    )



# Plan Trip
@app.route("/plan", methods=["POST"])
def plan():

    global vectorstore


    # Check PDF Upload
    if vectorstore is None:

        return render_template(
            "index.html",
            response="Please upload PDF first!"
        )


    # Get User Input
    location = request.form["location"]

    budget = request.form["budget"]

    days = request.form["days"]


    # Retrieve Similar Content
    docs = vectorstore.similarity_search(
        location
    )


    # Create Context
    context = ""

    for doc in docs:

        context += doc.page_content


    # AI Agents
    trip_plan = trip_planner_agent(
        location,
        days,
        context
    )

    hotels = hotel_agent(
        location,
        budget
    )

    budget = budget_agent(
        location,
        budget,
        days
    )


    # Final Response

    final_response = f"""

    <h2>Trip Plan</h2>

    {trip_plan}

    <h2>Hotel Suggestions</h2>

    {hotels}

    <h2>Budget Plan</h2>

    {budget}

    """


    return render_template(
        "index.html",
        response=final_response
    )



if __name__ == "__main__":

    app.run(debug=True)